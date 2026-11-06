import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 511
    # Nodes are 1..511
    # We want to maximize sum 1/(512-a)
    # Sort nodes by weight descending.
    # a=511 has weight 1/1=1.0
    # a=1 has weight 1/511
    # So order is 511, 510, ..., 1
    
    # Map sorted index to original value
    # sorted_nodes[i] = 511 - i
    # weights[i] = 1.0 / (512 - (511 - i)) = 1.0 / (1 + i)
    
    weights = [1.0 / (1 + i) for i in range(N)]
    
    # Precompute conflict masks
    # conflict_masks[i] is a bitmask of indices j such that gcd(sorted_nodes[i], sorted_nodes[j]) > 1
    # Note: i and j are indices in 0..510
    # sorted_nodes[i] = 511 - i
    
    sorted_nodes = [511 - i for i in range(N)]
    
    conflict_masks = [0] * N
    for i in range(N):
        ni = sorted_nodes[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            nj = sorted_nodes[j]
            if gcd(ni, nj) > 1:
                mask |= (1 << j)
        conflict_masks[i] = mask
        
    # Precompute sum tables for fast upper bound calculation
    # Chunk size 16
    chunk_size = 16
    num_chunks = (N + chunk_size - 1) // chunk_size
    sum_tables = []
    for c in range(num_chunks):
        table = [0.0] * (1 << chunk_size)
        for m in range(1 << chunk_size):
            s = 0.0
            for k in range(chunk_size):
                if m & (1 << k):
                    idx = c * chunk_size + k
                    if idx < N:
                        s += weights[idx]
            table[m] = s
        sum_tables.append(table)
        
    def get_sum(mask, start_idx):
        # Sum weights of set bits in mask at indices >= start_idx
        total = 0.0
        # Mask out bits below start_idx
        if start_idx > 0:
            mask &= ~((1 << start_idx) - 1)
        if mask == 0:
            return 0.0
            
        start_chunk = start_idx // chunk_size
        for c in range(start_chunk, num_chunks):
            # Extract 16 bits from mask at position c*16
            shift = c * chunk_size
            chunk_mask = (mask >> shift) & ((1 << chunk_size) - 1)
            total += sum_tables[c][chunk_mask]
        return total

    best_weight = 0.0
    best_set = []
    
    # Initial greedy solution to set a good lower bound
    # Greedy: pick highest weight available
    avail_mask = (1 << N) - 1
    current_set = []
    current_weight = 0.0
    for i in range(N):
        if avail_mask & (1 << i):
            current_set.append(i)
            current_weight += weights[i]
            avail_mask &= ~conflict_masks[i]
    best_weight = current_weight
    best_set = current_set[:]
    
    # B&B
    # To avoid recursion depth issues, we might need to be careful, but N=511 is small enough for depth if we prune well?
    # Actually, depth is N. 511 is fine for Python recursion limit if set high, but stack overflow is a risk.
    # We'll use recursion and set limit.
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(idx, mask, current_w, current_indices):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        if idx >= N:
            if current_w > best_weight:
                best_weight = current_w
                best_set = current_indices[:]
            return
            
        # Upper bound
        ub = current_w + get_sum(mask, idx)
        if ub <= best_weight:
            return
            
        # Find next available node >= idx
        # m = mask & ~((1 << idx) - 1)
        m = mask
        if idx > 0:
            m &= ~((1 << idx) - 1)
            
        if m == 0:
            # No more nodes available
            if current_w > best_weight:
                best_weight = current_w
                best_set = current_indices[:]
            return
            
        # Get lowest set bit index in m
        # v = (m & -m).bit_length() - 1
        v = (m & -m).bit_length() - 1
        
        # Branch 1: Include v
        # Check if including v can beat best
        # new_w = current_w + weights[v]
        # new_mask = mask & ~conflict_masks[v]
        # We can do a quick check: new_w + get_sum(new_mask, v+1) > best_weight?
        
        new_mask_include = mask & ~conflict_masks[v]
        # Pruning check for include branch
        # This get_sum is expensive. Only do it if current_w + weights[v] > best_weight? No.
        # Let's just recurse. The ub check inside will handle it.
        
        current_indices.append(v)
        dfs(v + 1, new_mask_include, current_w + weights[v], current_indices)
        current_indices.pop()
        
        # Branch 2: Exclude v
        new_mask_exclude = mask & ~(1 << v)
        dfs(v + 1, new_mask_exclude, current_w, current_indices)
        
    # Start search
    # Initial mask is all ones
    initial_mask = (1 << N) - 1
    dfs(0, initial_mask, 0.0, [])
    
    # Convert indices back to values
    # sorted_nodes[i] = 511 - i
    result = [sorted_nodes[i] for i in best_set]
    
    return result
