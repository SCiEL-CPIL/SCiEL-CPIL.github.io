import time
import math

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    N = 256
    # Squares up to 2*256 = 512
    squares = set()
    for i in range(1, 23):
        squares.add(i*i)
    
    # 1. Filter vertices where 2*v is a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    M = len(valid_vertices)
    
    # Map original value to index
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitset of indices j such that valid_vertices[i] + valid_vertices[j] is a square
    adj = [0] * M
    for i in range(M):
        vi = valid_vertices[i]
        mask = 0
        for j in range(M):
            if i != j:
                if vi + valid_vertices[j] in squares:
                    mask |= (1 << j)
        adj[i] = mask
        
    all_bits = (1 << M) - 1
    
    # 3. Greedy Independent Set to get initial best
    def greedy_is(cand_bits):
        size = 0
        remaining = cand_bits
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            best_v = -1
            min_deg = M + 1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            size += 1
            # Remove v and its neighbors
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return size

    best_size = greedy_is(all_bits)
    
    # Store the actual set for the best solution found
    # We need to track the set, not just the size.
    # For efficiency, we'll store the bitset of the best solution.
    best_set_bits = 0
    
    # Re-run greedy to get the bits
    def greedy_is_bits(cand_bits):
        bits = 0
        remaining = cand_bits
        while remaining:
            best_v = -1
            min_deg = M + 1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            bits |= (1 << best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return bits

    best_set_bits = greedy_is_bits(all_bits)
    best_size = best_set_bits.bit_count()
    
    # 4. Branch and Bound
    # We need a helper to compute the clique cover bound
    def clique_cover_bound(cand_bits):
        count = 0
        remaining = cand_bits
        while remaining:
            count += 1
            # Find a maximal clique in G[remaining]
            # Greedy: pick a vertex, then only its neighbors
            # To maximize clique size (and thus minimize count), we should pick wisely?
            # Actually, any maximal clique works for the bound, but larger cliques give tighter bounds.
            # Let's pick the vertex with max degree in remaining to form a large clique.
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
                
            clique = 1 << best_v
            candidates = remaining & adj[best_v]
            
            while candidates:
                # Pick next vertex for clique
                # Pick max degree in candidates
                next_v = -1
                max_deg_c = -1
                temp = candidates
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = (adj[v] & candidates).bit_count()
                    if deg > max_deg_c:
                        max_deg_c = deg
                        next_v = v
                    temp ^= lsb
                
                if next_v == -1:
                    break
                    
                clique |= (1 << next_v)
                candidates &= adj[next_v]
            
            remaining &= ~clique
            
        return count

    # Recursive search
    # To avoid recursion depth issues and overhead, we can use a stack or just recursion with sys.setrecursionlimit
    # Given M ~ 240, depth is manageable.
    
    import sys
    sys.setrecursionlimit(10000)
    
    search_calls = 0
    
    def search(cand_bits, current_size, current_bits):
        nonlocal best_size, best_set_bits, search_calls
        
        if time.time() - start_time > time_limit:
            return
            
        search_calls += 1
        
        if cand_bits == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_bits = current_bits
            return
            
        # Upper bound using clique cover
        ub = current_size + clique_cover_bound(cand_bits)
        
        if ub <= best_size:
            return
            
        # Pick a vertex to branch on
        # Heuristic: Max degree in cand_bits
        best_v = -1
        max_deg = -1
        temp = cand_bits
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand_bits).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates: cand_bits minus best_v and its neighbors
        new_cand_1 = cand_bits & ~v_bit & ~adj[best_v]
        search(new_cand_1, current_size + 1, current_bits | v_bit)
        
        # Branch 2: Exclude best_v
        # New candidates: cand_bits minus best_v
        new_cand_2 = cand_bits & ~v_bit
        search(new_cand_2, current_size, current_bits)

    search(all_bits, 0, 0)
    
    # Convert best_set_bits back to list of integers
    result = []
    for i in range(M):
        if best_set_bits & (1 << i):
            result.append(valid_vertices[i])
            
    # Ensure non-empty
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
