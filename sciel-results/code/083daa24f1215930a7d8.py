"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
import time

def construct():
    N = 256
    # 1. Identify valid candidates (a such that 2a is not a square)
    valid_values = []
    for a in range(1, N + 1):
        if not _is_square(2 * a):
            valid_values.append(a)
    
    M = len(valid_values)
    if M == 0:
        return [1]
        
    # 2. Build conflict graph
    squares = set()
    k = 1
    while k*k <= 2*N:
        squares.add(k*k)
        k += 1
        
    # Adjacency matrix or list of lists
    # Since M is small (~170), we can use bitmasks (integers) for speed
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(valid_values)}
    
    # Bitmasks for neighbors
    # adj[i] is a bitmask where bit j is 1 if i and j conflict
    adj = [0] * M
    for i in range(M):
        vi = valid_values[i]
        for j in range(i + 1, M):
            vj = valid_values[j]
            if (vi + vj) in squares:
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Bitset Branch and Bound for Maximum Independent Set
    # State: candidates (int bitmask), current_set (list of indices), current_size
    best_size = 0
    best_set = []
    
    # Heuristic: Order vertices by degree descending?
    # For B&B, selecting the vertex with highest degree in the current subgraph is often good.
    # We can just iterate or pick max degree.
    
    # Initial Greedy to get a good lower bound
    # Sort by degree
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    
    # Simple Greedy Initialization
    init_candidates = (1 << M) - 1
    init_set = []
    for v in order:
        if (init_candidates & (1 << v)) == 0:
            continue
        # Check if v conflicts with any in init_set
        # Actually, if we maintain a mask of "available" nodes, we just pick from available.
        # Greedy:
        avail = (1 << M) - 1
        greedy_set = []
        # We want to pick nodes that don't conflict.
        # Standard greedy: pick v, remove v and its neighbors from avail.
        for v in order:
            if (avail & (1 << v)) != 0:
                greedy_set.append(v)
                # Remove v and neighbors
                avail &= ~(adj[v] | (1 << v))
        
    best_size = len(greedy_set)
    best_set = greedy_set[:]
    
    # B&B Search
    # To prevent timeout, we can limit recursion or just rely on the fact that M=170 is small enough for B&B with good heuristics.
    # However, worst case is exponential. We should add a time check or a node limit.
    # Given 60s limit, and M~170, B&B might be risky if the graph is dense.
    # Let's add a simple time limit check inside the recursion or use a more robust heuristic.
    # Actually, for N=256, the graph is not extremely dense.
    # Let's try a recursive B&B with a node limit or time limit.
    
    start_time = time.time()
    node_count = 0
    MAX_NODES = 500000 # Safety limit
    
    def popcount(x):
        return bin(x).count('1')
        
    def search(candidates, current_set, current_size):
        nonlocal best_size, best_set, node_count, start_time
        
        if time.time() - start_time > 55: # Safety margin
            return
            
        node_count += 1
        if node_count > MAX_NODES:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Upper bound: current_size + number of bits in candidates
        if current_size + popcount(candidates) <= best_size:
            return
            
        # Select a vertex from candidates
        # Heuristic: Pick vertex with max degree in the induced subgraph?
        # Or just the first one?
        # Let's pick the vertex with max degree in the *original* graph restricted to candidates?
        # Or simply the lowest index for simplicity, but max degree is better.
        
        # To keep it fast, let's just pick the first available vertex.
        # Or better: pick vertex v in candidates that maximizes the number of conflicts within candidates.
        # This is expensive to compute every time.
        # Let's use a static order based on global degree, but only consider candidates.
        
        # Find v in candidates
        # Iterate through global order, find first one in candidates
        v = -1
        for idx in order:
            if (candidates & (1 << idx)) != 0:
                v = idx
                break
                
        if v == -1:
            return # Should not happen if candidates != 0
            
        bit_v = 1 << v
        
        # Branch 1: Include v
        # New candidates: candidates minus v and its neighbors
        new_candidates_include = candidates & ~(adj[v] | bit_v)
        search(new_candidates_include, current_set + [v], current_size + 1)
        
        # Branch 2: Exclude v
        new_candidates_exclude = candidates & ~bit_v
        search(new_candidates_exclude, current_set, current_size)
        
    # Run search
    # Initial candidates is all valid nodes
    all_candidates = (1 << M) - 1
    search(all_candidates, [], 0)
    
    # Convert best_set (indices) to values
    result = [valid_values[i] for i in best_set]
    
    if not result:
        return [1]
        
    # Verify
    if _verify(result):
        return sorted(result)
        
    # Fallback if B&B failed or returned invalid (shouldn't happen)
    return [1]

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify(s):
    s_set = set(s)
    if not s_set:
        return False
    for a in s_set:
        if _is_square(2 * a):
            return False
    for i, a in enumerate(s):
        for b in s[i+1:]:
            if _is_square(a + b):
                return False
    return True
# EVOLVE-BLOCK-END
