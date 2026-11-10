# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Construct the 82-element base solution using CRT insight
    # QRs mod 5 are {1, 4}. Non-QRs are {0, 2, 3}.
    # Independent set in the constraint graph (C5) is size 2.
    # Picking residues {0, 2} mod 5 ensures no forbidden differences from mod 5 part.
    base_set = [x for x in range(m) if x % 5 in (0, 2)]
    best_set = base_set[:]
    best_size = len(best_set)
    
    # 3. Prepare for Branch and Bound on Maximum Clique in Complement Graph
    # adj[i] is a bitmask of j such that (i-j) % m is NOT in forbidden.
    adj = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d not in forbidden:
                mask |= (1 << j)
        adj[i] = mask
        
    # Order vertices by degree in the allowed graph (descending)
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    order = sorted(range(m), key=lambda x: degrees[x], reverse=True)
    verts = order
    
    start_time = time.time()
    time_limit = 590  # 10s margin
    
    def popcount(x):
        return bin(x).count('1')
        
    def expand(candidates_mask, size, current_clique):
        nonlocal best_set, best_size
        
        if time.time() - start_time > time_limit:
            return False
            
        # Pruning: even if we add all remaining candidates, we won't beat best
        if size + popcount(candidates_mask) <= best_size:
            return True
            
        if candidates_mask == 0:
            if size > best_size:
                best_size = size
                best_set = current_clique[:]
            return True
            
        # Heuristic: Pick the vertex with highest degree in the candidate set
        # To keep it fast, we pick the first one in 'verts' that is available.
        # A better heuristic: pick v that maximizes popcount(candidates & adj[v])
        # But that is O(N) per step. With N=205, it's acceptable.
        
        v = -1
        max_deg = -1
        # Iterate through sorted vertices to find a good candidate
        # We check all available vertices to pick the one with highest connectivity
        for u in verts:
            if candidates_mask & (1 << u):
                deg = popcount(candidates_mask & adj[u])
                if deg > max_deg:
                    max_deg = deg
                    v = u
                    
        if v == -1:
            return True
            
        # Option 1: Include v
        new_candidates = candidates_mask & adj[v]
        current_clique.append(v)
        
        if time.time() - start_time > time_limit:
            current_clique.pop()
            return False
            
        expand(new_candidates, size + 1, current_clique)
        
        current_clique.pop()
        
        # Option 2: Exclude v
        candidates_mask ^= (1 << v)
        
        if time.time() - start_time > time_limit:
            return False
            
        expand(candidates_mask, size, current_clique)
        
        return True

    # Initial candidates: all vertices
    initial_mask = (1 << m) - 1
    
    # Run search
    expand(initial_mask, 0, [])
    
    result = np.array(best_set, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
