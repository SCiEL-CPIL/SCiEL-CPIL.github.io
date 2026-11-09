import time
import math

def construct():
    t0 = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    N = 256
    # Build compatibility bitsets: compat[i] has bit j set iff i and j can coexist
    # Can coexist means (a*b) % (a+b) != 0
    compat = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j:
                a, b = i, j
                if (a * b) % (a + b) != 0:
                    compat[i] |= (1 << j)
    
    # Maximum clique via branch and bound with bitsets
    best_clique = []
    best_size = 0
    
    # Order vertices by degree (descending) for better pruning
    # Use a simple ordering: sort by number of compatible neighbors
    degrees = [bin(compat[i]).count('1') for i in range(1, N + 1)]
    order = list(range(1, N + 1))
    order.sort(key=lambda x: -degrees[x - 1])
    
    # Map to 0-indexed positions for the clique algorithm
    # We'll work with a relabeled graph where vertices are in order[0], order[1], ...
    # P[i] in the new labeling corresponds to order[i]
    
    # Rebuild compat in the new ordering
    # new_compat[i] = bitset of new indices j where order[i] and order[j] are compatible
    new_compat = [0] * N
    for i in range(N):
        a = order[i]
        for j in range(N):
            if i != j:
                b = order[j]
                if (a * b) % (a + b) != 0:
                    new_compat[i] |= (1 << j)
    
    # Branch and bound for maximum clique using bitset representation
    # candidates is a bitset of available vertices
    def expand(clique_list, candidates):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        
        # Upper bound: len(clique_list) + popcount(candidates)
        bound = len(clique_list) + bin(candidates).count('1')
        if bound <= best_size:
            return
        
        if candidates == 0:
            if len(clique_list) > best_size:
                best_size = len(clique_list)
                best_clique = list(clique_list)
            return
        
        # Choose a vertex from candidates with max degree within candidates
        # For simplicity, pick the first one (or the one with highest original degree)
        # Try to pick vertex with max (degree in candidates)
        best_v = -1
        best_deg = -1
        c = candidates
        while c:
            v = c & (-c)
            v_idx = v.bit_length() - 1
            deg_in_cand = bin(new_compat[v_idx] & candidates).count('1')
            if deg_in_cand > best_deg:
                best_deg = deg_in_cand
                best_v = v_idx
            c &= c - 1
        
        if best_v == -1:
            return
        
        # Branch: include best_v
        new_clique = clique_list + [best_v]
        new_cand = candidates & new_compat[best_v]
        new_cand &= ~(1 << best_v)  # remove self
        expand(new_clique, new_cand)
        
        # Check time
        if time.time() > deadline:
            return
        
        # Branch: exclude best_v
        cand_without = candidates & ~(1 << best_v)
        expand(clique_list, cand_without)
    
    # Start with all vertices as candidates
    all_cand = (1 << N) - 1
    expand([], all_cand)
    
    # Convert back to original values
    result = [order[v] for v in best_clique]
    if not result:
        result = [1]
    
    return sorted(result)
