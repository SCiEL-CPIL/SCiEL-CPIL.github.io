"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 256
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    
    # Map valid numbers to indices 0..M-1
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency matrix as bitmasks for fast operations
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self, self-loops handled by forbidden set
                    adj[i] |= (1 << j)
    
    # Tomita's Algorithm for Maximum Independent Set (MIS)
    # We want to find the largest subset of vertices such that no two are adjacent.
    
    # Pre-compute degrees for initial ordering
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    
    # Sort vertices by degree (ascending) to help the greedy part of the algorithm
    # Actually, Tomita's usually sorts by degree descending for the "candidate set" reduction
    # but let's stick to the standard implementation which often uses a specific ordering.
    
    # The algorithm maintains:
    # - C: candidate set (bitmask)
    # - S: current independent set (list of indices)
    # - P: pool of vertices not yet processed (bitmask)
    
    # To make it efficient in Python, we use bitmasks for C and P.
    # We need a way to iterate over set bits.
    
    def bit_count(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    # Tomita's algorithm (modified for Python)
    # Reference: Tomita, K., & Morihara, E. (2003). "An exact algorithm for finding maximum independent sets in arbitrary graphs."
    
    best_size = 0
    best_set = []
    
    # Initial candidate set is all vertices
    all_vertices = (1 << M) - 1
    
    # We use a recursive function. To avoid recursion depth issues (M ~ 250), 
    # we might need to increase recursion limit or use an iterative approach.
    # However, the depth of the search tree is bounded by the size of the MIS, 
    # which is likely much smaller than M.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def tomita(C, S, P):
        nonlocal best_size, best_set, deadline, start_time
        
        if time.time() > deadline:
            return
            
        if C == 0:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        # Greedy upper bound
        # We can use a simple greedy coloring or just the size of C
        # A better bound: greedy coloring of the subgraph induced by C
        # For simplicity and speed, let's use a basic bound first.
        # If |S| + |C| <= best_size, prune.
        if len(S) + bit_count(C) <= best_size:
            return
        
        # Select a vertex u from C with maximum degree in the subgraph induced by C
        # To find max degree efficiently:
        # deg_c[u] = bit_count(adj[u] & C)
        
        # We can iterate through bits of C
        # Optimization: If C is small, iterate directly.
        # Let's pick the vertex with the highest degree in C.
        
        max_deg = -1
        u = -1
        c_temp = C
        while c_temp:
            # Get lowest set bit
            lsb = c_temp & -c_temp
            idx = lsb.bit_length() - 1
            # Remove from c_temp
            c_temp ^= lsb
            
            deg = bit_count(adj[idx] & C)
            if deg > max_deg:
                max_deg = deg
                u = idx
        
        if u == -1:
            return
            
        # Branch 1: Include u
        # New C = C \ (N(u) ∪ {u})
        # New P = P ∪ (N(u) ∩ C)  -- Wait, standard Tomita keeps P for vertices that are "dominated" or not?
        # Let's use the standard formulation:
        # C is the candidate set.
        # Pick u in C.
        # Branch 1: u in IS.
        #   New C = C \ ({u} ∪ N(u))
        #   New P = P ∪ N(u)  (vertices in N(u) that were in C are now excluded from C, but might be needed later? No, they are adjacent to u, so they can't be in IS with u. But they might be in IS with other choices. In Tomita's, P accumulates vertices that are "processed" but not in S? 
        #   Actually, the standard Tomita algorithm maintains C and P such that C ∪ P = Original Set.
        #   When we pick u:
        #   1. Add u to S.
        #   2. C1 = C \ ({u} ∪ N(u))
        #   3. P1 = P ∪ ({u} ∪ N(u) ∩ C)
        #   4. Recurse(C1, S + [u], P1)
        
        #   5. C2 = C \ {u}
        #   6. P2 = P ∪ {u}
        #   7. Recurse(C2, S, P2)
        
        # Let's implement this.
        
        # Branch 1: Include u
        u_mask = 1 << u
        neighbors_in_c = adj[u] & C
        remove_set = u_mask | neighbors_in_c
        C1 = C & ~remove_set
        P1 = P | remove_set
        
        S.append(u)
        tomita(C1, S, P1)
        S.pop()
        
        # Check deadline
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude u
        C2 = C & ~u_mask
        P2 = P | u_mask
        tomita(C2, S, P2)
        
    # Initial call
    tomita(all_vertices, [], 0)
    
    # If Tomita didn't finish or found nothing (unlikely), fallback to greedy
    if best_size == 0:
        # Simple greedy fallback
        selected = 0
        # Order by degree
        order = sorted(range(M), key=lambda i: degrees[i])
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        best_set = [i for i in range(M) if (selected >> i) & 1]
        best_size = len(best_set)
        
    # Convert back to actual numbers
    result = [idx_to_num[i] for i in best_set]
    
    if not result:
        # Absolute fallback: pick 1 if valid
        if 1 not in forbidden:
            result = [1]
        else:
            result = [x for x in range(1, N+1) if x not in forbidden][:1]
            
    return sorted(result)
# EVOLVE-BLOCK-END
