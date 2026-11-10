"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 125  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as numpy matrix for fast vectorized operations
    A = np.zeros((n, n), dtype=np.uint8)
    
    def add_edge(u, v):
        A[u, v] = 1
        A[v, u] = 1
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(5 * k + a, 5 * k + b)
    
    # Track best state
    best_A = A.copy()
    best_count = int(np.sum(A)) // 2
    
    def has_c6(u, v):
        """Check if adding edge (u,v) creates a C6.
        Adding edge (u,v) creates a C6 iff there is a path of length 5
        from u to v in the current graph with all intermediate vertices distinct
        from u and v.
        Path: u -> a -> b -> c -> e -> v (5 edges, 4 intermediate vertices a,b,c,e)
        All of a,b,c,e must be distinct and not equal to u or v.
        """
        # For each a in N(u)\{v}, e in N(v)\{u}, a!=e:
        #   Need b in N(a) and c in N(b)∩N(e) with b,c distinct from {u,v,a,e} and b!=c
        # Vectorized: for each (a,e) pair, check if there exists b,c
        # s.t. A[a,b]=1, A[b,c]=1, A[c,e]=1, all distinct constraints met
        
        neighbors_u = np.where(A[u] == 1)[0]
        neighbors_u = neighbors_u[neighbors_u != v]
        neighbors_v = np.where(A[v] == 1)[0]
        neighbors_v = neighbors_v[neighbors_v != u]
        
        for a in neighbors_u:
            for e in neighbors_v:
                if e == a:
                    continue
                # Need path a -> b -> c -> e of length 3
                # b in N(a), c in N(b) ∩ N(e)
                # Constraints: b,c distinct from {u,v,a,e} and b != c
                b_candidates = np.where(A[a] == 1)[0]
                # Filter b: not u, v, a, e
                mask_b = np.ones(n, dtype=bool)
                mask_b[u] = False
                mask_b[v] = False
                mask_b[a] = False
                mask_b[e] = False
                b_candidates = b_candidates[mask_b[b_candidates]]
                
                if len(b_candidates) == 0:
                    continue
                
                # For each b, check if there's c in N(b) ∩ N(e) with c distinct from {u,v,a,b,e}
                # c must be in N(b) and N(e), and c != u,v,a,b,e
                for b in b_candidates:
                    # c in N(b) ∩ N(e), c not in {u,v,a,b,e}
                    common = np.where((A[b] == 1) & (A[e] == 1))[0]
                    # Filter c: not u, v, a, b, e
                    mask_c = np.ones(n, dtype=bool)
                    mask_c[u] = False
                    mask_c[v] = False
                    mask_c[a] = False
                    mask_c[b] = False
                    mask_c[e] = False
                    common = common[mask_c[common]]
                    if len(common) > 0:
                        return True
        return False
    
    # Greedy edge addition with multiple passes
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        random.shuffle(all_pairs)
        added_any = False
        
        for (u, v) in all_pairs:
            if time.time() > deadline - 3:
                break
            if A[u, v] == 1:
                continue
            if not has_c6(u, v):
                add_edge(u, v)
                added_any = True
        
        current_count = int(np.sum(A)) // 2
        if current_count > best_count:
            best_count = current_count
            best_A = A.copy()
        
        if not added_any:
            # Try perturbation: remove a random edge and try to re-add different ones
            if time.time() > deadline - 3:
                break
            # Pick a random edge to remove
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
            if edge_list:
                ru, rv = random.choice(edge_list)
                A[ru, rv] = 0
                A[rv, ru] = 0
                # Now try to add edges again
                random.shuffle(all_pairs)
                for (u, v) in all_pairs:
                    if time.time() > deadline - 3:
                        break
                    if A[u, v] == 1:
                        continue
                    if not has_c6(u, v):
                        add_edge(u, v)
                current_count = int(np.sum(A)) // 2
                if current_count > best_count:
                    best_count = current_count
                    best_A = A.copy()
            else:
                break
    
    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
