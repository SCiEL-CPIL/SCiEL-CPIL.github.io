import numpy as np
import random
import time

def construct():
    target_n = 17
    random.seed(42)
    start_time = time.time()
    
    # Initialize random coloring
    n_edges = target_n * (target_n - 1) // 2
    colors = np.random.randint(0, 4, n_edges)
    
    def get_adj_matrices(colors, n):
        adj = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = 1
                adj[c][j, i] = 1
                idx += 1
        return adj
    
    def penalty(adj_matrices):
        total = 0
        for c in range(4):
            A = adj_matrices[c]
            # Calculate Tr(A^6)
            # A^2 = A @ A
            A2 = A @ A
            A4 = A2 @ A2
            A6 = A4 @ A2
            total += np.trace(A6)
        return total
    
    # Simulated Annealing
    T = 10.0
    cool = 0.999
    best_adj = None
    best_pen = penalty(get_adj_matrices(colors, target_n))
    curr_pen = best_pen
    curr_adj = get_adj_matrices(colors, target_n)
    
    for i in range(20000):
        if time.time() - start_time > 100:
            break
        
        # Pick random edge
        e_idx = random.randint(0, n_edges - 1)
        # Determine u, v from e_idx
        # e_idx = i*n - (i+1)*i//2 + (j-i-1) is complex.
        # Use precomputed list of edges
        # Let's precompute edge list
        if not hasattr(construct, 'edge_list'):
            construct.edge_list = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
        
        u, v = construct.edge_list[e_idx]
        old_c = colors[e_idx]
        new_c = random.randint(0, 3)
        while new_c == old_c:
            new_c = random.randint(0, 3)
        
        # Update adjacency matrices
        curr_adj[old_c][u, v] = 0
        curr_adj[old_c][v, u] = 0
        curr_adj[new_c][u, v] = 1
        curr_adj[new_c][v, u] = 1
        
        new_pen = penalty(curr_adj)
        delta = new_pen - curr_pen
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            curr_pen = new_pen
            colors[e_idx] = new_c
            if curr_pen < best_pen:
                best_pen = curr_pen
                best_adj = [a.copy() for a in curr_adj]
        else:
            # Revert
            curr_adj[old_c][u, v] = 1
            curr_adj[old_c][v, u] = 1
            curr_adj[new_c][u, v] = 0
            curr_adj[new_c][v, u] = 0
        
        T *= cool
        if T < 0.01:
            T = 0.01

    # Final check for C6s
    # Use best_adj if available, else curr_adj
    final_adj = best_adj if best_adj is not None else curr_adj
    
    # Strict C6 check
    def has_c6(adj, n):
        for c in range(4):
            A = adj[c]
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        continue
                    # Count simple paths of length 4 from u to v in A without (u,v)
                    # Path u-a-b-c-d-v
                    # a in N(u), d in N(v)
                    # b in N(a), c in N(d)
                    # Distinctness: u,a,b,c,d,v distinct
                    count = 0
                    for a in range(n):
                        if a == u or a == v or A[u, a] == 0:
                            continue
                        for d in range(n):
                            if d == u or d == v or d == a or A[v, d] == 0:
                                continue
                            for b in range(n):
                                if b in (u, v, a, d) or A[a, b] == 0:
                                    continue
                                for c in range(n):
                                    if c in (u, v, a, d, b) or A[d, c] == 0:
                                        continue
                                    if A[b, c] == 1:
                                        count += 1
                    if count > 0:
                        return True
        return False

    if not has_c6(final_adj, target_n):
        # Reconstruct colors list from final_adj
        final_colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                for c in range(4):
                    if final_adj[c][i, j] == 1:
                        final_colors.append(c)
                        break
        return {'n': target_n, 'colors': final_colors}
    else:
        # Fallback to n=9
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
