        def _check_girth_fast(n, edges, min_girth):
            adj = [[] for _ in range(n)]
            for i, j in edges:
                adj[i].append(j)
                adj[j].append(i)
            
            # Map neighbor to index in adj list for state encoding
            # State: (v, u) encoded as v * n + u? No, u is a neighbor.
            # Let's stick to (v, u) but optimize the loop.
            
            # We only need to check cycles of length < min_girth.
            # A cycle of length L corresponds to a non-backtracking walk of length L from (v, u) back to (u, v)?
            # No, from (v, u) to (v, w) where w is a neighbor of v, w != u?
            # If we start at (v, u), and after L steps we are at (v, w), and w != u, it's a cycle of length L+1?
            # Actually, standard definition:
            # Walk $v_0, v_1, ..., v_L$.
            # Non-backtracking: $v_{i-1} \neq v_{i+1}$.
            # Cycle: $v_0 = v_L$ and $v_{L-1} \neq v_1$ (to avoid backtracking at the end).
            
            # Let's iterate over all directed edges (u, v).
            # Start state: (v, u).
            # If we reach state (u, w) where w is a neighbor of u, and w != v?
            # No, if we reach (u, v) again, that's a cycle of length L+1?
            
            # Let's use the property:
            # Girth < g iff there exists a non-backtracking walk of length L < g that starts at (u,v) and ends at (v,u) ?
            # No.
            
            # Let's just optimize the existing BFS.
            # The existing BFS starts from (start_nb, start_v).
            # It checks if cur == start_v and prev != start_nb.
            # This correctly identifies cycles.
            
            # Optimization:
            # 1. Pre-compute adjacency as list of lists.
            # 2. Use a 2D array `dist` or `visited`?
            #    Size n x n is too big (6250^2 = 39M).
            #    But we only visit states reachable in < g steps.
            #    The number of visited states per start is small.
            #    Using a set is okay if the size is small.
            
            # The main cost is the number of BFS calls: n * k = 31250 calls.
            # Each BFS explores ~ 4^4 = 256 nodes on average before hitting depth limit or cycle?
            # Actually, depth is up to 8.
            # Total nodes per BFS ~ 1 + 4 + 16 + ... + 4^7 ~ 21845.
            # 31250 * 21845 ~ 6.8 * 10^8.
            # This is too slow in Python.
            
            # We need a smarter check.
            # Check girth >= g is equivalent to checking that the graph contains no cycles of length < g.
            # For small g (9), we can check for triangles, 4-cycles, etc.
            # But general check is hard.
            
            # Alternative:
            # Since LUW graphs are known to have girth >= 2d+1, we can verify this property by checking a smaller subset?
            # No, "certified exactly".
            
            # What if we use `numpy`?
            # Construct the non-backtracking matrix B (size nk x nk).
            # B is sparse.
            # We want to know if any cycle of length < g exists.
            # This is equivalent to checking if trace(B^L) > 0 for any L < g?
            # No, trace counts closed walks.
            # Closed non-backtracking walks of length L correspond to cycles of length L.
            # So if trace(B^L) > 0 for any L in [3, g-1], girth < g.
            
            # B is size 31250 x 31250.
            # Sparse matrix multiplication is fast.
            # We need B^L for L up to 8.
            # 8 multiplications of sparse matrices of size 31k x 31k.
            # This should be very fast with scipy.sparse.
            
            import scipy.sparse as sp
            import numpy as np
            
            n_states = n * k
            # Map state (v, u) to index
            # Index = v * k + index_of_u_in_adj[v]
            # We need to build B.
            # B[i, j] = 1 if state j can go to state i?
            # State (v, u) goes to (w, v) if w is neighbor of v and w != u.
            
            rows = []
            cols = []
            for v in range(n):
                for i, u in enumerate(adj[v]):
                    state_from = v * k + i
                    for j, w in enumerate(adj[v]):
                        if w == u:
                            continue
                        state_to = w * k + j
                        rows.append(state_to)
                        cols.append(state_from)
            
            B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_states, n_states))
            
            # Check powers
            B_pow = B.copy()
            for L in range(3, min_girth):
                B_pow = B_pow @ B
                # Check diagonal
                # trace is sum of diagonal
                if B_pow.diagonal().sum() > 0:
                    return False
            
            return True
        