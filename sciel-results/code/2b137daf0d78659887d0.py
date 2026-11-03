    import time
    import sys

    DIMENSIONS = [3, 5, 6]

    def construct():
        n, m, p = DIMENSIONS
        # We will use Strassen's algorithm for 2x2x2 blocks where possible.
        # Indices: i in [0, n), j in [0, m), k in [0, p)
        
        # Initialize U, V, W as lists of lists (rows)
        # We will append columns dynamically
        U = [[] for _ in range(n * m)]
        V = [[] for _ in range(m * p)]
        W = [[] for _ in range(p * n)]
        
        # Strassen blocks
        i_blocks = [(0, 1)]
        j_blocks = [(0, 1), (2, 3)]
        k_blocks = [(0, 1), (2, 3), (4, 5)]
        
        for i0, i1 in i_blocks:
            for j0, j1 in j_blocks:
                for k0, k1 in k_blocks:
                    # 7 terms for this block
                    # ... (patterns same as before) ...
                    
                    U_pat = [
                        [[1, 0], [0, 1]],
                        [[0, 0], [1, 1]],
                        [[1, 0], [0, 0]],
                        [[0, 0], [0, 1]],
                        [[1, 1], [0, 0]],
                        [[-1, 0], [1, 0]],
                        [[0, 1], [0, -1]]
                    ]
                    V_pat = [
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 0]],
                        [[0, 1], [0, -1]],
                        [[0, -1], [1, 0]],
                        [[0, 0], [0, 1]],
                        [[1, 1], [0, 0]],
                        [[0, 1], [1, 1]]
                    ]
                    W_pat = [
                        [[1, 0], [0, 1]],
                        [[0, 0], [1, 0]],
                        [[0, 1], [0, 0]],
                        [[0, 0], [0, 1]],
                        [[-1, 0], [0, 1]],
                        [[0, 0], [1, 0]],
                        [[1, 0], [0, 1]]
                    ]
                    
                    for t in range(7):
                        # Append to U
                        for ip in range(2):
                            for jp in range(2):
                                val = U_pat[t][ip][jp]
                                U[(i0 + ip) * m + (j0 + jp)].append(val)
                        
                        # Append to V
                        for jp in range(2):
                            for kp in range(2):
                                val = V_pat[t][jp][kp]
                                V[(j0 + jp) * p + (k0 + kp)].append(val)
                        
                        # Append to W
                        for ip in range(2):
                            for kp in range(2):
                                val = W_pat[t][ip][kp]
                                W[(k0 + kp) * n + (i0 + ip)].append(val)
        
        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if this term was covered by Strassen
                    covered = False
                    if i in [0, 1]:
                        if j in [0, 1, 2, 3]:
                            if k in [0, 1, 2, 3, 4, 5]:
                                covered = True
                    if not covered:
                        U[i * m + j].append(1)
                        V[j * p + k].append(1)
                        W[k * n + i].append(1)
        
        return {"U": U, "V": V, "W": W}
    