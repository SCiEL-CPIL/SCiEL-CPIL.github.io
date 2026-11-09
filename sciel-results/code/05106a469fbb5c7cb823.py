import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition sizes for 5x5x5
    sizes = [2, 2, 1]
    starts = [0, 2, 4]
    
    # Initialize U, V, W with zeros. We don't know the rank ahead of time exactly 
    # if we want to be safe, but we calculated 117. 
    # Let's allocate a bit more and trim, or just append.
    # Appending is safer.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: list of (global_index, coeff)
        v_coeffs: list of (global_index, coeff)
        w_coeffs: list of (global_index, coeff)
        """
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for idx, coeff in u_coeffs:
            U_col[idx] = coeff
        for idx, coeff in v_coeffs:
            V_col[idx] = coeff
        for idx, coeff in w_coeffs:
            W_col[idx] = coeff
            
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def map_u(i, k, p_local, q_local):
        # A block (i, k)
        # Global row: starts[i] + p_local
        # Global col: starts[k] + q_local
        return (starts[i] + p_local) * m + (starts[k] + q_local)

    def map_v(k, j, q_local, t_local):
        # B block (k, j)
        # Global row: starts[k] + q_local
        # Global col: starts[j] + t_local
        return (starts[k] + q_local) * p + (starts[j] + t_local)

    def map_w(i, j, p_local, t_local):
        # C block (i, j)
        # Global row: starts[i] + p_local
        # Global col: starts[j] + t_local
        # W index: col * n + row
        return (starts[j] + t_local) * n + (starts[i] + p_local)

    # Iterate over output blocks i, j
    for i in range(3):
        for j in range(3):
            # Iterate over intermediate block k
            for k in range(3):
                # Determine sizes
                r = sizes[i] # rows of A_ik
                s = sizes[k] # cols of A_ik, rows of B_kj
                c = sizes[j] # cols of B_kj
                
                # Determine global indices for local (p, q, t)
                # We will generate terms for the sub-problem A_ik * B_kj -> C_ij part
                
                if r == 2 and s == 2 and c == 2:
                    # Strassen
                    # Local indices p, q, t in 0..1
                    
                    # Term 1
                    add_term(
                        [(map_u(i, k, 0, 0), 1), (map_u(i, k, 1, 1), 1)],
                        [(map_v(k, j, 0, 0), 1), (map_v(k, j, 1, 1), 1)],
                        [(map_w(i, j, 0, 0), 1), (map_w(i, j, 1, 1), 1)]
                    )
                    # Term 2
                    add_term(
                        [(map_u(i, k, 1, 0), 1), (map_u(i, k, 1, 1), 1)],
                        [(map_v(k, j, 0, 0), 1)],
                        [(map_w(i, j, 1, 0), 1), (map_w(i, j, 1, 1), -1)]
                    )
                    # Term 3
                    add_term(
                        [(map_u(i, k, 0, 0), 1)],
                        [(map_v(k, j, 0, 1), 1), (map_v(k, j, 1, 1), -1)],
                        [(map_w(i, j, 0, 1), 1), (map_w(i, j, 1, 1), 1)]
                    )
                    # Term 4
                    add_term(
                        [(map_u(i, k, 1, 1), 1)],
                        [(map_v(k, j, 1, 0), 1), (map_v(k, j, 0, 0), -1)],
                        [(map_w(i, j, 0, 0), 1), (map_w(i, j, 1, 0), 1)]
                    )
                    # Term 5
                    add_term(
                        [(map_u(i, k, 0, 0), 1), (map_u(i, k, 0, 1), 1)],
                        [(map_v(k, j, 1, 1), 1)],
                        [(map_w(i, j, 0, 0), -1), (map_w(i, j, 0, 1), 1)]
                    )
                    # Term 6
                    add_term(
                        [(map_u(i, k, 1, 0), 1), (map_u(i, k, 0, 0), -1)],
                        [(map_v(k, j, 0, 0), 1), (map_v(k, j, 0, 1), 1)],
                        [(map_w(i, j, 1, 1), 1)]
                    )
                    # Term 7
                    add_term(
                        [(map_u(i, k, 0, 1), 1), (map_u(i, k, 1, 1), -1)],
                        [(map_v(k, j, 1, 0), 1), (map_v(k, j, 1, 1), 1)],
                        [(map_w(i, j, 0, 0), 1)]
                    )
                else:
                    # Schoolbook
                    for p_local in range(r):
                        for q_local in range(s):
                            for t_local in range(c):
                                add_term(
                                    [(map_u(i, k, p_local, q_local), 1)],
                                    [(map_v(k, j, q_local, t_local), 1)],
                                    [(map_w(i, j, p_local, t_local), 1)]
                                )
    
    rank = len(U_cols)
    
    # Convert lists of lists to list of lists (standard format)
    # U: (n*m, rank)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
