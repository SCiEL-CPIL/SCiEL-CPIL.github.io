import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Part 1: A[0:2,0:2] * B[0:2,0:9] using Strassen for 2-column blocks
    for block_start in range(0, 8, 2):
        if time.time() > deadline:
            break
        k0 = block_start
        k1 = block_start + 1
        
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
    
    # Single column k=8: schoolbook 2x2 x 2x1
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    # Part 2: A[0:2,2] * B[2,0:9]
    for i in range(n):
        if time.time() > deadline:
            break
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
    
    rank = len(U_cols)
    
    # Try to reduce rank by removing redundant terms
    if time.time() < deadline - 5 and rank > 27:
        U_mat = np.array(U_cols).T  # shape (n*m, rank)
        V_mat = np.array(V_cols).T  # shape (m*p, rank)
        W_mat = np.array(W_cols).T  # shape (p*n, rank)
        
        # The tensor constraint: for each (i,k,j), sum_r U_r[i,k] * V_r[k,j] * W_r[i,j] = 1
        # Non-zero entries: (i,k,j) for i in range(n), k in range(m), j in range(p)
        # We need: sum_r U_r[i*m+k] * V_r[k*p+j] * W_r[j*n+i] = 1
        
        # Build the constraint matrix
        # For each non-zero tensor entry (i,k,j), we have:
        # sum_r U_r[i*m+k] * V_r[k*p+j] * W_r[j*n+i] = 1
        # This is nonlinear in the decomposition, but if we fix all terms except one,
        # it becomes linear.
        
        # Try removing each term
        for r_remove in range(rank):
            if time.time() > deadline - 5:
                break
            
            # Build the system: find d_s (s != r_remove) such that
            # U_r ⊗ V_r ⊗ W_r = sum_{s != r_remove} d_s * (U_s ⊗ V_s ⊗ W_s)
            
            # This is equivalent to: for each (a,b,c) where the tensor is non-zero,
            # U_r[a] * V_r[b] * W_r[c] = sum_{s != r_remove} d_s * U_s[a] * V_s[b] * W_s[c]
            
            # The non-zero entries are: a = i*m+k, b = k*p+j, c = j*n+i
            # for i in range(n), k in range(m), j in range(p)
            
            equations = []
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        lhs = U_mat[a, r_remove] * V_mat[b, r_remove] * W_mat[c, r_remove]
                        if lhs != 0:
                            row = np.zeros(rank - 1)
                            for s in range(rank):
                                if s == r_remove:
                                    continue
                                val = U_mat[a, s] * V_mat[b, s] * W_mat[c, s]
                                idx = s if s < r_remove else s - 1
                                row[idx] = val
                            equations.append((row, lhs))
            
            if not equations:
                continue
            
            A_sys = np.array([eq[0] for eq in equations])
            b_sys = np.array([eq[1] for eq in equations])
            
            # Solve the linear system
            try:
                d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                # Check if the solution is valid
                if np.allclose(A_sys @ d, b_sys, atol=1e-10):
                    # We found a valid removal!
                    # Remove term r_remove
                    new_U_cols = [U_cols[s] for s in range(rank) if s != r_remove]
                    new_V_cols = [V_cols[s] for s in range(rank) if s != r_remove]
                    new_W_cols = [W_cols[s] for s in range(rank) if s != r_remove]
                    
                    # Verify the decomposition
                    U_new = np.array(new_U_cols).T
                    V_new = np.array(new_V_cols).T
                    W_new = np.array(new_W_cols).T
                    
                    valid = True
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                a = i * m + k
                                b = k * p + j
                                c = j * n + i
                                val = sum(U_new[a, s] * V_new[b, s] * W_new[c, s] for s in range(len(new_U_cols)))
                                if abs(val - 1) > 1e-10:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        rank = len(new_U_cols)
                        U_cols = new_U_cols
                        V_cols = new_V_cols
                        W_cols = new_W_cols
                        break  # Restart the removal process with the new rank
            except:
                pass
    
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
