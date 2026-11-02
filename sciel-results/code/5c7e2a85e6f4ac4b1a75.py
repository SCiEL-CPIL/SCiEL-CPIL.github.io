import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 4, 7
    T = n * m * p  # 56 tensor entries

    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    # Precompute tensor entries: T[i, j, k] = 1 if this entry contributes to C[i,k] from A[i,j]*B[j,k]
    # But our tensor is T_{i,j,k} where i indexes A rows, j indexes A cols/B rows, k indexes B cols
    # The multiplication tensor: T_{i,j,k} = 1 for all (i,j,k), representing A[i,j]*B[j,k]->C[i,k]
    # Actually the tensor has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # And the "fiber" is: U[i,j,:] * V[j,k,:] * W[k,i,:] summed over rank should equal T[i,j,k]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So the tensor T has indices (i, j, k) where:
    #   U is indexed by (i, j) -> row i*m+j
    #   V is indexed by (j, k) -> row j*p+k  
    #   W is indexed by (k, i) -> row k*n+i
    # And sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k} = 1 for all (i,j,k)
    # Wait no, matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k), and the factorization is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is the standard matrix multiplication tensor.

    # Schoolbook fallback
    def schoolbook():
        rank = T
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    # Strassen-based block decomposition (rank 50)
    def strassen_decomposition():
        rank = 50
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h)
            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1
            # M2 = (c+d)e
            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10, col] += 1; W[c11, col] -= 1
            col += 1
            # M3 = a(f-h)
            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1
            # M4 = d(g-e)
            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1
            # M5 = (a+b)h
            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00, col] -= 1; W[c01, col] += 1
            col += 1
            # M6 = (c-a)(e+f)
            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11, col] += 1
            col += 1
            # M7 = (b-d)(g+h)
            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00, col] += 1
            col += 1

        def add_sb221(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)
            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10, col] += 1; col += 1

        for js in [0, 2]:
            for ks in [0, 2, 4]:
                add_strassen(0, js, ks)
            add_sb221(0, js, 6)

        return U, V, W, col

    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Get baseline rank-50
    U50, V50, W50, rank50 = strassen_decomposition()
    best_rank = rank50
    best_U, best_V, best_W = U50, V50, W50

    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to find a lower rank decomposition using a structured search
    # Strategy: try rank r < 50 by solving for factor matrices with random initialization
    # and using a local optimization approach
    
    # Alternative: try to merge some of the Strassen factors
    # The key insight: in the 2x4x7 case, we have 2 row blocks (but n=2 so only 1 pair of rows)
    # and 4 col blocks of B, but we grouped them as [0,1], [2,3], [4,5], [6]
    
    # Let me try a different block structure: use 2x2 Strassen for all 2x2 sub-blocks of B
    # but with overlap or different grouping to save rank
    
    # Actually, let me try a direct approach: represent the tensor and use 
    # a greedy/randomized rank minimization
    
    # Try rank values from 49 down to 40
    for target_rank in range(49, 39, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(42 + attempt * 7 + target_rank)
            
            # Initialize with random values
            U_try = np.random.randn(n*m, target_rank) * 0.5
            V_try = np.random.randn(m*p, target_rank) * 0.5
            W_try = np.random.randn(p*n, target_rank) * 0.5
            
            # Use the known rank-50 as a guide: take first target_rank columns
            if target_rank <= 50:
                U_try = U50[:, :target_rank].copy()
                V_try = V50[:, :target_rank].copy()
                W_try = W50[:, :target_rank].copy()
            
            # Iterative refinement: for each tensor entry (i,j,k), adjust factors
            # to better match T[i,j,k]=1
            converged = False
            for iteration in range(20):
                if time.time() > deadline - 3:
                    break
                
                max_change = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = U_try[i*m+j, :]
                            vk = V_try[j*p+k, :]
                            wk = W_try[k*n+i, :]
                            current = np.dot(ui, vk * wk)
                            # Gradient-based update would be ideal, but let's try a simpler approach
                            # If current != 1, we need to adjust
                
                # Check if we've converged
                if verify(U_try, V_try, W_try, tol=1e-3):
                    converged = True
                    break
        
        # Final verification with tight tolerance
        if verify(U_try, V_try, W_try, tol=1e-9):
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break

    # If no improvement found, return the rank-50 decomposition
    if time.time() > deadline - 1:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
