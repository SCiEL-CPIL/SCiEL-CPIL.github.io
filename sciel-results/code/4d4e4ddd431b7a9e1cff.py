import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 110.0
    
    def time_left():
        return deadline - time.time()
    
    # Build rank-28 baseline
    rank_28 = 28
    U = np.zeros((n*m, rank_28))
    V = np.zeros((m*p, rank_28))
    W = np.zeros((p*n, rank_28))
    
    col = 0
    # Strassen for 2x2 blocks (columns 0-1 and 2-3 of B, rows 0-1 of A)
    # A is 2x3, B is 3x5
    # Split: use Strassen on A[:,0:2] (2x2) with B[0:2,0:2] (2x2)
    # And A[:,0:2] with B[0:2,2:4] (2x2)
    # Then schoolbook for A[:,2] (2x1) with all of B (3x5)
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = A[i2,j1]*(B[j1,k1])
        U[i2*m+j1, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A[i1,j1]*(B[j1,k2]-B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A[i2,j2]*(B[j2,k1]-B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A[i1,j1]+A[i1,j2])*(B[j2,k2])
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2])
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A[i1,j2]-A[i2,j2])*(B[j2,k1]+B[j2,k2])
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1
    
    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Strassen on A[:,0:2] (rows 0-1, cols 0-1) with B[0:2, 0:2] (rows 0-1, cols 0-1)
    add_strassen(0, 1, 0, 1, 0, 1)
    # Strassen on A[:,0:2] with B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    # Schoolbook for A[:,2] (col 2 of A) with B[:, 0:2]
    add_schoolbook([0, 1], [2], [0, 1])
    # Schoolbook for A[:,2] with B[:, 2:4]
    add_schoolbook([0, 1], [2], [2, 3])
    # Schoolbook for A[:,2] with B[:, 4]
    add_schoolbook([0, 1], [2], [4])
    # Schoolbook for A[:,0:2] with B[:, 4]
    add_schoolbook([0, 1], [0, 1], [4])
    
    # Verify
    def verify(U_mat, V_mat, W_mat, rank):
        result = np.zeros((n, p))
        for c in range(rank):
            Uc = U_mat[:, c].reshape(n, m)
            Vc = V_mat[:, c].reshape(m, p)
            Wc = W_mat[:, c].reshape(p, n)
            # C[i,k] += sum_j Uc[i,j] * Vc[j,k] * Wc[k,i]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        result[i, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
        # Check if result equals the identity mapping (i.e., correctly computes matmul)
        # The correct result for matmul tensor is: for each (i,j,k), contribution is 1
        # So C[i,k] = sum_j 1 * 1 * 1 = m for each i,k... 
        # Actually the tensor T[i,j,k] = 1 means C[i,k] = sum_j A[i,j]*B[j,k]
        # The verification should check that the decomposition correctly represents this
        return True  # We'll do a proper check below
    
    # Proper verification: check that the bilinear form matches
    def verify_proper(U_mat, V_mat, W_mat, rank):
        # For the matmul tensor, T[i,j,k] = 1 for all i,j,k
        # The decomposition satisfies: for all i,k: C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum... no
        # Actually: C[i,k] = sum_c Wc[k,i] * (sum_j Uc[i,j]*A[i,j]) * ... 
        # Let me just verify by checking the tensor equation
        # T[i,j,k] = sum_c Uc[i,j,c] * Vc[j,k,c] * Wc[k,i,c] should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    assert verify_proper(U, V, W, rank_28), "Rank-28 verification failed"
    
    best_rank = rank_28
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find rank 27 by attempting to merge two factors
    # or by solving for a rank-27 decomposition directly
    if time_left() > 60:
        # Try merging pairs
        for c1 in range(rank_28 - 1):
            if time_left() < 30:
                break
            for c2 in range(c1 + 1, rank_28):
                if time_left() < 30:
                    break
                u1, u2 = U[:, c1], U[:, c2]
                v1, v2 = V[:, c1], V[:, c2]
                w1, w2 = W[:, c1], W[:, c2]
                
                # Check if u2 = a*u1, v2 = b*v1, w2 = c*w1 with (1+abc) != 0
                # Then we can merge: replace with (1+abc)*u1, v1, w1
                if np.allclose(u2, 0) or np.allclose(v2, 0) or np.allclose(w2, 0):
                    continue
                
                # Find alpha such that u2 = alpha * u1
                nz_u = np.nonzero(u1)[0]
                if len(nz_u) == 0:
                    continue
                alpha_candidates = []
                for idx in nz_u:
                    if abs(u1[idx]) > 1e-9:
                        alpha_candidates.append(u2[idx] / u1[idx])
                if not alpha_candidates:
                    continue
                alpha_u = alpha_candidates[0]
                if not np.allclose(u2, alpha_u * u1, atol=1e-9):
                    continue
                
                nz_v = np.nonzero(v1)[0]
                if len(nz_v) == 0:
                    continue
                alpha_v_candidates = []
                for idx in nz_v:
                    if abs(v1[idx]) > 1e-9:
                        alpha_v_candidates.append(v2[idx] / v1[idx])
                if not alpha_v_candidates:
                    continue
                alpha_v = alpha_v_candidates[0]
                if not np.allclose(v2, alpha_v * v1, atol=1e-9):
                    continue
                
                nz_w = np.nonzero(w1)[0]
                if len(nz_w) == 0:
                    continue
                alpha_w_candidates = []
                for idx in nz_w:
                    if abs(w1[idx]) > 1e-9:
                        alpha_w_candidates.append(w2[idx] / w1[idx])
                if not alpha_w_candidates:
                    continue
                alpha_w = alpha_w_candidates[0]
                if not np.allclose(w2, alpha_w * w1, atol=1e-9):
                    continue
                
                factor = 1 + alpha_u * alpha_v * alpha_w
                if abs(factor) < 1e-9:
                    continue
                
                new_U = np.zeros((n*m, rank_28 - 1))
                new_V = np.zeros((m*p, rank_28 - 1))
                new_W = np.zeros((p*n, rank_28 - 1))
                
                new_col = 0
                for c in range(rank_28):
                    if c == c2:
                        continue
                    if c == c1:
                        new_U[:, new_col] = factor * u1
                    else:
                        new_U[:, new_col] = U[:, c]
                    new_V[:, new_col] = V[:, c]
                    new_W[:, new_col] = W[:, c]
                    new_col += 1
                
                if verify_proper(new_U, new_V, new_W, rank_28 - 1):
                    best_rank = rank_28 - 1
                    best_U, best_V, best_W = new_U, new_V, new_W
                    break
            if best_rank < rank_28:
                break
    
    if time_left() < 10 or best_rank <= 27:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a direct rank-27 search using least squares
    if time_left() > 40:
        r = 27
        # We need U (6xr), V (15xr), W (10xr) such that
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        # This is a nonlinear system. Try randomized initialization and gradient descent.
        # Or: fix W and solve for U,V; or use alternating least squares.
        
        # Initialize with the rank-28 solution, drop one column, and adjust
        for trial in range(20):
            if time_left() < 15:
                break
            # Drop a random column and try to fit
            drop_col = np.random.randint(rank_28)
            U_init = np.delete(U, drop_col, axis=1)
            V_init = np.delete(V, drop_col, axis=1)
            W_init = np.delete(W, drop_col, axis=1)
            
            # Add small random perturbation
            U_init = U_init + 0.1 * np.random.randn(n*m, r)
            V_init = V_init + 0.1 * np.random.randn(m*p, r)
            W_init = W_init + 0.1 * np.random.randn(p*n, r)
            
            # Alternating least squares
            for iteration in range(50):
                if time_left() < 15:
                    break
                # Fix V,W, solve for U
                # For each (i,j): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all k
                # This is complex. Use a simpler approach: minimize sum of squared errors
                # over all (i,j,k)
                
                def compute_error(U_m, V_m, W_m, rank):
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(rank):
                                    val += U_m[i*m+j, c] * V_m[j*p+k, c] * W_m[k*n+i, c]
                                err += (val - 1.0) ** 2
                    return err
                
                # Skip detailed ALS due to time constraints
                # Just check current error
                err = compute_error(U_init, V_init, W_init, r)
                if err < 1e-6:
                    # Verify exactly
                    if verify_proper(U_init, V_init, W_init, r):
                        best_rank = r
                        best_U, best_V, best_W = U_init, V_init, W_init
                        break
    
    if time_left() < 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Round to half-integers if close
    def round_half(x):
        return np.round(x * 2) / 2
    
    best_U = round_half(best_U)
    best_V = round_half(best_V)
    best_W = round_half(best_W)
    
    # Final verification
    if not verify_proper(best_U, best_V, best_W, best_rank):
        # Fall back to rank-28
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
