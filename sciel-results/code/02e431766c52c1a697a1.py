"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def check(U, V, W, tol=1e-9):
        """Verify exact decomposition via random sampling."""
        if time.time() > deadline:
            return False
        rng = np.random.default_rng(42)
        for _ in range(200):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            # U is (nm, rank), V is (mp, rank), W is (pn, rank)
            # C = W^T @ (U^T @ V^T)^T ... let's use the tensor form
            # C_flat[k*n+i] = sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c]
            # Vectorized: C_flat = sum over c of W[:,c] * (outer product of U[:,c] and V[:,c] reshaped)
            # U_c = U[:, c] -> shape (n*m,)
            # V_c = V[:, c] -> shape (m*p,)
            # term_c = np.einsum('i,j->ij', U_c.reshape(n,m), V_c.reshape(m,p)) -> shape (n, p)
            # C_c = term_c.flatten() -> shape (n*p,) = shape (p*n,)
            # C_flat = sum_c W[:,c] * C_c
            # Let's do this efficiently.
            # C_flat = np.einsum('ic,ic->c', U, V) ... no, U and V have different shapes.
            
            # Let's use the property: C = sum_c (W[:,c] * (U_c @ V_c))
            # where U_c is (n, m), V_c is (m, p), W_c is (p, n)
            # This is getting complex. Let's just do the full tensor check.
            # T = sum_c U_c V_c^T W_c^T ... no.
            
            # The bilinear form is:
            # C[k*n + i] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Let's just verify a few specific entries to be fast.
            # Actually, let's just verify the full tensor equality using np.einsum.
            # C[i, k] = sum_j A[i, j] B[j, k]
            # Our formula: C[i, k] = sum_c U[i, c] (where U is reshaped?) ...
            
            # Let's stick to the index notation for verification, it's slow but reliable.
            # For speed, let's just check 10 random entries.
            for _ in range(10):
                i = rng.integers(0, n)
                j = rng.integers(0, m)
                k = rng.integers(0, p)
                idx_c = k * n + i
                idx_u = i * m + j
                idx_v = j * p + k
                
                expected = A[i, j] * B[j, k]
                # This is just one term, not the sum.
                # We need to sum over c.
                # C[i, k] = sum_c W[idx_c, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
                # Wait, the decomposition is:
                # C[i, k] = sum_c (sum_j U[i*m+j, c] * A[i, j]) * (sum_j V[j*p+k, c] * B[j, k]) ... no, that's not right.
                
                # The correct bilinear form for matrix multiplication is:
                # C[i, k] = sum_c U_c(i, k) * A * V_c(i, k) * B
                # Where U_c, V_c are fixed linear maps.
                
                # Let's just compute the full C matrix and compare.
                C_calc = np.zeros((n, p))
                for c in range(rank):
                    U_c = U[:, c].reshape(n, m)
                    V_c = V[:, c].reshape(m, p)
                    W_c = W[:, c].reshape(p, n)
                    # Term = W_c^T @ (U_c^T @ A^T @ B^T?) No.
                    # Standard form: C = sum_c (U_c @ A @ V_c) * W_c^T? No.
                    
                    # Let's use the index form:
                    # C[i, k] += W[k*n+i, c] * sum_j (U[i*m+j, c] * A[i, j]) * ... no.
                    
                    # The tensor decomposition is:
                    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Then C[i, k] = sum_j T[i, j, k] * A[i, j] * B[j, k]
                    #             = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i, j] * B[j, k]
                    #             = sum_c W[k*n+i, c] * (sum_j A[i, j] U[i*m+j, c]) * (sum_j B[j, k] V[j*p+k, c])
                    #             = sum_c W[k*n+i, c] * (U_c^T A_i) * (V_c^T B_k)
                    
                # This is getting too complicated for a quick check.
                # Let's just use np.einsum on the full tensors.
                # C[i, k] = sum_j A[i, j] B[j, k]
                # Our C[i, k] = sum_c W[k*n+i, c] * sum_j (U[i*m+j, c] * A[i, j]) * ... 
                # Actually, the standard Strassen-like decomposition is:
                # C = sum_c (L_c(A) * R_c(B)) * M_c
                # where L_c: R^{n*m} -> R, R_c: R^{m*p} -> R, M_c: R -> R^{n*p}
                
                # Let's just trust the index form and do a full check with 10 random A, B.
                C_calc = np.zeros((n, p))
                for c in range(rank):
                    # L_c(A) = sum_i,j U[i*m+j, c] * A[i, j]
                    # R_c(B) = sum_j,k V[j*p+k, c] * B[j, k]
                    # M_c is the matrix W[:, c].reshape(p, n)^T
                    Lc = np.dot(A.flatten(), U[:, c])
                    Rc = np.dot(B.flatten(), V[:, c])
                    Mc = W[:, c].reshape(p, n).T  # (n, p)
                    C_calc += Lc * Rc * Mc
                
                if not np.allclose(C_calc, C, atol=tol):
                    return False
        return True
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(nm)]
    V = [[0] * rank_schoolbook for _ in range(mp)]
    W = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition using random search
    # We know rank is at least max(n*p, n*m, m*p) = max(32, 4, 32) = 32
    # So we try ranks from 32 to 63.
    
    for r in range(32, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        while attempts < 50 and time.time() < deadline - 10:
            attempts += 1
            # Random initialization
            U_rand = np.random.default_rng(attempts).standard_normal((nm, r))
            V_rand = np.random.default_rng(attempts+1).standard_normal((mp, r))
            W_rand = np.random.default_rng(attempts+2).standard_normal((pn, r))
            
            # Convert to lists for the check function
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if check(U_list, V_list, W_list):
                found = True
                best_rank = r
                best_U, best_V, best_W = U_list, V_list, W_list
                break
        
        if found:
            continue
    
    # Return the best result
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
