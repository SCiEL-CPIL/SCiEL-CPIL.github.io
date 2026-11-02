import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p), dtype=float)

    # Search for lower rank
    # Known lower bound for M(2,5,6) is not trivial, but we know it's < 60.
    # Strassen-like algorithms might get close to ~40-50.
    # Let's search from 55 down to 35.
    for r in range(55, 34, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        # Use alternating least squares (ALS) which is more robust for tensor decomposition
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            converged = False
            # ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                # U[:, c] should minimize || T - sum_c' U_c V_c W_c ||_F^2
                # For a fixed c, the optimal U_c is given by solving a linear system or using gradient descent.
                # Here we use a simple power method / coordinate descent approach similar to inspiration 2.
                
                c = it % r  # Cycle through columns
                
                # Compute residual excluding column c
                # Current sum S = sum_{c'} U[:, c'] V[:, c'] W[:, c']
                # We want to update U[:, c] to best fit T - (S - U[:, c] V[:, c] W[:, c])
                
                # Efficient calculation:
                # Let R = T - sum_{c' != c} U[:, c'] V[:, c'] W[:, c']
                # We want U_new = argmin || R - U_new V[:, c] W[:, c] ||^2
                # This is a least squares problem.
                
                # Reshape for einsum
                V_c = V[:, c]
                W_c = W[:, c]
                
                # Compute current sum S
                # S[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # Using einsum: 'ic,jc,kc->ijk'
                # But we need to exclude c.
                
                # Let's compute the full sum and subtract the c-th term
                # Full sum:
                S_full = np.einsum('ic,jc,kc->ijk', U, V, W)
                # Term c:
                S_c = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                
                R = T - (S_full - S_c)
                
                # Now we want to find u, v, w (we are updating u here, keeping v,w fixed)
                # Minimize || R - u * (v o w) ||^2
                # Let A = v o w. Then we want u to minimize || R - u A ||^2
                # This is a scalar least squares problem if u is a scalar? No, u is a vector.
                # Wait, U[:, c] is a vector of length n*m.
                # The term is sum_i u_i * (v o w)_i? No.
                # The tensor product is u_i v_j w_k.
                # So for fixed v, w, the term is sum_i u_i * (v o w)_{i, j, k}?
                # No, U is indexed by (i*m+j).
                # Let's flatten indices.
                # U_flat[idx_u, c], V_flat[idx_v, c], W_flat[idx_w, c]
                # Tensor T_flat[idx_u, idx_v, idx_w]
                
                # Let's reshape U, V, W to 2D arrays where columns are the rank components.
                # U: (n*m, r)
                # V: (m*p, r)
                # W: (p*n, r)
                
                # The contribution of rank component c is:
                # T_c = U[:, c] outer V[:, c] outer W[:, c]
                # This is a tensor of size (n*m, m*p, p*n).
                
                # We want to update U[:, c] to minimize || T - (S_full - T_c + U_new[:, c] outer V[:, c] outer W[:, c]) ||^2
                # Let R = T - (S_full - T_c)
                # Minimize || R - U_new outer V_c outer W_c ||^2
                
                # This is a linear least squares problem for U_new.
                # Let A be a matrix where A[:, idx] = (V_c outer W_c).flat
                # Then U_new is the solution to U_new A = R.flat
                # U_new = R.flat @ A^+ (pseudo-inverse)
                
                # Construct A
                # V_c: (m*p,), W_c: (p*n,)
                # Outer product: (m*p, p*n)
                # Flatten: (m*p * p*n,)
                # A is (n*m, m*p * p*n)
                
                # This might be expensive. m*p = 30, p*n = 12. 30*12 = 360.
                # n*m = 10.
                # A is 10 x 360.
                # R.flat is 10 * 30 * 12 = 3600.
                # This is a standard linear system.
                
                # Let's try to solve it.
                # However, constructing A every iteration might be slow.
                # Alternative: use gradient descent with a few steps.
                
                # Let's stick to the power method / coordinate descent from inspiration 2 as it was simpler and potentially faster to implement correctly.
                
                # Power method update for U[:, c]
                # u_new = R contracted with V_c, W_c
                # R[i, j, k] * V_c[j] * W_c[k] -> sum over j, k
                # u_new[i] = sum_{j,k} R[i, j, k] V_c[j] W_c[k]
                
                # Reshape R to (n, m, p)
                R_3d = R.reshape(n, m, p)
                V_c_2d = V_c.reshape(m, p)
                W_c_2d = W_c.reshape(p, n)
                
                # u_new[i] = sum_{j,k} R_3d[i, j, k] V_c_2d[j, k] W_c_2d[k, i]
                # Note: W_c is indexed by (k, n) in the tensor, but V_c by (m, p).
                # Wait, the indices in the tensor T are (i, j, k).
                # U is indexed by i*m+j.
                # V is indexed by j*p+k.
                # W is indexed by k*n+i.
                
                # So V_c[j*p+k] corresponds to V_c_2d[j, k].
                # W_c[k*n+i] corresponds to W_c_2d[k, i].
                
                # u_new[i*m+j] = sum_{k} R_3d[i, j, k] V_c_2d[j, k] W_c_2d[k, i]
                
                # Let's compute this.
                # First contract R with V_c over j, k? No, j is fixed in U.
                # u_new[i*m+j] = sum_k R_3d[i, j, k] * V_c_2d[j, k] * W_c_2d[k, i]
                
                # This can be done with einsum.
                # 'ijk,jk,ki->ij'
                u_new_2d = np.einsum('ijk,jk,ki->ij', R_3d, V_c_2d, W_c_2d)
                u_new = u_new_2d.flatten()
                
                # Normalize? No, we want the optimal scale.
                # The optimal u is given by the projection.
                # The power method direction is correct, but we need the right scale.
                # Let's find the scale s such that u_new = s * u_dir minimizes the error.
                # Actually, the least squares solution is better.
                
                # Let's try the least squares approach for U[:, c].
                # It's more robust.
                
                # Construct A
                # A[:, idx] where idx corresponds to (j, k) in V and (k, i) in W?
                # No, the term is sum_{j,k} U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # For fixed c, this is sum_{j,k} U[i*m+j, c] * (V_c[j*p+k] * W_c[k*n+i])
                
                # Let B[i, j, k] = V_c[j*p+k] * W_c[k*n+i]
                # Then the term is sum_{j,k} U[i*m+j, c] B[i, j, k]
                
                # We want to minimize || R - U_new B ||^2
                # This is a linear system U_new * A = R_flat
                # Where A is constructed from B.
                
                # Let's form A.
                # A has shape (n*m, n*m*p) ? No.
                # The unknowns are U_new[0..n*m-1].
                # The equations are for each (i, j, k) in R.
                # There are n*m*p equations.
                # Each equation involves U_new[i*m+j].
                
                # So A is (n*m*p, n*m).
                # A[(i, j, k), i*m+j] = B[i, j, k]
                
                # Let's construct A.
                A = np.zeros((n*m*p, n*m))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A[(i*m+j)*p + k, i*m+j] = V_c[j*p+k] * W_c[k*n+i]
                
                R_flat = R.flatten()
                
                # Solve least squares
                try:
                    U_new, residuals, rank_A, s = np.linalg.lstsq(A, R_flat, rcond=None)
                    U[:, c] = U_new
                except:
                    pass

                # Check convergence
                S_full_new = np.einsum('ic,jc,kc->ijk', U, V, W)
                if np.linalg.norm(T - S_full_new) < 1e-9:
                    converged = True
                    break

            if converged:
                # We have a real-valued solution.
                # Try to round to half-integers.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    # Convert to lists
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
