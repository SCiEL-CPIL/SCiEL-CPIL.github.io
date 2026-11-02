import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Indices:
    # U: (i, j) -> i*m + j
    # V: (j, k) -> j*p + k
    # W: (k, i) -> k*n + i
    
    # Target Tensor Indices
    # The tensor is non-zero only at (i, j, k) where the term A_ij B_jk contributes to C_ki
    # Wait, C is n x p. C[i, k].
    # The standard schoolbook term for A[i, j] * B[j, k] contributes to C[i, k].
    # So the triplet is (i, j, k).
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i  (Note: W index is k*n + i, which corresponds to C[k, i]? No, C[i, k] is usually row i, col k.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Let's check the standard trace convention.
    # Usually C = A B. C_{ik} = sum_j A_{ij} B_{jk}.
    # If W index is k*n + i, that's column-major for C if C is k x i?
    # Or is it just a specific indexing?
    # Let's assume the target is:
    # Sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if (i,j,k) is a valid triplet, else 0.
    # The valid triplets are all i in [0,n), j in [0,m), k in [0,p).
    # There are n*m*p = 48 such triplets.
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def get_residuals(U, V, W):
        res = {}
        for (i, j, k) in triplets:
            val = 0.0
            # Vectorized dot product
            val = np.dot(U[u_idx(i, j)], V[v_idx(j, k)]) * 0 # Wait, sum over c
            # val = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is not a simple dot product of two vectors because of the third factor.
            # It is the element-wise product of three vectors, then sum.
            # val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[(i, j, k)] = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)]) - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return np.sum([r*r for r in res.values()])

    def optimize(U, V, W, max_iters=100, tol=1e-8):
        # Simple Gradient Descent
        lr = 0.01
        for _ in range(max_iters):
            res = get_residuals(U, V, W)
            err = np.sum([r*r for r in res.values()])
            if err < tol:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k), r in res.items():
                ui = u_idx(i, j)
                vi = v_idx(j, k)
                wi = w_idx(k, i)
                
                # grad_U[ui, c] += 2 * r * V[vi, c] * W[wi, c]
                # grad_V[vi, c] += 2 * r * U[ui, c] * W[wi, c]
                # grad_W[wi, c] += 2 * r * U[ui, c] * V[vi, c]
                
                term_VW = V[vi] * W[wi]
                term_UW = U[ui] * W[wi]
                term_UV = U[ui] * V[vi]
                
                dU[ui] += 2 * r * term_VW
                dV[vi] += 2 * r * term_UW
                dW[wi] += 2 * r * term_UV
                
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        return U, V, W

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = 110 # 10s margin
    
    # Try ranks from 44 to 48
    for r in range(44, 49):
        if time.time() > deadline:
            break
            
        # Initialize with schoolbook terms (first r)
        U_init = np.zeros((n*m, r))
        V_init = np.zeros((m*p, r))
        W_init = np.zeros((p*n, r))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= r:
                        break
                    U_init[u_idx(i, j), col] = 1.0
                    V_init[v_idx(j, k), col] = 1.0
                    W_init[w_idx(k, i), col] = 1.0
                    col += 1
                if col >= r:
                    break
            if col >= r:
                break
        
        # Add small noise to help escape local minima
        noise_level = 0.1
        U_init += noise_level * np.random.randn(n*m, r)
        V_init += noise_level * np.random.randn(m*p, r)
        W_init += noise_level * np.random.randn(p*n, r)
        
        # Run optimization
        U_opt, V_opt, W_opt = optimize(U_init, V_init, W_opt if False else W_init, max_iters=200)
        
        err = total_error(U_opt, V_opt, W_opt)
        
        if err < 1e-6:
            # Success!
            # Convert to list of lists
            return {
                "U": U_opt.tolist(),
                "V": V_opt.tolist(),
                "W": W_opt.tolist()
            }
        else:
            # Keep the best found so far (lowest error)
            if best_U is None or err < best_err:
                best_err = err
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                best_rank = r

    # If no success, return the best found or fallback to schoolbook
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
