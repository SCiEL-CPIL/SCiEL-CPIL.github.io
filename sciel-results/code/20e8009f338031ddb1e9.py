"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Tensor identity: sum_r W[k*n+i, r] * U[i1*m+j1, r] * V[j2*p+k2, r] = δ_{i,i1} δ_{j1,j2} δ_{k,k2}
    
    # Build target tensor T[i1, j1, j2, k2] = δ_{i,i1} δ_{j1,j2} δ_{k,k2}
    # But we need to express this in terms of (i,k) output and (i1,j1) and (j2,k2) inputs.
    # Actually, the target is: for each (i,k), the bilinear form should give sum_j A[i,j]B[j,k]
    
    # Let me define the target as a 4-index tensor:
    # T[i1, j1, j2, k2] where the output (i,k) is determined by i1 and k2 when the deltas match.
    # Actually, it's easier to think of it as: for each output (i,k), we need:
    # sum_r W[k*n+i, r] * U[i1*m+j1, r] * V[j2*p+k2, r] = δ_{i,i1} δ_{j1,j2} δ_{k,k2}
    
    # Let me build a 4D target tensor: target[i, i1, j1, j2, k2, k] ... this is getting complicated.
    # 
    # Alternative: build the target as a 3D tensor with indices (output_idx, in1_idx, in2_idx)
    # where output_idx = k*n+i (12 values), in1_idx = i1*m+j1 (9 values), in2_idx = j2*p+k2 (12 values)
    # target[out, in1, in2] = 1 if out corresponds to (i,k), in1 to (i,j), in2 to (j,k) with matching indices
    
    # out_idx = k*n + i, in1_idx = i*m + j, in2_idx = j*p + k
    # target[out, in1, in2] = 1 if i_from_out == i_from_in1 AND j_from_in1 == j_from_in2 AND k_from_in2 == k_from_out
    
    target = np.zeros((p*n, n*m, m*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = k * n + i
                in1_idx = i * m + j
                in2_idx = j * p + k
                target[out_idx, in1_idx, in2_idx] = 1.0
    
    # Decomposition: target[out, in1, in2] = sum_r W[out, r] * U[in1, r] * V[in2, r]
    
    def try_rank(r, max_iter=3000, lr=0.02, seed=42):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute current estimate: est[out, in1, in2] = sum_r W[out,r] * U[in1,r] * V[in2,r]
            est = np.zeros_like(target)
            for d in range(r):
                est += np.einsum('o,i,j->oij', W[:,d], U[:,d], V[:,d])
            
            R = target - est
            err = np.linalg.norm(R)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Gradients
            # dL/dW[out,r] = sum_{in1,in2} R[out,in1,in2] * U[in1,r] * V[in2,r]
            # dL/dU[in1,r] = sum_{out,in2} R[out,in1,in2] * W[out,r] * V[in2,r]
            # dL/dV[in2,r] = sum_{out,in1} R[out,in1,in2] * W[out,r] * U[in1,r]
            
            grad_W = np.einsum('oij,ir,jr->or', R, U, V)
            grad_U = np.einsum('oij,or,jr->ir', R, W, V)
            grad_V = np.einsum('oij,or,ir->jr', R, W, U)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Final error
        est = np.zeros_like(target)
        for d in range(r):
            est += np.einsum('o,i,j->oij', W[:,d], U[:,d], V[:,d])
        err = np.linalg.norm(target - est)
        
        if err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None
    
    # Try to find minimal rank
    # Lower bound considerations: the target tensor has a certain structure.
    # Let's try ranks from 15 to 36.
    
    best_result = None
    for r in range(15, 37):
        if time.time() > deadline:
            break
        for seed in [42, 123, 456, 789, 1011]:
            if time.time() > deadline:
                break
            result = try_rank(r, max_iter=2000, lr=0.05, seed=seed)
            if result is not None:
                best_result = (r, result)
                break
        if best_result is not None:
            break
    
    if best_result is not None:
        r, (U, V, W, err) = best_result
        # Try to round to integers or half-integers
        for scale in [1, 2, 4]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            est = np.zeros_like(target)
            for d in range(r):
                est += np.einsum('o,i,j->oij', W_r[:,d], U_r[:,d], V_r[:,d])
            if np.allclose(est, target, atol=1e-9):
                U_list = [[float(U_r[i,d]) for d in range(r)] for i in range(n*m)]
                V_list = [[float(V_r[i,d]) for d in range(r)] for i in range(m*p)]
                W_list = [[float(W_r[i,d]) for d in range(r)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Return numerical solution (may not be exact, but best effort)
        U_list = [[float(U[i,d]) for d in range(r)] for i in range(n*m)]
        V_list = [[float(V[i,d]) for d in range(r)] for i in range(m*p)]
        W_list = [[float(W[i,d]) for d in range(r)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
