import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * p  # 90
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # We search for a rank r decomposition.
    # We try ranks from 89 down to 80.
    # If we find a valid integer/half-integer decomposition, we return it.
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Schoolbook decomposition
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find a lower rank using ALS
    # We will try a few random initializations for each rank.
    
    for rank in range(schoolbook_rank - 1, 79, -1):
        for trial in range(5):
            np.random.seed(42 + rank * 10 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # ALS iterations
            converged = False
            for it in range(100):
                # Solve for W
                # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # We can reshape T to (n*m*p,).
                # We want to solve A_w @ w_flat = t_flat
                # Where w_flat is W flattened (pn * rank).
                # A_w is (n*m*p, pn * rank).
                
                # Constructing A_w explicitly is expensive.
                # Instead, we can use the fact that for a fixed (i, j), 
                # the equation over k is:
                # T[i, j, :] = (U[i*m+j, :] * V[j*p+k, :]) @ W.T ? No.
                
                # Let's use a simple gradient step for speed and simplicity.
                # Loss = 0.5 * || T - sum_c U_c \otimes V_c \otimes W_c ||^2
                # Gradient wrt W:
                # dL/dW[k*n+i, c] = sum_{j} ( T[i, j, k] - pred[i, j, k] ) * U[i*m+j, c] * V[j*p+k, c]
                
                # Compute prediction
                # pred = sum_c U[:, c] * V[:, c] * W[:, c] (broadcasting)
                # This is hard to vectorize directly without loops.
                
                # Given the small size (90 entries), we can use a loop for gradient.
                pred = np.zeros((n, m, p))
                for c in range(rank):
                    # u_c is (n*m, 1) -> (n, m, 1)
                    # v_c is (m*p, 1) -> (1, m, p)
                    # w_c is (p*n, 1) -> (n, 1, p)
                    u_c = U[:, c].reshape(n, m, 1)
                    v_c = V[:, c].reshape(1, m, p)
                    w_c = W[:, c].reshape(n, 1, p)
                    pred += u_c * v_c * w_c
                
                error = T - pred
                loss = np.sum(error**2)
                
                if loss < 1e-6:
                    converged = True
                    break
                
                # Update W
                grad_W = np.zeros_like(W)
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m, 1)
                    v_c = V[:, c].reshape(1, m, p)
                    # grad_W[k*n+i, c] = sum_j error[i, j, k] * u_c[i, j, 0] * v_c[0, j, k]
                    # We can compute this with einsum
                    # grad_w_c = np.einsum('ijk,ij,jk->ki', error, u_c, v_c)
                    grad_w_c = np.einsum('ijk,ij,jk->ki', error, u_c, v_c)
                    grad_W[:, c] = grad_w_c.reshape(pn)
                
                # Update V
                grad_V = np.zeros_like(V)
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m, 1)
                    w_c = W[:, c].reshape(n, 1, p)
                    # grad_V[j*p+k, c] = sum_i error[i, j, k] * u_c[i, j, 0] * w_c[i, 0, k]
                    # grad_v_c = np.einsum('ijk,ij,ik->jk', error, u_c, w_c)
                    grad_v_c = np.einsum('ijk,ij,ik->jk', error, u_c, w_c)
                    grad_V[:, c] = grad_v_c.reshape(mp)
                
                # Update U
                grad_U = np.zeros_like(U)
                for c in range(rank):
                    v_c = V[:, c].reshape(1, m, p)
                    w_c = W[:, c].reshape(n, 1, p)
                    # grad_U[i*m+j, c] = sum_k error[i, j, k] * v_c[0, j, k] * w_c[i, 0, k]
                    # grad_u_c = np.einsum('ijk,jk,ik->ij', error, v_c, w_c)
                    grad_u_c = np.einsum('ijk,jk,ik->ij', error, v_c, w_c)
                    grad_U[:, c] = grad_u_c.reshape(nm)
                
                # Gradient Descent Step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
            
            if converged or loss < 1e-4:
                # Check if entries are close to integers/half-integers
                # Snap to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify the snapped solution
                pred_snap = np.zeros((n, m, p))
                for c in range(rank):
                    u_c = U_snap[:, c].reshape(n, m, 1)
                    v_c = V_snap[:, c].reshape(1, m, p)
                    w_c = W_snap[:, c].reshape(n, 1, p)
                    pred_snap += u_c * v_c * w_c
                
                if np.allclose(pred_snap, T, atol=1e-6):
                    best_rank = rank
                    best_decomp = (U_snap, V_snap, W_snap)
                    break
        
        if best_rank < schoolbook_rank:
            break
    
    def to_list(mat):
        return mat.tolist()
    
    U, V, W = best_decomp
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
