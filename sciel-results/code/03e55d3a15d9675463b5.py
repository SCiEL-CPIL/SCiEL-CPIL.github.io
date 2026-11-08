import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T_{i,j,k} = delta_{j, j'} is not quite right.
    # The condition is: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j, j'}
    
    # Build the target tensor E of shape (n*m, m*p, p*n) ? No.
    # The error is on the bilinear map.
    # We want to minimize sum_{i,j,j',k} (sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] - delta_{j,j'})^2
    
    # This is a tensor of shape (n*m, m*p, p*n) with entries E[i*m+j, j'*p+k, k*n+i] = delta_{j,j'}
    # Note: The index in W is k*n+i. The index in V is j'*p+k.
    
    # Let's construct the target tensor T_target of shape (n*m, m*p, p*n)
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    # U index: i*m + j
                    # V index: j_prime*p + k
                    # W index: k*n + i
                    # Target value: 1 if j == j_prime else 0
                    T_target[i*m + j, j_prime*p + k, k*n + i] = 1.0 if j == j_prime else 0.0

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank decomposition using optimization
    # We try ranks 23, 24, ... up to 26
    # Given time constraints, we only try rank 23 with a few restarts.
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    for rank in [23]:
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Simple gradient descent / ALS
            # We minimize || T_target - sum_r U_r V_r W_r ||_F^2
            # This is a 3-way tensor decomposition.
            # We can use ALS: fix V, W, solve for U, etc.
            
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Update U
                # Residual R = T_target - sum_{r'} U_r' V_r' W_r'
                # For each r, we want to minimize || R + U_r V_r W_r - U_old_r V_r W_r ||^2 ?
                # Actually, standard ALS for CP decomposition:
                # Fix V, W. Solve for U.
                # The term for U_r is U_r (V_r * W_r)^T ?
                # The tensor is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
                # Here our indices are flattened.
                # T_target[a, b, c] where a=i*m+j, b=j'*p+k, c=k*n+i.
                # We want T_target[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
                
                # Residual R = T_target - current_estimate
                # current_estimate = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Update U:
                # Minimize || R + U_r V_r W_r - U_old_r V_r W_r ||^2
                # This is not standard ALS because the indices are coupled in the target?
                # No, the target is a fixed tensor.
                # So it IS standard CP decomposition of T_target.
                
                # Compute current estimate
                # Efficient computation:
                # M1 = V.T @ W -> (rank, m*p, p*n) ? No.
                # V is (m*p, rank), W is (p*n, rank).
                # V.T is (rank, m*p). W.T is (rank, p*n).
                # We need to form the tensor.
                # Let's use einsum for speed in small dimensions.
                
                if it % 10 == 0:
                    est = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(T_target - est)
                    # print(f"Rank {rank}, Restart {restart}, It {it}, Err {err}")
                
                # Update U
                # For each r, U[:, r] should be the projection of (T_target - est_without_r) onto (V[:, r] * W[:, r])
                # Let M = V.T @ W ? No.
                # The factor for U_r is the outer product of V_r and W_r.
                # V_r is a vector of length m*p. W_r is a vector of length p*n.
                # Outer product P_r = V_r[:, None] * W_r[None, :] -> shape (m*p, p*n)
                # We want to minimize || T_target[:,:,r_part] ... ||
                # Actually, for fixed V, W, the optimal U_r is:
                # U[:, r] = (T_target * (V_r * W_r)) . reshape(n*m, -1) . dot( (V_r*W_r).reshape(-1) ) / norm^2
                # Let's do it simply.
                
                for r in range(rank):
                    # Compute factor F = V[:, r] * W[:, r] ? No, V and W are vectors.
                    # The term in the sum is U[:, r] * V[:, r] * W[:, r] (broadcasting)
                    # We want to update U[:, r].
                    # Let Residual = T_target - (est - U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :])
                    # This is expensive.
                    
                    # Simpler:
                    # Let F = np.outer(V[:, r], W[:, r])
                    # Then T_target can be seen as a matrix of shape (n*m, m*p*p*n)
                    # T_mat = T_target.reshape(n*m, -1)
                    # F_mat = F.reshape(-1)
                    # U[:, r] = (T_mat * F_mat[None, :]) .dot(F_mat) / (F_mat.dot(F_mat))
                    
                    F = np.outer(V[:, r], W[:, r])
                    F_flat = F.flatten()
                    norm_sq = np.dot(F_flat, F_flat)
                    if norm_sq < 1e-10:
                        continue
                    # T_target is (n*m, m*p, p*n)
                    # T_mat is (n*m, m*p*p*n)
                    # We need to contract T_target with F over the last two dims.
                    # T_contracted = np.einsum('abc, bc -> a', T_target, F)
                    T_contracted = np.tensordot(T_target, F, axes=([1, 2], [0, 1]))
                    U[:, r] = T_contracted / norm_sq

                # Update V
                for r in range(rank):
                    F = np.outer(U[:, r], W[:, r])
                    F_flat = F.flatten()
                    norm_sq = np.dot(F_flat, F_flat)
                    if norm_sq < 1e-10:
                        continue
                    T_contracted = np.tensordot(T_target, F, axes=([0, 2], [0, 1]))
                    V[:, r] = T_contracted / norm_sq

                # Update W
                for r in range(rank):
                    F = np.outer(U[:, r], V[:, r])
                    F_flat = F.flatten()
                    norm_sq = np.dot(F_flat, F_flat)
                    if norm_sq < 1e-10:
                        continue
                    T_contracted = np.tensordot(T_target, F, axes=([0, 1], [0, 1]))
                    W[:, r] = T_contracted / norm_sq

            # After optimization, check for exactness
            est = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_target - est)
            if err < 1e-5:
                # Snap to nearest half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                est_snapped = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
                err_snapped = np.linalg.norm(T_target - est_snapped)
                if err_snapped < 1e-5:
                    # Verify with random integer matrices
                    if verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
                        best_rank = rank
                        best_U = U_snapped
                        best_V = V_snapped
                        best_W = W_snapped
                        break
        
        if best_rank == rank:
            break

    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}

def verify_decomposition(U, V, W, n, m, p):
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    rank = U.shape[1]
    
    # Test with a few random integer matrices
    for _ in range(10):
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        # Compute C_decomposed
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            # U_r is a vector of length n*m. Reshape to (n, m)
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n) # Wait, W index is k*n+i. So W[k*n+i] is the coefficient for C[i,k]?
            # In the decomposition: C[i,k] = sum_r (sum_ij U[i*m+j, r] A[i,j]) (sum_jk V[j*p+k, r] B[j,k]) W[k*n+i, r]
            # Let a_r = sum_ij U[i*m+j, r] A[i,j]
            # Let b_r = sum_jk V[j*p+k, r] B[j,k]
            # Then C[i,k] += a_r * b_r * W[k*n+i, r]
            
            a_r = np.sum(U[:, r].reshape(n, m) * A)
            b_r = np.sum(V[:, r].reshape(m, p) * B)
            
            # Add to C
            # W[:, r] is a vector of length p*n. Index is k*n+i.
            # We need to add a_r * b_r * W[k*n+i, r] to C[i,k].
            # Loop over i, k
            for i in range(n):
                for k in range(p):
                    C_decomposed[i, k] += a_r * b_r * W[k*n + i, r]
        
        if not np.allclose(C_expected, C_decomposed, atol=1e-5):
            return False
    return True

# EVOLVE-BLOCK-END
