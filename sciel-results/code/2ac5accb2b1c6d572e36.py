"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    def verify_decomposition(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        # T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_comp, T, atol=1e-8)

    def try_rank(R, max_time_per_rank=10.0):
        """Attempt to find a decomposition of rank R using gradient descent."""
        nonlocal best_result, best_rank
        
        size_U = nm
        size_V = mp
        size_W = pn
        
        def compute_error(U, V, W):
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.sum((T_comp - T) ** 2)

        def compute_grads(U, V, W):
            # T_comp = U @ V^T @ W^T (conceptually, via einsum)
            # dT_comp/dU = V @ W^T (broadcasting over i)
            # Actually: T_comp[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # dT_comp/dU[i,r] = sum_{j,k} (T_comp[i,j,k] - T[i,j,k]) V[j,r] W[k,r]
            # This is expensive. Let's use the identity:
            # E = sum (T_comp - T)^2
            # dE/dU[i,r] = 2 * sum_{j,k} (T_comp[i,j,k] - T[i,j,k]) V[j,r] W[k,r]
            # Let D = T_comp - T
            # dE/dU[i,r] = 2 * sum_{j,k} D[i,j,k] V[j,r] W[k,r]
            # This can be written as 2 * (D * (V[:, :, None] * W[None, :, :])) summed over j,k
            # Or using einsum: 2 * np.einsum('ijk,jr,kr->ir', D, V, W)
            
            D = np.einsum('ir,jr,kr->ijk', U, V, W) - T
            grad_U = 2 * np.einsum('ijk,jr,kr->ir', D, V, W)
            grad_V = 2 * np.einsum('ijk,ir,kr->jr', D, U, W)
            grad_W = 2 * np.einsum('ijk,ir,jr->kr', D, U, V)
            return grad_U, grad_V, grad_W, compute_error(U, V, W)

        # Multiple random restarts
        for attempt in range(30):
            if time.time() > deadline - 2.0:
                return False
            
            np.random.seed(123 + R * 100 + attempt)
            
            # Initialize factors
            U = np.random.randn(size_U, R) * 0.1
            V = np.random.randn(size_V, R) * 0.1
            W = np.random.randn(size_W, R) * 0.1
            
            # Gradient descent
            lr = 0.05
            for step in range(500):
                if time.time() > deadline - 2.0:
                    break
                
                grad_U, grad_V, grad_W, err = compute_grads(U, V, W)
                
                if err < 1e-10:
                    break
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: normalize factors to prevent explosion
                # U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
                # V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-10)
                # W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
            
            if err < 1e-6:
                # Try rounding to integers or half-integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if verify_decomposition(U_int, V_int, T):
                    best_rank = R
                    best_result = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    return True
                
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                if verify_decomposition(U_half, V_half, T):
                    best_rank = R
                    best_result = {
                        "U": U_half.tolist(),
                        "V": V_half.tolist(),
                        "W": W_half.tolist()
                    }
                    return True

        return False

    # Try candidate ranks in increasing order
    candidate_ranks = [30, 32, 34, 35, 36, 38, 40, 42, 45]
    candidate_ranks = [r for r in candidate_ranks if r < best_rank]
    
    for R in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
        
        success = try_rank(R)
        if success:
            break

    return best_result
# EVOLVE-BLOCK-END
