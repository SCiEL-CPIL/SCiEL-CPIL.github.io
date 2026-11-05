import numpy as np
import time
import random

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin

    n, m, p = 2, 3, 4
    
    # Precompute the target tensor indices
    # Target T[i, j, k] = delta_{i, k} ? No, T is defined by C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor index: U[i*m+j], V[j*p+k], W[k*n+i]
    # We want sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 if i==k? 
    # Wait, the problem statement says: "output factor follows the official trace convention W[k*n+i,r]"
    # And "exactly decompose the matrix multiplication tensor".
    # The multiplication tensor T has entries T_{i,j,k} = 1 if the term A_{ij}B_{jk} appears in C_{ik}.
    # So for a fixed t, the term is U_{ij}^t V_{jk}^t W_{ki}^t.
    # We require sum_t U_{ij}^t V_{jk}^t W_{ki}^t = 1 if i==k? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T_{i,j,k} is 1 if the product A_{ij}B_{jk} contributes to C_{ik}.
    # This is always true for the term A_{ij}B_{jk} contributing to C_{ik}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} where i is row of A, j is col of A/row of B, k is col of B.
    # The output is C_{i,k}.
    # The term is U_{i,j} V_{j,k} W_{k,i}.
    # We want sum_t U_{i,j}^t V_{j,k}^t W_{k,i}^t = 1 if the term A_{ij}B_{jk} is part of C_{ik}.
    # In standard matrix multiplication, every A_{ij}B_{jk} contributes to C_{ik}.
    # So the target value is 1 for ALL i, j, k?
    # Let's check the verification code in the seed:
    # s = sum U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
    # if abs(s - 1.0) > 1e-9: return False
    # Yes, the target is 1 for all (i,j,k).
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # Vectorized verification
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        # Loop over j is efficient enough for small m
        for j in range(m):
            # U_slice: (n, rank) -> U[i*m+j] for i in 0..n-1
            # V_slice: (p, rank) -> V[j*p+k] for k in 0..p-1
            # W_slice: (p*n, rank) -> W[k*n+i] for k in 0..p-1, i in 0..n-1
            # This indexing for W is tricky to vectorize directly without reshaping.
            # Let's just use numpy einsum or loops for speed.
            for i in range(n):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-4):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def calculate_residual(U, V, W, rank):
        """Calculate the sum of squared errors."""
        residual = 0.0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual += (s - 1.0) ** 2
        return residual

    def try_gradient_descent(rank, max_iter=5000, lr=0.01, momentum=0.9, seed=42):
        """
        Gradient descent with momentum to minimize the residual.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # Momentum buffers
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > max_time - 5:
                break
                
            # Calculate gradients
            # Residual R = sum_{i,j,k} (sum_t U_{ij}^t V_{jk}^t W_{ki}^t - 1)^2
            # dR/dU_{ij}^t = 2 * sum_{j,k} (sum_t ... - 1) * V_{jk}^t W_{ki}^t
            # Note: U index is i*m+j.
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Precompute errors for each (i,j,k)
            # err[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] - 1
            # To save time, we compute these on the fly or vectorize.
            
            for j in range(m):
                for i in range(n):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # s = dot(U[u_idx], V[v_idx] * W[w_idx])
                        prod_vw = V[v_idx] * W[w_idx]
                        s = np.dot(U[u_idx], prod_vw)
                        err = s - 1.0
                        
                        # Gradients
                        # dU[u_idx] += 2 * err * prod_vw
                        # dV[v_idx] += 2 * err * U[u_idx] * W[w_idx]
                        # dW[w_idx] += 2 * err * U[u_idx] * V[v_idx]
                        
                        factor = 2.0 * err
                        grad_U[u_idx] += factor * prod_vw
                        grad_V[v_idx] += factor * (U[u_idx] * W[w_idx])
                        grad_W[w_idx] += factor * (U[u_idx] * V[v_idx])
            
            # Update with momentum
            dU = momentum * dU + lr * grad_U
            dV = momentum * dV + lr * grad_V
            dW = momentum * dW + lr * grad_W
            
            U -= dU
            V -= dV
            W -= dW
            
            # Check residual periodically
            if it % 100 == 0:
                res = calculate_residual(U, V, W, rank)
                if res < best_residual:
                    best_residual = res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if res < 1e-8:
                        break
        
        # Final check
        res = calculate_residual(best_U, best_V, best_W, rank)
        if res < 1e-4: # Threshold for snapping
            U_snap = snap_to_half_integer(best_U)
            V_snap = snap_to_half_integer(best_V)
            W_snap = snap_to_half_integer(best_W)
            
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
        
        return None

    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank

    # Try to find lower rank decompositions
    for rank in range(19, 22):
        if time.time() - start_time > max_time - 20:
            break
            
        found = False
        # Try multiple seeds
        for seed in range(20):
            if time.time() - start_time > max_time - 10:
                break
                
            result = try_gradient_descent(rank, max_iter=2000, lr=0.05, momentum=0.9, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                if verify(U_list, V_list, W_list, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback: return the proven rank-22 construction
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
