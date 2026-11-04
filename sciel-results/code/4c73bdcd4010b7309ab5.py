"""Search for rank-minimal exact tensor decomposition of 3x3x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
n, m, p = DIMENSIONS


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_strassen_block():
        """
        Try a Strassen-like block decomposition.
        For 3x3 * 3x6, we can think of it as computing three 3x3 * 3x2 blocks.
        Or use a Kronecker structure.
        
        Strategy: Use the fact that 3x3x3 has rank 23 (best known).
        For 3x3x6, we can potentially use rank around 23*2 = 46 or less with sharing.
        
        Let's try a structured approach:
        - Split the 6 columns into two groups of 3
        - Use a 3x3x3 decomposition for each, with shared U factors
        """
        if time.time() > deadline - 20:
            return None

        # Try: use two independent 3x3x3 decompositions (rank 23 each = 46 total)
        # But we can do better by sharing structure.
        
        # Let's try a direct algebraic approach for smaller ranks.
        # The system is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # This is 54 equations. For rank r, we have r*(9+18+18) = 45r unknowns.
        # For r >= 2, we have more unknowns than equations (underdetermined).
        # But it's nonlinear (trilinear), so it's not trivial.
        
        # Let's try a gradient-based approach with multiple restarts.
        return None

    def try_gradient_optimization(rank, max_iters=50000, seed=42):
        """
        Use gradient-based optimization to find a rank-r decomposition.
        Minimize the residual: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        """
        if time.time() > deadline - 10:
            return None

        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5

        def compute_residual(U, V, W):
            """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
            R = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        R[i, j, k] = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i]) - 1.0
            return R

        def compute_loss(U, V, W):
            R = compute_residual(U, V, W)
            return np.sum(R ** 2)

        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = compute_loss(U, V, W)
        
        # Simple gradient descent with adaptive learning rate
        lr = 0.01
        patience = 0
        max_patience = 5000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            R = compute_residual(U, V, W)
            loss = np.sum(R ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
            
            if best_loss < 1e-10:
                break
            
            # Compute gradients
            # d/dU[i*m+j, r] = 2 * R[i,j,k] * V[j*p+k, r] * W[k*n+i, r] summed over k
            # Actually, for each (i,j,r): dU[i*m+j,r] = 2 * sum_k R[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        # dU[i*m+j, r] = 2 * sum_k R[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        dU[i * m + j, r] = 2.0 * np.sum(R[i, j, :] * V[j * p:, r] * W[:, r][np.arange(p) * n + i])
                        
                        # dV[j*p+k, r] = 2 * sum_i R[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        # dW[k*n+i, r] = 2 * sum_j R[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                    
                    for k in range(p):
                        # dV[j*p+k, r] for each r
                        for r in range(rank):
                            dV[j * p + k, r] = 2.0 * np.sum(R[:, j, k] * U[:, r][np.arange(n) * m + j] * W[k * n:, r][np.arange(n)])
                            dW[k * n + i, r] = 2.0 * np.sum(R[i, :, k] * U[i * m:, r][np.arange(m)] * V[j * p + k, r])

            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            new_loss = compute_loss(U, V, W)
            if new_loss > loss * 1.1:
                lr *= 0.5
                U += lr * dU  # rollback
                V += lr * dV
                W += lr * dW
            elif new_loss < loss * 0.9:
                lr *= 1.1

        if best_loss < 1e-6:
            # Try to round to half-integers
            for scale in [1, 2, 4]:
                U_round = np.round(best_U * scale) / scale
                V_round = np.round(best_V * scale) / scale
                W_round = np.round(best_W * scale) / scale
                if verify(U_round, V_round, W_round, tol=1e-6):
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
            # Try integer rounding
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if verify(U_int, V_int, W_int, tol=1e-6):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        return None

    def try_structured_decomposition():
        """
        Try structured decompositions based on known results.
        
        For 3x3 * 3x6, one approach:
        - View it as three 3x3 * 3x2 multiplications
        - Or use a Kronecker product structure
        
        Let's try: rank = 2 * rank(3x3x3) = 46 (using two independent 3x3x3 decomp)
        But we can do better.
        
        Actually, let's try a simpler structured approach:
        Use the fact that the tensor has a specific structure.
        
        For the 3x3x3 case, Strassen's algorithm gives rank 7 for 2x2x2.
        For 3x3x3, the best known is 23.
        
        Let me try a direct construction for small ranks using the algebraic structure.
        """
        if time.time() > deadline - 30:
            return None

        # Try rank from a reasonable lower bound up
        # Lower bound: max(n*p, m*p, n*m) = max(18, 18, 9) = 18
        # But structural constraints likely push it higher.
        
        # Let's try ranks: 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52
        for rank in range(20, 55, 2):
            if time.time() > deadline - 15:
                break
            
            # Try gradient optimization with different seeds
            for seed in [42, 123, 456, 789]:
                if time.time() > deadline - 10:
                    break
                
                result = try_gradient_optimization(rank, max_iters=20000, seed=seed)
                if result is not None:
                    U_list, V_list, W_list = result
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try structured decomposition first
    result = try_structured_decomposition()
    if result is not None:
        return result

    # Fallback: return schoolbook
    return make_schoolbook()
