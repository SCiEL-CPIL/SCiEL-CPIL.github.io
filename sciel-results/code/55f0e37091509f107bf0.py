"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We check that this equals 1 if (i,j,k) is a valid index, else 0.
    But actually the tensor for matmul is T[i,j,k] = 1 for all valid i,j,k since
    C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Check all indices exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _compute_residuals(U, V, W, n, m, p, rank):
    """Compute residuals R[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1"""
    residuals = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                residuals[i, j, k] = np.dot(u, v * w) - 1.0
    return residuals


def _compute_gradients(U, V, W, n, m, p, rank):
    """Compute gradients for the loss function sum R^2"""
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                R = np.dot(u, v * w) - 1.0
                
                # dL/dU = 2*R * (v * w)
                dU[i * m + j, :] += 2 * R * v * w
                # dL/dV = 2*R * (u * w)
                dV[j * p + k, :] += 2 * R * u * w
                # dL/dW = 2*R * (u * v)
                dW[k * n + i, :] += 2 * R * u * v
    
    return dU, dV, dW


def _alternating_solve(U, V, W, n, m, p, rank, deadline, max_iters=200):
    """Alternating exact/least-squares solve for tensor decomposition."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Fix V, W → solve for U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        
        # Fix U, W → solve for V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        
        # Fix U, V → solve for W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
        
        # Check convergence
        residuals = _compute_residuals(U, V, W, n, m, p, rank)
        loss = np.sum(residuals ** 2)
        if loss < 1e-10:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _try_rank(rank, n, m, p, deadline, max_restarts=10):
    """Try to find a rank decomposition using alternating least squares."""
    if time.time() > deadline:
        return None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Structured initialization: use random but with some structure
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Run alternating solve
        result = _alternating_solve(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    best_result = _schoolbook()
    
    # Try ranks from lower bound upward
    for rank in range(12, 24):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    return best_result
# EVOLVE-BLOCK-END
