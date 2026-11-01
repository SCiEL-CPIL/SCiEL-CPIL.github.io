"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = start_time + 115  # 115 second limit (safety margin)

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54

    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Try structured decompositions with lower rank
    # Strategy: use numerical optimization with random restarts
    # Allocate more time to higher ranks (more likely to succeed)
    for target_rank in range(53, 40, -1):
        if time.time() > DEADLINE - 10:
            break

        # Allocate time based on rank: higher ranks get more time
        remaining = DEADLINE - time.time() - 10
        if remaining < 2:
            break
        time_budget = min(8, max(3, int(remaining * 0.3)))
        
        result = _try_rank_numerical(n, m, p, target_rank, time_budget=time_budget)
        if result is not None:
            r = _verify(n, m, p, result)
            if r == target_rank:
                return result

    # If nothing found, return schoolbook
    return best_result


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(n, m, p, result):
    """Verify that the decomposition is correct. Returns rank if valid, None otherwise."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)

    rank = U.shape[1]

    # Check shapes
    if U.shape != (n * m, rank):
        return None
    if V.shape != (m * p, rank):
        return None
    if W.shape != (p * n, rank):
        return None

    # Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return None

    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return None

    return rank


def _try_rank_numerical(n, m, p, target_rank, time_budget=5):
    """Try to find a decomposition with the given target rank using numerical optimization."""
    start_time = time.time()
    
    U_size = n * m  # 6
    V_size = m * p  # 27
    W_size = p * n  # 18
    
    # Precompute index mappings
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    
    np.random.seed(123)
    
    best_loss = float('inf')
    best_params = None
    
    def compute_loss_and_grads(U, V, W):
        """Compute loss and gradients efficiently."""
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (ui, vi, wi) in triples:
            val = np.dot(U[ui, :], V[vi, :] * W[wi, :])
            error = val - 1.0
            loss += error * error
            
            grad_U[ui, :] += 2 * error * V[vi, :] * W[wi, :]
            grad_V[vi, :] += 2 * error * U[ui, :] * W[wi, :]
            grad_W[wi, :] += 2 * error * U[ui, :] * V[vi, :]
        
        loss /= len(triples)
        return loss, grad_U, grad_V, grad_W
    
    def polish(U, V, W, max_steps=500, tol=1e-12):
        """High-precision polish pass with adaptive learning rate."""
        lr = 0.05
        momentum = 0.9
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        
        for step in range(max_steps):
            if time.time() - start_time > time_budget:
                break
            
            loss, grad_U, grad_V, grad_W = compute_loss_and_grads(U, V, W)
            
            if loss < tol:
                break
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 0:
                lr = min(lr, 0.1 / grad_norm)
            
            v_U = momentum * v_U - lr * grad_U
            v_V = momentum * v_V - lr * grad_V
            v_W = momentum * v_W - lr * grad_W
            
            U = U + v_U
            V = V + v_V
            W = W + v_W
        
        return U, V, W
    
    # Strategy 1: Structured initialization based on schoolbook pattern
    # Use a subset of the schoolbook basis vectors
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        if restart == 0:
            # Structured init: pick target_rank schoolbook columns
            idx = list(range(target_rank))
            U = np.zeros((U_size, target_rank))
            V = np.zeros((V_size, target_rank))
            W = np.zeros((W_size, target_rank))
            for c, col_idx in enumerate(idx):
                i, j, k = col_idx // (m*p), (col_idx // p) % m, col_idx % p
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
        else:
            # Random init with varying scales
            scale = 0.3 + 0.2 * (restart % 3)
            U = np.random.randn(U_size, target_rank) * scale
            V = np.random.randn(V_size, target_rank) * scale
            W = np.random.randn(W_size, target_rank) * scale
        
        # Gradient descent with momentum
        lr = 0.02
        momentum = 0.9
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        
        for step in range(3000):
            if time.time() - start_time > time_budget:
                break
            
            loss, grad_U, grad_V, grad_W = compute_loss_and_grads(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                break
            
            v_U = momentum * v_U - lr * grad_U
            v_V = momentum * v_V - lr * grad_V
            v_W = momentum * v_W - lr * grad_W
            
            U = U + v_U
            V = V + v_V
            W = W + v_W
        
        if best_loss < 1e-8:
            break
    
    if best_params is not None and best_loss < 1e-6:
        U_opt, V_opt, W_opt = best_params
        
        # Polish pass
        U_opt, V_opt, W_opt = polish(U_opt, V_opt, W_opt)
        
        # Snap to half-integers
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
        # Verify the snapped version
        valid = True
        for (ui, vi, wi) in triples:
            val = np.dot(U_snap[ui, :], V_snap[vi, :] * W_snap[wi, :])
            if abs(val - 1.0) > 1e-6:
                valid = False
                break
        
        if valid:
            return {
                "U": U_snap.tolist(),
                "V": V_snap.tolist(),
                "W": W_snap.tolist()
            }
    
    return None


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
