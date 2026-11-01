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

    # Strategy: Solve 2x3x3 and replicate 3 times for 2x3x9
    # 2x3x9 can be viewed as 3 independent 2x3x3 multiplications if we split the 9 columns into 3 groups of 3.
    # If rank(2x3x3) = R, then rank(2x3x9) <= 3*R.
    # Schoolbook 2x3x3 is 18. 3*18 = 54.
    # If we find R < 18, we improve.
    
    # Try ranks for 2x3x3
    for r_small in range(12, 18):
        if time.time() > DEADLINE - 5:
            break
        
        # Time budget for each rank attempt
        time_budget = 15
        result_small = _find_2x3x3_decomp(r_small, time_budget)
        
        if result_small is not None:
            # Construct 2x3x9 from 2x3x3
            result_large = _expand_2x3x3_to_2x3x9(result_small)
            if result_large is not None:
                r_large = _verify(n, m, p, result_large)
                if r_large is not None:
                    return result_large

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


def _find_2x3x3_decomp(target_rank, time_budget=10):
    """Find a rank decomposition for 2x3x3 matrix multiplication."""
    start_time = time.time()
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
                
    np.random.seed(42)
    
    best_loss = float('inf')
    best_params = None
    
    def compute_loss_and_grads(U, V, W):
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
    
    def polish(U, V, W, max_steps=1000, tol=1e-12):
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

    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random init
        scale = 0.5
        U = np.random.randn(U_size, target_rank) * scale
        V = np.random.randn(V_size, target_rank) * scale
        W = np.random.randn(W_size, target_rank) * scale
        
        lr = 0.05
        momentum = 0.9
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        
        for step in range(5000):
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
            
            # Adaptive LR
            if step % 100 == 0:
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 0:
                    lr = min(0.1, max(0.01, 0.1 / grad_norm))
        
        if best_loss < 1e-8:
            break
    
    if best_params is not None and best_loss < 1e-6:
        U_opt, V_opt, W_opt = best_params
        U_opt, V_opt, W_opt = polish(U_opt, V_opt, W_opt)
        
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
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


def _expand_2x3x3_to_2x3x9(small_result):
    """Expand a 2x3x3 decomposition to 2x3x9 by replicating 3 times."""
    U_small = np.array(small_result["U"])
    V_small = np.array(small_result["V"])
    W_small = np.array(small_result["W"])
    
    r_small = U_small.shape[1]
    r_large = r_small * 3
    
    n, m, p_large = 2, 3, 9
    p_small = 3
    
    U_large = np.zeros((n * m, r_large))
    V_large = np.zeros((m * p_large, r_large))
    W_large = np.zeros((p_large * n, r_large))
    
    for t in range(3):
        offset = t * r_small
        # U is the same for all blocks
        U_large[:, offset:offset+r_small] = U_small
        
        # V: rows 0..8. Block t corresponds to rows 3t..3t+2
        # V_small has rows 0..8? No, V_small is 3x3. Rows 0..2.
        # V_small[j, k] maps to V_large[j, 3t+k]
        for j in range(m):
            for k in range(p_small):
                V_large[j * p_large + (3*t + k), offset:offset+r_small] = V_small[j, k]
                
        # W: rows 0..8. Block t corresponds to rows 3t..3t+2
        # W_small[k, i] maps to W_large[3t+k, i]
        for k in range(p_small):
            for i in range(n):
                W_large[(3*t + k) * n + i, offset:offset+r_small] = W_small[k, i]
                
    return {
        "U": U_large.tolist(),
        "V": V_large.tolist(),
        "W": W_large.tolist()
    }


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
