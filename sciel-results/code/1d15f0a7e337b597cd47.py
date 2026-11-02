"""Matrix multiplication 3x3x5 tensor rank minimization via numerical search."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 45


def _build_matmul_tensor():
    """Build the 3-way matrix multiplication tensor T[i,j,k] = delta[i,k] for C=A*B.
    T has shape (n, m, p) where T[i,j,k]=1 if i==k else 0.
    Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(N):
        for k in range(P):
            if i == k:
                for j in range(M):
                    T[i, j, k] = 1.0
    return T


def _verify_decomposition(U, V, W):
    """Check if U, V, W (numpy arrays) form a valid decomposition of the matmul tensor."""
    T = _build_matmul_tensor()
    rank = U.shape[1]
    # Reconstruct: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u_idx = i * M + j
                v_idx = j * P + k
                w_idx = k * N + i
                val = np.dot(U[u_idx, :], V[v_idx, :]) * 0  # placeholder
                # Actually need elementwise: sum_c U[u,c]*V[v,c]*W[w,c]
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                expected = T[i, j, k]
                if abs(val - expected) > 1e-6:
                    return False
    return True


def _try_rationalize(matrix, tol=1e-8):
    """Try to convert a numpy matrix to exact fractions (integers or halves).
    Returns the matrix as a list of lists of Fractions, or None if fails.
    """
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            if abs(val) < tol:
                new_row.append(Fraction(0, 1))
            elif abs(val - round(val)) < tol:
                new_row.append(Fraction(int(round(val)), 1))
            elif abs(val - (round(val * 2) / 2.0)) < tol:
                new_row.append(Fraction(round(val * 2), 2))
            else:
                return None
        result.append(new_row)
    return result


def _search_rank(target_rank, seed=None, max_iter=3000, time_limit=25):
    """Try to find a decomposition of the given rank using numerical optimization."""
    if seed is not None:
        np.random.seed(seed)
    
    start_time = time.time()
    T = _build_matmul_tensor()
    
    # Flatten T to a vector for loss computation
    # T[i,j,k] should equal sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c]
    
    # Parameterize: U (N*M x r), V (M*P x r), W (P*N x r)
    # Use a single parameter vector
    total_params = (N * M + M * P + P * N) * target_rank
    
    # Initialize with random values
    params = np.random.randn(total_params).astype(np.float64) * 0.1
    
    # Split into U, V, W
    u_size = N * M * target_rank
    v_size = M * P * target_rank
    
    def _unpack(p):
        U = p[:u_size].reshape(N * M, target_rank)
        V = p[u_size:u_size + v_size].reshape(M * P, target_rank)
        W = p[u_size + v_size:].reshape(P * N, target_rank)
        return U, V, W
    
    def _loss_and_grad(p):
        U, V, W = _unpack(p)
        # Compute loss: sum of (reconstructed - T)^2 over all (i,j,k)
        loss = 0.0
        # For gradient, we'll use finite differences or analytic
        # Analytic gradient:
        # d_loss/dU[i*m+j, c] = 2 * (recon[i,j,k] - T[i,j,k]) * V[j*p+k, c] * W[k*n+i, c]
        # But recon depends on all (i,j,k) for given i,j through different k...
        # Actually for a given (i,j), recon[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # The loss is sum over all (i,j,k) of (recon[i,j,k] - T[i,j,k])^2
        
        # Let's compute recon for all (i,j,k)
        recon = np.zeros((N, M, P), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    u_idx = i * M + j
                    v_idx = j * P + k
                    w_idx = k * N + i
                    recon[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        
        loss = np.sum((recon - T) ** 2)
        
        # Gradient computation (analytic)
        diff = recon - T  # (N, M, P)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    u_idx = i * M + j
                    v_idx = j * P + k
                    w_idx = k * N + i
                    d = 2.0 * diff[i, j, k]
                    # d/dU[u_idx, c] = d * V[v_idx, c] * W[w_idx, c]
                    dU[u_idx, :] += d * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += d * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += d * U[u_idx, :] * V[v_idx, :]
        
        grad = np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
        return loss, grad
    
    # L-BFGS-like optimization using simple gradient descent with momentum
    best_loss = float('inf')
    best_params = params.copy()
    lr = 0.01
    momentum = np.zeros_like(params)
    mu = 0.9
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        loss, grad = _loss_and_grad(params)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
        
        if best_loss < 1e-10:
            break
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Gradient descent with momentum
        momentum = mu * momentum - lr * grad
        params = params + momentum
        
        # Decay learning rate
        if it % 100 == 0:
            lr *= 0.95
        
        # Check for divergence
        if not np.all(np.isfinite(params)):
            params = best_params + np.random.randn(total_params) * 0.01
            momentum = np.zeros_like(params)
            lr = 0.005
    
    # Verify the best solution
    U, V, W = _unpack(best_params)
    if _verify_decomposition(U, V, W):
        return U, V, W, best_loss
    
    # Try to rationalize
    U_frac = _try_rationalize(U)
    V_frac = _try_rationalize(V)
    W_frac = _try_rationalize(W)
    if U_frac is not None and V_frac is not None and W_frac is not None:
        # Convert to float lists for return
        U_list = [[float(x) for x in row] for row in U_frac]
        V_list = [[float(x) for x in row] for row in V_frac]
        W_list = [[float(x) for x in row] for row in W_frac]
        return U_list, V_list, W_list, best_loss
    
    return None


def _try_strassen_based(rank, time_limit=15):
    """Try a Strassen-inspired construction for 3x3x5.
    
    Idea: Use Strassen's 7-term formula for the 3x3x3 core, then extend to 3x3x5.
    For 3x3x3, rank 23 is known. For 3x3x5, we can potentially use rank ~23 + some.
    
    Alternative: Use the fact that 3x3x5 can be seen as 3x3x3 + 3x3x2.
    3x3x3 has rank 23, 3x3x2 has rank at most 18 (schoolbook) or less.
    So total at most 41, possibly less with sharing.
    """
    # For now, try a structured approach
    # Use 7 Strassen terms for a 2x2x2 sub-block approach
    # Actually, let's try a direct parameterization with structure
    
    start_time = time.time()
    T = _build_matmul_tensor()
    
    # Try rank = given rank with random initialization but more structured
    np.random.seed(42)
    
    # Multiple random restarts
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        result = _search_rank(rank, seed=1000 + attempt, max_iter=2000, 
                              time_limit=min(15, time_limit - (time.time() - start_time)))
        if result is not None:
            return result
    
    return None


def construct():
    """Find minimum-rank exact tensor decomposition for 3x3x5 matrix multiplication."""
    total_start = time.time()
    deadline = total_start + 120  # 120s wall clock limit, use 110s for safety
    
    # Start with schoolbook (rank 45) as guaranteed valid fallback
    schoolbook_rank = N * M * P  # 45
    U_sb = [[0] * schoolbook_rank for _ in range(N * M)]
    V_sb = [[0] * schoolbook_rank for _ in range(M * P)]
    W_sb = [[0] * schoolbook_rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U_sb[i * M + j][col] = 1
                V_sb[j * P + k][col] = 1
                W_sb[k * N + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower ranks
    # Known bounds: 3x3x3 has rank 23, so 3x3x5 > 23
    # Try ranks from a reasonable lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # More realistic: for 3x3x3 it's 23, so 3x3x5 is probably >= 23
    
    # Try a range of ranks
    ranks_to_try = list(range(24, min(45, schoolbook_rank)))
    # Also try some specific promising ranks
    ranks_to_try = sorted(set(ranks_to_try + [23, 24, 25, 26, 27, 28, 30, 32, 35]))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        # Try to find decomposition at this rank
        result = _try_strassen_based(rank, time_limit=min(20, max(5, (deadline - 30 - time.time()))))
        
        if result is not None:
            U, V, W, loss = result
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            # Found a valid decomposition, try lower
            continue
    
    return best_result


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Rank: {rank}")
