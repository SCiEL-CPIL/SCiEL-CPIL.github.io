import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s to leave safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build the target tensor support
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0
    
    def evaluate(U, V, W, rank):
        """Check if the decomposition is exact using vectorized ops."""
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        
        # For each rank component c, the contribution to (a,b,c_idx) is U[a,c]*V[b,c]*W[c_idx,c]
        # We check all support elements
        for (a, b, c) in target.keys():
            s = float(np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :]))
            if abs(s - 1.0) > 1e-5:
                return False
        return True
    
    def try_rank(rank, max_iter=300, lr=0.02, seed=42):
        """Try to find a decomposition of given rank using gradient descent with momentum."""
        if time.time() > deadline - 15:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3
        
        # Momentum
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        beta = 0.9
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        support_keys = list(target.keys())
        
        for it in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Compute error on all support elements (vectorized)
            err = 0.0
            for (a, b, c) in support_keys:
                s = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Compute gradients using a random subset for speed
            np.random.shuffle(support_keys)
            sample_size = min(200, len(support_keys))
            sample = support_keys[:sample_size]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c) in sample:
                s = float(np.dot(U[a, :], V[b, :] * W[c, :]))
                resid = s - 1.0
                # Gradient w.r.t U[a, :]
                grad_U[a, :] += resid * V[b, :] * W[c, :]
                # Gradient w.r.t V[b, :]
                grad_V[b, :] += resid * U[a, :] * W[c, :]
                # Gradient w.r.t W[c, :]
                grad_W[c, :] += resid * U[a, :] * V[b, :]
            
            # Update with momentum
            vU = beta * vU + (1 - beta) * grad_U
            vV = beta * vV + (1 - beta) * grad_V
            vW = beta * vW + (1 - beta) * grad_W
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
        
        # Check if best is exact
        if evaluate(best_U, best_V, best_W, rank):
            return best_U, best_V, best_W
        
        # Try rounding to half-integers
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        if evaluate(U_round, V_round, W_round, rank):
            return U_round, V_round, W_round
        
        return None
    
    # Schoolbook fallback (rank 80)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    best_result = (U_school, V_school, W_school, rank_school)
    
    # Try to find better decompositions
    for rank in range(55, 80):
        if time.time() > deadline - 25:
            break
        
        # Try multiple seeds for robustness
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline - 25:
                break
            
            result = try_rank(rank, max_iter=200, lr=0.03, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, rank)
                break
        
        if best_result[3] != rank_school:
            break
    
    # Convert to lists
    U, V, W, rank = best_result
    
    if isinstance(U, np.ndarray):
        U_list = U.tolist()
    else:
        U_list = U
    if isinstance(V, np.ndarray):
        V_list = V.tolist()
    else:
        V_list = V
    if isinstance(W, np.ndarray):
        W_list = W.tolist()
    else:
        W_list = W
    
    return {"U": U_list, "V": V_list, "W": W_list}
