import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=3):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if s != 1:
                    return False
    return True

def _generate_random_tensor(n, m, p, coeff_range=(-1, 1)):
    """Generate a random rank-1 tensor with small integer coefficients."""
    # Choose random supports
    # To make it effective, supports should be somewhat large but not too large
    # Let's pick random subsets of indices for U, V, W
    u_idx = [(i, j) for i in range(n) for j in range(m) if random.random() < 0.5]
    v_idx = [(j, k) for j in range(m) for k in range(p) if random.random() < 0.5]
    w_idx = [(k, i) for k in range(p) for i in range(n) if random.random() < 0.5]
    
    if not u_idx or not v_idx or not w_idx:
        return None
        
    U = np.zeros(n * m)
    V = np.zeros(m * p)
    W = np.zeros(p * n)
    
    for i, j in u_idx:
        U[i * m + j] = random.randint(coeff_range[0], coeff_range[1])
    for j, k in v_idx:
        V[j * p + k] = random.randint(coeff_range[0], coeff_range[1])
    for k, i in w_idx:
        W[k * n + i] = random.randint(coeff_range[0], coeff_range[1])
        
    return U, V, W

def _tensor_contribution(U, V, W, n=3, m=3, p=3):
    """Calculate the tensor contribution of a single rank-1 term."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            if U[i * m + j] != 0:
                for k in range(p):
                    if V[j * p + k] != 0 and W[k * n + i] != 0:
                        T[i, j, k] += U[i * m + j] * V[j * p + k] * W[k * n + i]
    return T

def _greedy_search(target, n, m, p, max_rank, time_budget):
    """
    Greedy set cover to find a decomposition.
    """
    start_time = time.time()
    target = np.array(target)
    current = np.zeros((n, m, p))
    selected = []
    
    # Pool of candidate tensors
    candidates = []
    for _ in range(500):
        t = _generate_random_tensor(n, m, p)
        if t:
            U, V, W = t
            contrib = _tensor_contribution(U, V, W, n, m, p)
            # Only keep candidates that have non-zero contribution
            if np.any(contrib != 0):
                candidates.append((contrib, U, V, W))
                
    if not candidates:
        return None

    while len(selected) < max_rank:
        if time.time() - start_time > time_budget:
            break
            
        best_contrib = None
        best_candidate = None
        best_score = -1
        
        # Calculate residual
        residual = target - current
        
        for contrib, U, V, W in candidates:
            # Score: how well does this candidate reduce the residual?
            # We want to match the residual.
            # If residual is 1, and contrib is 1, score is high.
            # If residual is 0, and contrib is 0, score is 0.
            # If residual is 1, and contrib is -1, score is low.
            
            # Simple metric: sum of products of residual and contribution
            # If they match, it's positive.
            score = np.sum(residual * contrib)
            
            # Penalize if we exceed the target?
            # For now, just maximize alignment.
            if score > best_score:
                best_score = score
                best_contrib = contrib
                best_candidate = (U, V, W)
                
        if best_candidate is None:
            break
            
        U, V, W = best_candidate
        current += best_contrib
        selected.append((U, V, W))
        
        # Check if we have solved it
        if np.all(current == target):
            break
            
    # Verify the solution
    if len(selected) > 0:
        U_arr = np.array([s[0] for s in selected])
        V_arr = np.array([s[1] for s in selected])
        W_arr = np.array([s[2] for s in selected])
        
        # Construct full matrices
        U_full = U_arr.T # Shape (rank, n*m) -> (n*m, rank)
        V_full = V_arr.T
        W_full = W_arr.T
        
        # Verify
        if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
            return [U_full.tolist(), V_full.tolist(), W_full.tolist()]
            
    return None

def _schoolbook_decomposition(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140.0
    
    target = np.ones((n, m, p))
    
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Best known is 23. Let's try 24, 25, 26.
    for rank in [26, 25, 24, 23]:
        if time.time() - start_time > timeout - 10:
            break
            
        # Run greedy search with this max rank
        # We might need multiple restarts
        for attempt in range(5):
            if time.time() - start_time > timeout - 5:
                break
            result = _greedy_search(target, n, m, p, rank, time_budget=10.0)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break # Move to next lower rank
            
    return best_result
# EVOLVE-BLOCK-END
