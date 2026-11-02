import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _validate_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # Let's check a sample of entries or all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with schoolbook: rank 125
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # For 5x5 matrix multiplication, we can try to find a lower rank decomposition.
    # The known lower bounds and best constructions:
    # - Schoolbook: 125
    # - Strassen-like: For 2x2 it's 7 vs 8. For larger matrices, various algorithms exist.
    # - The best known for 5x5x5 is not trivially better than schoolbook in simple constructions.
    
    # Let's try a systematic search using the structure of matrix multiplication.
    # We can parameterize U, V, W with small integer/half-integer entries and check validity.
    
    # Strategy: Try random sparse decompositions with small entries
    # and use iterative refinement.
    
    # First, let's try to see if we can find a decomposition with rank < 125
    # using a greedy or random approach.
    
    # For each rank r from some lower bound up to 125, try to find a valid decomposition.
    # This is computationally expensive, so we'll use random sampling.
    
    # The tensor for 5x5x5 matrix multiplication has 125 non-zero entries (all ones).
    # A decomposition of rank r means:
    # T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try random initializations and gradient-free optimization.
    
    np.random.seed(42)
    
    for target_rank in range(100, 126):  # Try ranks from 100 to 125
        if time.time() > deadline - 5.0:
            break
            
        # Try multiple random initializations
        max_attempts = 20 if target_rank < 120 else 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Random sparse initialization with small values
            U_rand = np.random.choice([-1.0, 0.0, 0.5, 1.0], size=(n*m, target_rank))
            V_rand = np.random.choice([-1.0, 0.0, 0.5, 1.0], size=(m*p, target_rank))
            W_rand = np.random.choice([-1.0, 0.0, 0.5, 1.0], size=(p*n, target_rank))
            
            # Check if this decomposition is valid
            if _validate_decomposition(U_rand, V_rand, W_rand, n, m, p):
                best_rank = target_rank
                best_U = U_rand
                best_V = V_rand
                best_W = W_rand
                found = True
                break
        
        if found:
            break  # Found a valid decomposition at this rank, no need to try lower
    
    # Convert to lists for return
    def to_list(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    # Round to nearest half-integer for cleaner output
    def round_half(arr):
        return np.round(arr * 2) / 2
    
    U_out = to_list(round_half(best_U))
    V_out = to_list(round_half(best_V))
    W_out = to_list(round_half(best_W))
    
    return {"U": U_out, "V": V_out, "W": W_out}
