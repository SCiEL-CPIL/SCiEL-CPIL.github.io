"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72
    
    # Fast vectorized verification
    def verify_np(U, V, W, rank):
        """Fast vectorized verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Key insight: For matrix multiplication tensor T[i,j,k]=1,
    # we can try structured decompositions that exploit the fact
    # that all tensor entries are 1.
    
    # Approach 1: Try to find a lower-rank decomposition using the
    # structure that T is all-ones. We can use the fact that:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # One approach: use a decomposition where each column c contributes
    # to a subset of (i,j,k) entries, with the contributions summing to 1.
    
    # Method: Try rank values from 71 down, using random search with
    # half-integer entries.
    
    # First, try a few structured approaches
    
    # Structured approach: For 3x3x8, we can think of it as 8 copies of 3x3.
    # If we can find a rank-r decomposition for 3x3x3, we might be able to
    # combine them.
    
    # Actually, let's try a more direct approach: random search with 
    # small half-integer values.
    
    # For efficiency, we'll use numpy for the verification and try
    # multiple random initializations.
    
    for target_rank in range(best_rank - 1, max(40, best_rank - 30), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 20 if target_rank > 60 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small half-integer values
            # Use values in {-1, -0.5, 0, 0.5, 1}
            U = (np.random.randint(-2, 3, (n * m, target_rank)) * 0.5)
            V = (np.random.randint(-2, 3, (m * p, target_rank)) * 0.5)
            W = (np.random.randint(-2, 3, (p * n, target_rank)) * 0.5)
            
            if verify_np(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if time.time() > deadline:
            break
    
    # Method 2: Try a more structured approach based on the observation
    # that T[i,j,k] = 1 for all i,j,k.
    
    # We can try to construct a decomposition where each column c has
    # a specific pattern. For example, we can try to use columns that
    # are non-zero only for certain (i,j,k) combinations.
    
    # Approach: Use a "block" decomposition where we group the k dimension.
    # For 3x3x8, we can write 8 = 3+3+2, and try to decompose each block.
    
    # But first, let's try a simpler approach: use the fact that we can
    # represent 1 as a sum of products of small half-integers.
    
    # For example, 1 = 1*1*1, or 1 = 0.5*2*1, etc.
    
    # Let's try a different random search strategy: start from schoolbook
    # and try to merge columns.
    
    if time.time() < deadline - 30:
        for target_rank in range(min(best_rank - 1, 65), max(50, best_rank - 15), -1):
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue
            
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                # Start from schoolbook and try to merge columns
                U = np.zeros((n * m, best_rank), dtype=float)
                V = np.zeros((m * p, best_rank), dtype=float)
                W = np.zeros((p * n, best_rank), dtype=float)
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
                
                # Try to merge columns by adding random weights
                # and then rounding to half-integers
                for merge_attempt in range(20):
                    if time.time() > deadline:
                        break
                    
                    # Randomly select pairs of columns to merge
                    c1 = np.random.randint(0, best_rank)
                    c2 = np.random.randint(0, best_rank)
                    if c1 == c2:
                        continue
                    
                    weight = np.random.choice([-0.5, -0.25, 0.25, 0.5])
                    U[:, c1] += weight * U[:, c2]
                    V[:, c1] += weight * V[:, c2]
                    W[:, c1] += weight * W[:, c2]
                
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Remove zero columns
                U_nonzero = U[:, np.any(U != 0, axis=0)]
                V_nonzero = V[:, np.any(V != 0, axis=0)]
                W_nonzero = W[:, np.any(W != 0, axis=0)]
                
                actual_rank = U_nonzero.shape[1]
                
                if actual_rank < best_rank and actual_rank <= target_rank:
                    if verify_np(U_nonzero, V_nonzero, W_nonzero, actual_rank):
                        best_rank = actual_rank
                        best_result = {"U": U_nonzero.tolist(), "V": V_nonzero.tolist(), "W": W_nonzero.tolist()}
                        break
            
            if time.time() > deadline:
                break
    
    return best_result
# EVOLVE-BLOCK-END
