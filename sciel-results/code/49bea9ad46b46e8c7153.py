import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    # Tensor T[i,j,k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    # U: (n*m) × rank, V: (m*p) × rank, W: (p*n) × rank
    # Convention: W[k*n + i, c]
    
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Schoolbook as baseline
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # For each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Check for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # T[i,j,k] = 1 for all valid indices
                    actual = 0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build tensor from decomposition
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Expected T is all ones
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-9)
    
    # Try random search for lower rank
    random.seed(42)
    
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 30:
            break
        
        # Try several random initializations
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random sparse initialization
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with some structure: start from schoolbook and remove some terms
            # Or use random ±1 entries
            for c in range(target_rank):
                # Randomly assign non-zero entries
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.3:
                            U[i*m+j][c] = random.choice([1, -1])
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3:
                            V[j*p+k][c] = random.choice([1, -1])
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.3:
                            W[k*n+i][c] = random.choice([1, -1])
            
            if verify_fast(U, V, W, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
