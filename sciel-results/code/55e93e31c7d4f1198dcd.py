import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, np_ = n * m, m * p, p * n  # 6, 21, 14
    
    def check_time():
        return time.time() > deadline
    
    def verify_decomposition(U, V, W, rank, atol=1e-10):
        """Verify that the decomposition correctly computes C = A*B."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Test with multiple random matrices
        for trial in range(20):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Bilinear algorithm:
            # X_a = sum_{i,j} U[i*m+j, a] * A[i,j]
            # Y_a = sum_{j,k} V[j*p+k, a] * B[j,k]
            # C[i,k] = sum_a W[k*n+i, a] * X_a * Y_a
            X = U.T @ A.flatten()  # (rank,)
            Y = V.T @ B.flatten()  # (rank,)
            XY = X * Y  # (rank,)
            C_decomp = (W * XY).reshape(np_, )  # wrong reshape
            # Actually: C[i,k] = sum_a W[k*n+i, a] * X_a * Y_a
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] = np.dot(W[k * n + i, :], XY)
            
            if not np.allclose(C_true, C_decomp, atol=atol):
                return False
        return True
    
    def verify_exact(U, V, W, rank):
        """Verify with exact rational arithmetic."""
        U_frac = [[Fraction(int(round(x * 10**8)) / 10**8 if abs(x * 10**8 - round(x * 10**8)) < 1e-7 else Fraction(x).limit_denominator(10**8)) for x in row] for row in U]
        V_frac = [[Fraction(int(round(x * 10**8)) / 10**8 if abs(x * 10**8 - round(x * 10**8)) < 1e-7 else Fraction(x).limit_denominator(10**8)) for x in row] for row in V]
        W_frac = [[Fraction(int(round(x * 10**8)) / 10**8 if abs(x * 10**8 - round(x * 10**8)) < 1e-7 else Fraction(x).limit_denominator(10**8)) for x in row] for row in W]
        
        # Test with small integer matrices
        for trial in range(5):
            A = [[int(np.random.randint(-3, 4)) for _ in range(m)] for _ in range(n)]
            B = [[int(np.random.randint(-3, 4)) for _ in range(p)] for _ in range(m)]
            C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            
            X = [sum(U_frac[i * m + j][a] * A[i][j] for i in range(n) for j in range(m)) for a in range(rank)]
            Y = [sum(V_frac[j * p + k][a] * B[j][k] for j in range(m) for k in range(p)) for a in range(rank)]
            XY = [X[a] * Y[a] for a in range(rank)]
            
            C_decomp = [[sum(W_frac[k * n + i][a] * XY[a] for a in range(rank)) for k in range(p)] for i in range(n)]
            
            if C_true != C_decomp:
                return False
        return True
    
    # Schoolbook fallback (rank 42)
    rank_fb = n * m * p
    U_fb = np.zeros((nm, rank_fb))
    V_fb = np.zeros((mp, rank_fb))
    W_fb = np.zeros((np_, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_fb
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()
    
    # Strategy: Try to find low-rank decompositions using structured search
    # For matrix multiplication, known ranks:
    # 2x2x2: rank 7 (Strassen)
    # 2x3x3: rank ~12
    # 2x3x7: expected rank around 20-25
    
    # Use a greedy/iterative approach to find low-rank solutions
    def find_low_rank(target_rank, max_attempts=50, time_per_attempt=2.0):
        """Try to find a decomposition with the given target rank."""
        for attempt in range(max_attempts):
            if check_time():
                return None
            
            attempt_start = time.time()
            
            # Initialize with random small integers
            U = np.random.randint(-2, 3, size=(nm, target_rank))
            V = np.random.randint(-2, 3, size=(mp, target_rank))
            W = np.random.randint(-2, 3, size=(np_, target_rank))
            
            # Iterative refinement: adjust entries to satisfy constraints
            # The constraint is that for all test matrices, the product is correct
            # Equivalent to: the tensor T[(i,j),(j',k),(i',k')] must equal
            # delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
            
            # Let's use a different approach: start from schoolbook and try to merge terms
            # Or use random search with verification
            
            # Try random rational values
            for inner_iter in range(200):
                if time.time() - attempt_start > time_per_attempt or check_time():
                    break
                
                # Random perturbation
                mask = np.random.random((nm, target_rank)) < 0.1
                U[mask] += np.random.randint(-1, 2, size=mask.sum())
                
                mask = np.random.random((mp, target_rank)) < 0.1
                V[mask] += np.random.randint(-1, 2, size=mask.sum())
                
                mask = np.random.random((np_, target_rank)) < 0.1
                W[mask] += np.random.randint(-1, 2, size=mask.sum())
                
                if verify_decomposition(U, V, W, target_rank, atol=1e-8):
                    return U, V, W
            
        return None
    
    # Try ranks from low to high
    # Known lower bound for 2x3x7 is at least 14 (np_)
    # Try from 14 up to 41
    found = False
    for r in range(14, best_rank):
        if check_time():
            break
        
        result = find_low_rank(r, max_attempts=20, time_per_attempt=1.5)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            found = True
            break
    
    # If we found something, try to improve further
    if found and not check_time():
        for r in range(14, best_rank):
            if check_time():
                break
            result = find_low_rank(r, max_attempts=10, time_per_attempt=1.0)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            result.append([float(x) for x in row])
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
