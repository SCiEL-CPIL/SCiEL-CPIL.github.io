"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Start with the schoolbook (trivial) decomposition as fallback
    rank_school = n * m * p  # 36
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify_exact(U, V, rank):
        """Check if the decomposition is valid using exact arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r]
                    if s != 1:
                        return False
        return True
    
    def verify_float(U, V, rank, tol=1e-9):
        """Check if the decomposition is valid using floating point."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j], V_arr[j * p + k])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def make_W(rank):
        """Create a valid W matrix for the given rank."""
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                if col < rank:
                    W[k * n + i][col] = 1
                    col += 1
        return W
    
    # Strategy: Use structured decompositions based on Strassen-like ideas
    # For 3x3x4, we can try various algebraic constructions
    
    # First, try a systematic search with structured initializations
    np.random.seed(42)
    
    # Try different structured approaches
    # Approach 1: Block-based decomposition
    # Split the 3x3 matrix into blocks and use Strassen-like formulas
    
    # For 3x3x4, one approach is to use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try to find a decomposition with rank < 36 using a more structured approach
    # We'll use a combination of random and structured initializations
    
    # Try ranks from 35 down to 15
    for target_rank in range(best_rank - 1, 14, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple structured and random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            strategy = attempt % 5
            
            if strategy == 0:
                # Random Gaussian
                U_rand = np.random.randn(n * m, target_rank)
                V_rand = np.random.randn(m * p, target_rank)
            elif strategy == 1:
                # Sparse random
                U_rand = np.random.randint(0, 2, (n * m, target_rank)).astype(float)
                V_rand = np.random.randint(0, 2, (m * p, target_rank)).astype(float)
            elif strategy == 2:
                # Block-structured: group by i
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                cols_per_i = target_rank // n
                for i in range(n):
                    start = i * cols_per_i
                    end = start + cols_per_i
                    if end > target_rank:
                        end = target_rank
                    U_rand[i * m:(i + 1) * m, start:end] = np.random.randn(m, end - start)
                    V_rand[:, start:end] = np.random.randn(m * p, end - start)
            elif strategy == 3:
                # Block-structured: group by j
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                cols_per_j = target_rank // m
                for j in range(m):
                    start = j * cols_per_j
                    end = start + cols_per_j
                    if end > target_rank:
                        end = target_rank
                    U_rand[:, start:end] = np.random.randn(n * m, end - start)
                    V_rand[j * p:(j + 1) * p, start:end] = np.random.randn(p, end - start)
            else:
                # Mixed: some structured, some random
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                half = target_rank // 2
                U_rand[:, :half] = np.random.randn(n * m, half)
                V_rand[:, :half] = np.random.randn(m * p, half)
                U_rand[:, half:] = np.random.randint(0, 2, (n * m, target_rank - half)).astype(float)
                V_rand[:, half:] = np.random.randint(0, 2, (m * p, target_rank - half)).astype(float)
            
            # Quick float check first
            if verify_float(U_rand.tolist(), V_rand.tolist(), target_rank, tol=1e-6):
                # More precise check
                if verify_float(U_rand.tolist(), V_rand.tolist(), target_rank, tol=1e-9):
                    W_rand = make_W(target_rank)
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand}
                    best_rank = target_rank
                    break
        
        if best_rank == target_rank:
            break
    
    # If no better solution found with random search, try a more systematic algebraic approach
    # For now, return the best result found
    return best_result
# EVOLVE-BLOCK-END
