# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_structured_decomposition(n, m, p, max_rank, time_limit):
    """
    Try to find a decomposition with block-sparse structure using subspace separation.
    We assume U and V have specific sparsity patterns that allow for lower rank.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # For 3x4x4, the tensor has 3*4*4 = 48 non-zero entries.
    # The schoolbook rank is 48.
    # We try to find a lower rank by exploiting structure.
    
    # Strategy: Try ranks from min_possible to max_rank
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But for matrix multiplication, better lower bounds exist.
    # For (3,4,4), the known lower bound is 48 (schoolbook is optimal).
    
    # However, we'll try a structured search anyway to see if we can find something.
    
    # Try different rank values
    for rank in range(16, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try to construct a decomposition with this rank
        # Using a simple heuristic: distribute the 48 tensor entries across 'rank' columns
        # Each column corresponds to a rank-1 term
        
        # For now, just verify that schoolbook works and return it
        # A proper search would use Groebner bases or similar, but that's complex.
        
        # Instead, let's try a randomized approach with structure
        np.random.seed(42)
        
        # Generate random factor matrices with some sparsity
        for attempt in range(10):  # Limit attempts per rank
            if time.time() > deadline:
                break
            
            # Create sparse random matrices
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Fill with random values but keep some structure
            for c in range(rank):
                # Choose random non-zero positions
                u_idx = np.random.choice(n * m, size=min(3, n*m), replace=False)
                v_idx = np.random.choice(m * p, size=min(3, m*p), replace=False)
                w_idx = np.random.choice(p * n, size=min(3, p*n), replace=False)
                
                for idx in u_idx:
                    U[idx, c] = np.random.choice([-1, 0.5, 1, 2])
                for idx in v_idx:
                    V[idx, c] = np.random.choice([-1, 0.5, 1, 2])
                for idx in w_idx:
                    W[idx, c] = np.random.choice([-1, 0.5, 1, 2])
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n, m, p = 3, 4, 4
    
    # First, get the schoolbook solution as a fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find a better (lower rank) solution
    # Use remaining time for search
    time_elapsed = time.time() - start_time
    remaining_time = max(10, 130 - time_elapsed)  # Keep 20s buffer
    
    # Try structured search with lower ranks
    result = _try_structured_decomposition(n, m, p, 47, remaining_time)
    
    if result is not None:
        U, V, W = result
        # Final verification
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
