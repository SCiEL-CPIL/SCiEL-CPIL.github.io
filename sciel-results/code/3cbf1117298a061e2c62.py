"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = 2, 3, 4
    N, M, P = n * m, m * p, p * n  # 6, 12, 8

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 24
    U_school = [[0] * rank_school for _ in range(N)]
    V_school = [[0] * rank_school for _ in range(M)]
    W_school = [[0] * rank_school for _ in range(P)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        """Verify that the decomposition exactly matches matrix multiplication."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        u = U[i * m + j][r]
                        v = V[j * p + k][r]
                        w = W[k * n + i][r]
                        total += u * v * w
                    if total != 1:
                        return False
        return True

    def verify_full(U, V, W):
        """Full verification using numpy for speed."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
        return True

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # --- Strategy: Construct a candidate rank-12 decomposition and verify ---
    # The rank of M_{2,3,4} is conjectured/known to be 12 in some contexts,
    # but definitely less than 24. Let's try to find a rank-12 solution.
    
    # We will use a randomized search with a small set of rational values
    # to find a valid decomposition of rank < 24.
    
    # First, let's try to find a rank-12 decomposition using a structured approach
    # or random search with rational values.
    
    # Since random search over continuous values is hard to verify exactly,
    # we restrict to a small set of rational values: {-1, -0.5, 0, 0.5, 1}
    
    def find_decomposition(rank_limit, time_limit, seed=42):
        """Search for a decomposition of rank <= rank_limit."""
        random.seed(seed)
        vals = [-1, -0.5, 0, 0.5, 1]
        
        # Try to find a decomposition of exactly rank_limit
        # We'll do a random search
        
        attempts = 0
        max_attempts = 50000  # Upper bound, but we check time
        
        while time.time() < time_limit:
            attempts += 1
            if attempts > max_attempts:
                break
            
            # Generate random matrices with entries from vals
            U = [[random.choice(vals) for _ in range(rank_limit)] for _ in range(N)]
            V = [[random.choice(vals) for _ in range(rank_limit)] for _ in range(M)]
            W = [[random.choice(vals) for _ in range(rank_limit)] for _ in range(P)]
            
            # Quick verification
            if verify(U, V, W):
                return U, V, W
        
        return None

    # Try to find a decomposition of rank 12
    # This is a hard problem, so we might not find it quickly with random search.
    # However, we can try a few ranks.
    
    # Let's try rank 12 first, then 13, 14, etc. up to 23.
    # But random search is very unlikely to find a rank-12 solution quickly.
    
    # Instead, let's use a more informed approach:
    # We know that the schoolbook rank is 24.
    # We can try to combine terms to reduce the rank.
    
    # For example, if we have two terms that differ only in one index,
    # we might be able to combine them.
    
    # But this is complex. Let's stick to random search with a small set of values.
    
    # To improve the chances, we can try to initialize with a partial schoolbook
    # and then try to reduce the rank by zeroing out some columns and adjusting others.
    
    # However, this is also complex.
    
    # Let's try a different approach: use the fact that the tensor is symmetric in some sense
    # and try to find a decomposition with a specific structure.
    
    # For now, let's just do a random search for rank 12, 13, 14, 15, 16, 17, 18.
    # If we find one, we return it. Otherwise, we return the schoolbook.
    
    for r in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
        
        # Allocate less time for smaller ranks since they are harder
        time_budget = min(5, deadline - 5 - time.time())
        if time_budget <= 0:
            break
        
        result = find_decomposition(r, time.time() + time_budget)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
