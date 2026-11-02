import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Known rank-26 decomposition for 3x3x3 matrix multiplication (Strassen-like)
    def make_3x3x3_rank26():
        # This is a known rank-26 decomposition for 3x3 matrix multiplication
        U = np.zeros((9, 26), dtype=int)
        V = np.zeros((9, 26), dtype=int)
        W = np.zeros((9, 26), dtype=int)
        
        # Strassen's 7-term algorithm for 2x2, extended to 3x3
        # We use a known 26-term decomposition
        # This is the Alon-Boppana decomposition or similar
        
        # For 3x3x3, rank 23 is known, but let's use a simpler approach
        # Actually, let me just use the schoolbook for 3x3x3 which is rank 27
        # and see if we can do better for 3x3x4
        
        # Schoolbook 3x3x3: rank 27
        rank = 27
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((9, rank), dtype=int)
        W = np.zeros((9, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a decomposition for 3x3x4
    # The trivial bound is 36 (schoolbook)
    # We know 3x3x3 has rank at most 23 (or 26 with simpler algorithms)
    # For 3x3x4, we can potentially do better than 36
    
    # Strategy: Use known 3x3x3 decomposition and extend to 3x3x4
    # Or just do random search with {-1, 0, 1} entries
    
    best_result = None
    best_rank = 36
    
    # First, create the fallback schoolbook decomposition for 3x3x4
    rank_fb = n * m * p  # 36
    U_fb = np.zeros((n*m, rank_fb), dtype=int)
    V_fb = np.zeros((m*p, rank_fb), dtype=int)
    W_fb = np.zeros((p*n, rank_fb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    best_result = (U_fb, V_fb, W_fb, rank_fb)
    
    # Try random search for lower ranks
    # Focus on ranks 24-35, with more attempts for lower ranks
    for rank in range(24, 36):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        # Number of attempts proportional to how low the rank is
        attempts = 5000 if rank <= 28 else 2000 if rank <= 32 else 1000
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random matrices with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            # Compute the decomposed tensor
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            
            # Check if it matches the target
            if np.allclose(T_decomp, T, atol=1e-10):
                best_result = (U, V, W, rank)
                best_rank = rank
                break
        else:
            # Continue to next rank
            continue
        # If we found a decomposition at this rank, continue to try lower ranks
        # But actually, we're iterating from low to high, so if we found one at rank r,
        # we should try to find even lower ranks
        # Wait, we're iterating from 24 to 35, so lower ranks come first
        # If we found one at rank 24, we're done (that's the minimum we're trying)
        # Actually, let's continue to see if we can find rank 23, etc.
        # But we started at 24, so let's add lower ranks
        
    # Also try some structured approaches
    # Try ranks 20-23 with more sophisticated search
    for rank in range(20, 24):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        attempts = 2000
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            U = np.random.choice([-1, 0, 1], size=(n*m, rank))
            V = np.random.choice([-1, 0, 1], size=(m*p, rank))
            W = np.random.choice([-1, 0, 1], size=(p*n, rank))
            
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            
            if np.allclose(T_decomp, T, atol=1e-10):
                best_result = (U, V, W, rank)
                best_rank = rank
                break
    
    # Try half-integer entries for some ranks
    for rank in range(24, 30):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
            
        attempts = 1000
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Use {-0.5, 0, 0.5, 1} entries
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            
            if np.allclose(T_decomp, T, atol=1e-10):
                # Convert to fractions if needed, but the task says integer/half-integer
                # We can return as floats, but let's keep them as they are
                best_result = (U, V, W, rank)
                best_rank = rank
                break
    
    U_best, V_best, W_best, _ = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
