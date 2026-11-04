"""
Strategy: Structural Separation via Support Pattern Search.
Instead of random numerical search, we search for sparse support patterns (masks)
for U, V, W that allow for a low-rank decomposition. 
We model the problem as finding sets of indices for each rank term such that
the tensor equation is satisfied.
For small ranks, we can iterate over possible sparsity patterns or use a
heuristic to find "compatible" supports.
Given the small dimensions (3x5x5), we can also try to construct explicit
decompositions using known algebraic identities or simple block structures.

This program focuses on:
1. Fast validation using numpy.
2. A structured search for support patterns that are "compatible" (i.e., for each
   term r, the supports of U[:, :, r], V[:, :, r], W[:, :, r] must align such that
   their product contributes to the correct (i,j,k) entries).
3. Solving the resulting linear system for the non-zero entries.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS

_START_TIME = time.time()
DEADLINE = 140.0  # Safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _validate_numpy(U, V, W, n, m, p):
    """
    Fast validation using numpy. U, V, W are lists of lists or numpy arrays.
    Returns True if the decomposition is exact.
    """
    if not U or not V or not W:
        return False
        
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    if U_np.ndim != 2 or V_np.ndim != 2 or W_np.ndim != 2:
        return False
        
    rank = U_np.shape[1]
    if rank == 0:
        return False
        
    if U_np.shape[0] != n * m:
        return False
    if V_np.shape[0] != m * p:
        return False
    if W_np.shape[0] != p * n:
        return False
    if U_np.shape[1] != V_np.shape[1] or V_np.shape[1] != W_np.shape[1]:
        return False

    # Reshape for batch computation
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # We need to check if sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # This is a 3-way contraction.
    
    # Efficient way:
    # For each r, compute the outer product-like structure.
    # Let's use einsum to sum over r.
    # result[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    
    # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    # This might be slow if rank is large, but for rank < 75 it's fine.
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    except Exception:
        return False
        
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
        
    return True

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def _try_strassen_like_block(n, m, p):
    """
    Attempt to construct a decomposition by treating the 3x5x5 as blocks.
    Since 5 is not a power of 2, we can't use standard Strassen directly on the whole thing.
    However, we can split the 5x5 matrix into 2x2, 2x2, 2x1, 1x2, 1x1 blocks?
    This is getting complicated. 
    Instead, let's look for a specific low-rank construction for 1x5x5.
    
    The tensor T_{1,5,5} corresponds to the map (a, B) -> a B.
    Rank of T_{1,5,5} is 5*5 = 25? 
    Actually, the rank of the matrix multiplication tensor T_{n,m,p} is the bilinear complexity.
    For T_{1,m,p}, the bilinear complexity is m*p.
    So the "tensor product" 3x1x1 (rank 1) and 1x5x5 (rank 25) gives rank 1*25 = 25?
    No, the tensor product of decompositions multiplies ranks.
    If T_{3,5,5} = T_{3,1,1} \otimes T_{1,5,5}, then rank(T_{3,5,5}) <= rank(T_{3,1,1}) * rank(T_{1,5,5}).
    rank(T_{3,1,1}) = 3 (since it's just 3 scalar multiplications, or is it? 
    T_{3,1,1} multiplies a 3x1 matrix by a 1x1 matrix. 
    (A B)_{i,1} = sum_j A_{i,j} B_{j,1} = A_{i,1} B_{1,1}.
    This is 3 independent scalar multiplications. So rank is 3.
    So rank(T_{3,5,5}) <= 3 * 25 = 75. This is just the schoolbook bound.
    
    Is there a better way?
    The rank of T_{3,5,5} is known to be 75? 
    Let's check literature or bounds.
    Lower bound for T_{3,5,5}: 
    Max(n^3, m^3, p^3) is not a bound.
    Bound: max(n*m*p, ...)?
    Actually, for rectangular matrices, the rank is often n*m*p unless there's a specific structure.
    
    However, we can try to find a decomposition with rank < 75 using random linear algebra.
    The previous program did random search. 
    Let's try a more directed search: 
    1. Generate random U, V.
    2. Solve for W.
    3. Check if W is "simple" (e.g., sparse or integer).
    
    This is a linear system for each k,i.
    For fixed k,i, we have:
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all j.
    This is a system of m equations for the variable vector w_{k,i} = [W[k,i,0], ..., W[k,i,rank-1]].
    Let M_{k,i} be the m x rank matrix where M_{k,i}[j, r] = U[i,j,r] V[j,k,r].
    Then M_{k,i} w_{k,i} = 1_m.
    
    If rank > m, we have many solutions. We can pick a sparse one or a minimum norm one.
    But we need the SAME W to work for all k,i? No, W[k,i,r] is specific to k,i.
    So for each (k,i), we can solve for W[k,i,:] independently!
    
    Wait, this is the key insight.
    The tensor equation is:
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    For a fixed (i,k), let's look at the equation for all j:
    sum_j? No, for each j, the sum is over r.
    sum_r (U[i,j,r] V[j,k,r]) W[k,i,r] = 1.
    
    Let A_{i,k} be a vector of length m? No.
    For fixed i,k, define vector v_{i,k} in R^rank where v_{i,k}[r] = ... wait.
    The term is U[i,j,r] * V[j,k,r]. This depends on j.
    So for fixed i,k, we have m equations (one for each j) and rank unknowns (W[k,i,0]...W[k,i,rank-1]).
    
    Let M_{i,k} be an m x rank matrix.
    M_{i,k}[j, r] = U[i,j,r] * V[j,k,r].
    
    We need M_{i,k} * w_{i,k} = 1_m (vector of ones).
    
    So, if we pick random U and V, we can solve for W for each (i,k) independently!
    We just need M_{i,k} to have full column rank? No, we need the system to be solvable.
    Since m=5 and rank can be anything, if rank >= 5, we can usually solve it.
    If we want rank < 75, we can pick rank = 50, say.
    Then for each of the 3*5=15 pairs (i,k), we solve a 5x50 system.
    We can choose a solution with nice properties (e.g., minimum L2 norm, or sparse).
    
    BUT, does this guarantee a valid decomposition?
    Yes, if we construct U, V, and then solve for W exactly, the decomposition holds by construction.
    The only constraint is that W must have "nice" entries (integers/half-integers) or just be valid floats?
    The task says "integer/half-integer entries". 
    If we solve with floats, we might get irrational numbers.
    However, if we pick U and V to be integers, and M_{i,k} is integer, the solution W might be rational.
    If the denominator is a power of 2, it's a half-integer? No, half-integer means k/2.
    Rational numbers with denominator 2 are half-integers.
    Denominator 1 is integer.
    
    So, strategy:
    1. Pick a target rank R < 75.
    2. Generate random integer U and V.
    3. For each (i,k), solve M_{i,k} w = 1.
    4. Check if w has entries that are integers or half-integers (denominator 1 or 2).
    5. If yes, we have a valid decomposition!
    
    This is a huge structural change: instead of searching for U,V,W simultaneously, we fix U,V and solve for W.
    """
    import random
    import numpy.linalg as la
    from fractions import Fraction
    
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    rng = random.Random(123)
    
    # Try ranks from 74 down to 60
    for R in range(74, 59, -1):
        if _time_left() < 10:
            break
            
        found = False
        for trial in range(50):
            if _time_left() < 5:
                break
                
            # Generate random U and V with small integers
            # Density should be high enough to ensure solvability
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            
            # Fill with random -1, 0, 1
            for i in range(n * m):
                for r in range(R):
                    if rng.random() < 0.5:
                        U[i, r] = rng.choice([-1, 1])
            for j in range(m * p):
                for r in range(R):
                    if rng.random() < 0.5:
                        V[j, r] = rng.choice([-1, 1])
                        
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = [[0] * R for _ in range(p * n)]
            
            valid = True
            for i in range(n):
                for k in range(p):
                    # Construct M_{i,k}
                    # M[j, r] = U[i, j, r] * V[j, k, r]
                    # U index: i*m + j
                    # V index: j*p + k
                    M = np.zeros((m, R), dtype=int)
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        M[j] = u_row * v_row
                        
                    # Solve M w = 1
                    # Use least squares or exact solve if possible
                    # Since M is integer, we can use Gaussian elimination with fractions
                    # or just check if a simple solution exists.
                    
                    # For speed, try to find a sparse solution or just use np.linalg.lstsq
                    # and check if the result is close to half-integers.
                    
                    # Actually, we can use the fact that we can add null space vectors.
                    # But let's start with the minimum norm solution.
                    try:
                        w, residuals, rank, s = la.lstsq(M, np.ones(m), rcond=None)
                    except Exception:
                        valid = False
                        break
                    
                    # Check if w entries are close to half-integers
                    w_half_int = np.round(w * 2) / 2
                    if not np.allclose(w, w_half_int, atol=1e-9):
                        valid = False
                        break
                        
                    # Store W
                    for r in range(R):
                        W_list[k * n + i][r] = w_half_int[r]
                        
            if valid:
                # Final validation
                if _validate_numpy(U_list, V_list, W_list, n, m, p):
                    best_rank = R
                    best_decomp = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break # Try lower rank
        
        if found:
            # We found a valid decomposition at this rank. 
            # We can continue to try lower ranks, but we have a good candidate.
            # Given time constraints, we might break here if rank is low enough.
            if R < 70:
                break
                
    return best_decomp

def construct():
    """
    Main entry point.
    1. Start with schoolbook (rank 75).
    2. Try the structured "Fix U,V, Solve W" search.
    3. Return best found.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    sb = _schoolbook(n, m, p)
    best = sb
    best_rank = n * m * p
    
    # Validate schoolbook just in case
    if not _validate_numpy(sb["U"], sb["V"], sb["W"], n, m, p):
        # Should not happen, but if it does, we have a problem.
        return sb
        
    # Try the new structured search
    if _time_left() > 20:
        result = _try_strassen_like_block(n, m, p)
        if result is not None:
            r = len(result["U"][0]) if result["U"] else 0
            if r < best_rank:
                # Validate again to be safe
                if _validate_numpy(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
                    
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Rank achieved: {len(result['U'][0]) if result['U'] else 'unknown'}")
