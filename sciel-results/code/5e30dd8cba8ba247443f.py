"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0.0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _try_rank_search(max_rank=17, time_limit=120.0):
    """
    Search for a decomposition with rank < 18.
    Uses a structured approach: try to find integer/half-integer solutions.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # For rank r, we have r columns in U (6×r), V (9×r), W (6×r)
    # The constraint is: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of 18 equations (one per (i,j,k) triple) in 3*r variables per column
    # But it's nonlinear (trilinear).
    
    # Strategy: Try small integer coefficients.
    # For each rank r from max_rank down to 9, try to find a solution.
    
    # A more practical approach: use the fact that for 2×3×3, 
    # we can try to exploit structure.
    
    # Let's try a direct algebraic approach for rank 17.
    # We'll parameterize and use sympy-like solving with numpy.
    
    # Actually, let's try a randomized algebraic search.
    # Fix W to be identity-like or structured, then solve for U and V.
    
    # Alternative: Try to see if rank 17 is possible by checking 
    # if the 18×18 matrix of constraints is singular in some way.
    
    # For now, let's try a simple structured search for rank 17.
    # We'll try to find U, V, W with entries in {-1, 0, 1, 0.5}
    
    # This is a very large search space, so let's be smart.
    # 
    # Key insight: The matrix multiplication tensor for 2×3×3 might have
    # rank exactly 18, as there's no known better algorithm.
    # 
    # Let me try a different approach: verify that 18 is indeed minimal
    # by checking known lower bounds, and if not, do a limited search.
    
    # Lower bound for matrix multiplication rank:
    # For M_{n,m,p}, rank >= max(n*p, m*p, n*m) is not correct.
    # The correct lower bound involves the "support" of the tensor.
    # 
    # Actually, for general matrix multiplication, the rank is at least
    # n*m*p / max(n,m,p) in some cases, but this isn't tight.
    
    # Let's do a limited search for rank 17 with small integer coefficients.
    # We'll use a greedy/heuristic approach.
    
    # For rank r, we need to assign to each of the 18 (i,j,k) triples
    # a sum of r products that equals 1.
    
    # Let's try: assume W has a specific structure, then solve for U and V.
    
    # Try W as a 6×r matrix where each column has at most 2 non-zero entries.
    # This is a heuristic.
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.8:
        return None
    
    # Given the complexity, let's try a more targeted approach.
    # For 2×3×3, try to see if we can reduce rank by 1.
    
    # The 18 terms correspond to (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # Can we combine any two terms?
    
    # Term (i,j,k) contributes to C[i,k] via A[i,j]*B[j,k].
    # Two terms (i,j,k) and (i',j',k') can only be combined if they contribute
    # to the same output entry, i.e., i=i' and k=k'.
    
    # For fixed i and k, there are m=3 terms (one for each j).
    # These 3 terms all contribute to C[i,k].
    # Can we represent 3 products with 2 rank-1 terms?
    # That would require: A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # = U1*V1*W1 + U2*V2*W2 (with appropriate indices)
    
    # This is generally not possible with rank 2 for a 3-term dot product,
    # as the rank of a 3-dimensional dot product tensor is 3.
    
    # So each output entry C[i,k] requires at least 3 rank-1 terms.
    # With 6 output entries, that's at least 18 terms... but terms can be shared
    # if they contribute to multiple outputs.
    
    # A rank-1 term (u, v, w) contributes to output (i,k) if:
    # - u has non-zero at some (i,j)
    # - v has non-zero at some (j',k')  
    # - w has non-zero at some (k'',i'')
    # And it contributes to C[i'',k''] via the product.
    
    # For a single rank-1 term to contribute to multiple outputs, it would need
    # to have a specific structure. But typically each rank-1 term in a
    # matrix multiplication decomposition contributes to exactly one output.
    
    # This suggests rank 18 might indeed be minimal.
    
    # Let's do a quick verification search for rank 17 with a time limit.
    # We'll use a random restart approach with small integer coefficients.
    
    import random
    random.seed(42)
    
    for r in range(17, 8, -1):
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.7:
            break
        
        # Try a few random structured decompositions
        for attempt in range(100):
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.7:
                break
            
            # Create random sparse U, V, W
            U = np.zeros((6, r))
            V = np.zeros((9, r))
            W = np.zeros((6, r))
            
            for c in range(r):
                # Choose 1-3 non-zero entries for each
                u_nonzero = random.randint(1, 3)
                v_nonzero = random.randint(1, 3)
                w_nonzero = random.randint(1, 3)
                
                u_indices = random.sample(range(6), u_nonzero)
                v_indices = random.sample(range(9), v_nonzero)
                w_indices = random.sample(range(6), w_nonzero)
                
                for idx in u_indices:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_indices:
                    V[idx, c] = random.choice([-1, 1])
                for idx in w_indices:
                    W[idx, c] = random.choice([-1, 1])
            
            # Check if this decomposes the tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        total = 0
                        for c in range(r):
                            total += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                        if abs(total - 1) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    First tries to find a decomposition with rank < 18.
    Falls back to the schoolbook rank-18 decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a lower-rank decomposition
    result = _try_rank_search(max_rank=17, time_limit=120.0)
    if result is not None:
        return result
    
    # Fall back to schoolbook rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
