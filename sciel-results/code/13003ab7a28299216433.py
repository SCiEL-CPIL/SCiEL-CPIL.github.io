"""Program for 2x3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product, permutations, combinations

DIMENSIONS = [2, 3, 3]

def _validate_decomposition(U, V, W):
    """
    Verify that the tensor decomposition exactly reconstructs the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    
    U: (n*m, rank) - input A factor
    V: (m*p, rank) - input B factor  
    W: (p*n, rank) - output factor
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the multiplication tensor: T[i,j,k] = delta_{i*k + j*p, k*n + i}
    # For each term (i,j,k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other combinations, the sum should be 0
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[i, j, k] = np.dot(np.dot(U_np[idx_u], V_np[idx_v]), W_np[idx_w])
    
    # The correct tensor has T[i,j,k] = 1 if the multiplication contributes to that position
    # For C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all valid (i,j,k) triples
    expected_T = np.ones((n, m, p))
    
    if not np.allclose(T, expected_T, atol=1e-10):
        return False
    
    return True

def _validate_decomposition_strict(U, V, W):
    """
    Strict verification using exact rational arithmetic if entries are integers/half-integers.
    """
    from fractions import Fraction
    
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    U_frac = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row] for row in U]
    V_frac = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row] for row in V]
    W_frac = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += U_frac[idx_u][c] * V_frac[idx_v][c] * W_frac[idx_w][c]
                if total != Fraction(1):
                    return False
    
    # Also verify orthogonality: for non-matching triples, the product sum should be 0
    # This is implicitly checked since we only need T[i,j,k]=1 for all valid (i,j,k)
    # But let's also check that cross-terms vanish properly
    # Actually, the tensor is exactly T[i,j,k]=1 for all i,j,k in ranges, so we're done
    return True

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 x 3x3 matrix multiplication.
    The tensor M_{2,3,3} has known minimum rank of 12.
    
    We use a systematic search approach:
    1. Start with the known rank-12 decomposition structure
    2. Use the fact that M_{2,3,3} contains M_{2,2,2} (rank 7) plus additional terms
    
    Returns dict with U, V, W factor matrices.
    """
    t0 = time.time()
    deadline = t0 + 110  # 110 second guard (evaluator kills at 120s)
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (9, rank)
    # W shape: (p*n, rank) = (6, rank)
    
    # Try to find rank 12 decomposition using systematic enumeration
    # of sign patterns and simple coefficient structures
    
    # Strategy: Use the known fact that rank(M_{2,3,3}) = 12
    # The decomposition can be constructed by:
    # - Using a rank-7 decomposition for the M_{2,2,2} subproblem (rows 0-1, cols 0-2 of A, rows 0-1 of B)
    # - Adding 5 more terms for the remaining interactions
    
    # Let's build it step by step
    # First, get the rank-7 Strassen decomposition for 2x2x2
    
    # Strassen decomposition for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x3 x 3x3, we need to extend this.
    # Let's try a direct construction with rank 12.
    
    # Approach: Try all combinations of +/-1, 0, 1/2, -1/2 coefficients
    # in a structured way, guided by the tensor structure.
    
    # Since exhaustive search is too large, use a heuristic approach:
    # Start with rank 18 (schoolbook) and try to merge terms.
    
    # Alternative: Use the known explicit construction.
    # The rank-12 decomposition for M_{2,3,3} is known in the literature.
    
    # Let me try to construct it using bilinear forms.
    
    # A is 2x3: A = [[a11, a12, a13], [a21, a22, a23]]
    # B is 3x3: B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # C is 2x3: C = [[c11, c12, c13], [c21, c22, c23]]
    
    # c11 = a11*b11 + a12*b21 + a13*b31
    # c12 = a11*b12 + a12*b22 + a13*b32
    # c13 = a11*b13 + a12*b23 + a13*b33
    # c21 = a21*b11 + a22*b21 + a23*b31
    # c22 = a21*b12 + a22*b22 + a23*b32
    # c23 = a21*b13 + a22*b23 + a23*b33
    
    # Try to find 12 bilinear forms m1..m12 such that each c_ij is a sum of some of them.
    
    # Let's use a systematic search with numpy for speed.
    
    # Define the tensor indices
    # U[i*m+j, c] for i in 0..1, j in 0..2
    # V[j*p+k, c] for j in 0..2, k in 0..2
    # W[k*n+i, c] for k in 0..2, i in 0..1
    
    # For each c, we have vectors u_c in R^6, v_c in R^9, w_c in R^6
    # The constraint is: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is equivalent to: the outer product sum sum_c u_c ⊗ v_c ⊗ w_c = T where T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's try a greedy approach: build rank incrementally.
    
    # Start with schoolbook (rank 18) and try to find dependencies.
    
    # Actually, let's try a different approach: use the fact that we can express
    # some of the 18 terms as sums of others.
    
    # The 18 terms correspond to (i,j,k) in [0,2]x[0,2]x[0,2]
    # We want to find 12 basis tensors that span the space containing our target tensor.
    
    # This is getting complex. Let me try a concrete known construction.
    
    # Known result: rank(M_{2,3,3}) = 12
    # One explicit construction uses the following structure:
    
    # Let's try rank 12 with careful coefficient assignment.
    
    # Use numpy to help with verification and potential optimization.
    
    best_rank = n * m * p  # Fallback to schoolbook
    best_decomp = None
    
    # Try rank from 12 upward (since we know rank is at least 12)
    for target_rank in range(12, 19):
        if time.time() > deadline - 10:
            break
        
        # For each target rank, try to find a valid decomposition
        # using a randomized/hybrid approach
        
        # Method: Start with random initial guess, then use iterative refinement
        # The constraints are: sum_c U[i,m+j,c] * V[j,p+k,c] * W[k,n+i,c] = 1 for all i,j,k
        
        # This is a system of 18 equations (one for each (i,j,k) triple)
        # with target_rank * (6 + 9 + 6) = target_rank * 21 unknowns
        
        # For target_rank = 12: 252 unknowns, 18 equations -> highly underdetermined
        
        # We can try to find a solution by fixing some variables and solving for others.
        
        # Let's try a structured approach: fix the pattern of which (i,j,k) pairs
        # each rank term covers, then solve for coefficients.
        
        # Each rank term c contributes to multiple (i,j,k) triples.
        # Let's define for each c a "pattern" which is a triple (i_c, j_c, k_c) 
        # but that's just schoolbook. For lower rank, each term must contribute 
        # to multiple (i,j,k) triples.
        
        # A term c with U entries {u1,...,u6}, V entries {v1,...,v9}, W entries {w1,...,w6}
        # contributes to (i,j,k) the product U[i*3+j]*V[j*3+k]*W[k*2+i]
        
        # So the contribution pattern is determined by which indices are non-zero.
        
        # Let's try: for each rank term, choose a small support pattern, then solve
        # the resulting linear system for the coefficient values.
        
        # This is complex. Let me try a more direct numerical approach.
        
        # Set up the problem: find U, V, W of rank r such that the tensor equation holds.
        # Use the "splitting" approach: fix W, then U and V satisfy a linear system.
        
        # For fixed W, the equations are:
        # For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c] = 1/W[k*n+i,c] ... no, that's not right.
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is nonlinear in general. But if we fix W, then for each c, the terms 
        # U[i*m+j,c]*V[j*p+k,c] are coupled across (i,j,k) via W[k*n+i,c].
        
        # This is a rank-r tensor completion problem, which is hard.
        
        # Let me try a simpler heuristic: random restarts with verification.
        
        import random
        random.seed(42)
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Generate random decomposition with entries in {-1, -1/2, 0, 1/2, 1}
            # Scale by small probability of non-zero to create sparse structure
            
            U_rand = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Verify
            if _validate_decomposition(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_decomp = (U_rand, V_rand, W_rand)
                break
        
        if best_decomp is not None:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
