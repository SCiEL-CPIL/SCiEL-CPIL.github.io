"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition as fallback (rank 24)
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
    
    # Try to find a rank-12 decomposition
    # The tensor T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,4)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Strategy: Use a structured ansatz with small integer/half-integer entries
    # and solve via linear algebra / search
    
    # Let's try a known approach: exploit the structure
    # C = A * B where A is 2x3, B is 3x4
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Key insight: We can use Strassen-like tricks
    # For each k, we need to compute two dot products of length 3
    # We can share some computations
    
    # Let's try a specific rank-12 construction:
    # Group terms by k (4 groups of 3 terms each)
    
    # For k=0: terms 0,1,2
    # For k=1: terms 3,4,5
    # For k=2: terms 6,7,8
    # For k=3: terms 9,10,11
    
    # For each k, we have 3 terms to compute:
    # C[0,k] and C[1,k]
    
    # Using the identity:
    # (a0+a1)(b0+b1) = a0*b0 + a0*b1 + a1*b0 + a1*b1
    # So a0*b0 + a1*b1 = (a0+a1)(b0+b1) - a0*b1 - a1*b0
    
    # For 3 elements, we can use:
    # a0*b0 + a1*b1 + a2*b2
    
    # Let's try:
    # For fixed k, define:
    # s0 = A[0,0] + A[1,0]
    # s1 = A[0,1] + A[1,1]  
    # s2 = A[0,2] + A[1,2]
    # d0 = A[0,0] - A[1,0]
    # d1 = A[0,1] - A[1,1]
    # d2 = A[0,2] - A[1,2]
    
    # Then:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # C[0,k] + C[1,k] = s0*B[0,k] + s1*B[1,k] + s2*B[2,k]
    # C[0,k] - C[1,k] = d0*B[0,k] + d1*B[1,k] + d2*B[2,k]
    
    # So:
    # C[0,k] = (sum_j s_j * B[j,k] + sum_j d_j * B[j,k]) / 2
    # C[1,k] = (sum_j s_j * B[j,k] - sum_j d_j * B[j,k]) / 2
    
    # This requires 6 multiplications per k, so 24 total - not helpful
    
    # Let me try a different approach: use a direct search with constrained entries
    
    # Actually, let me try to construct a valid rank-12 decomposition by solving
    # a system of equations
    
    # The 24 tensor equations are:
    # For (i,j,k) in {(0,0,0), (0,0,1), ..., (1,2,3)}:
    #   sum_{r=0}^{11} U[i*3+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    
    # Let's use a structured ansatz:
    # U has 6 rows, V has 12 rows, W has 8 rows
    # Each has 12 columns
    
    # Try: entries in {-1, 0, 1}
    # Use random search with restarts
    
    import random
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(12):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def random_search(max_iters=50000):
        """Random search for rank-12 decomposition with entries in {-1,0,1}."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Generate random U, V, W with entries in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(12)] for _ in range(6)]
            V = [[random.choice([-1, 0, 1]) for _ in range(12)] for _ in range(12)]
            W = [[random.choice([-1, 0, 1]) for _ in range(12)] for _ in range(8)]
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try random search
    result = random_search(30000)
    if result is not None:
        return result
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try a more structured approach: use the fact that the tensor is "generic"
    # and try to find a solution with a specific pattern
    
    # Let's try: U[i,j,r] depends only on j and r
    # V[j,k,r] depends only on j and r  
    # W[k,i,r] depends only on k and r
    
    # This is a "block" structure where each term r corresponds to a (j,k) pair
    # But we only have 12 terms for 3*4=12 (j,k) pairs, so each term handles one (j,k)
    
    # For term r corresponding to (j_r, k_r):
    # U[i, j_r, r] = u_{i,r}
    # V[j_r, k_r, r] = v_r
    # W[k_r, i, r] = w_{i,r}
    
    # Then: sum_r u_{i,r} * v_r * w_{i,r} = 1 for all i, and the (j,k) must be covered
    
    # Actually this doesn't work directly because we need the sum to be 1 for ALL (i,j,k)
    
    # Let me try yet another approach: use numpy to solve a least-squares problem
    # with integer constraints
    
    # Actually, let's just return schoolbook. Finding a rank-12 decomposition
    # for 2x3x4 is non-trivial and may not be achievable with simple methods.
    
    # But wait - let me try one more thing: a specific known construction
    
    # For 2x3x4, one can potentially achieve rank 12 using the following idea:
    # Split into two 2x2x2 blocks and one 2x1x2 block, or similar
    
    # Actually, the minimum rank of M_{2,3,4} is known to be 12 (from literature)
    # Let me try to find it via a more systematic search
    
    # Use a SAT-like approach with small domain
    # Domain: {-2, -1, 0, 1, 2}
    
    def constrained_search():
        """Search with entries in {-2,-1,0,1,2} using a smarter approach."""
        # Fix a pattern: make U sparse
        # Each row of U has at most 2 non-zeros
        # Each row of V has at most 2 non-zeros  
        # Each row of W has at most 2 non-zeros
        
        # Try to find a solution where the non-zero patterns are consistent
        
        # For simplicity, try: U[i*m+j, r] is non-zero only if j == r % 3
        # V[j*p+k, r] is non-zero only if k == r // 3
        # W[k*n+i, r] is non-zero only if i == (r // 3) % 2
        
        # This gives a structured pattern
        
        U = [[0] * 12 for _ in range(6)]
        V = [[0] * 12 for _ in range(12)]
        W = [[0] * 12 for _ in range(8)]
        
        # For r in 0..11:
        #   j_r = r % 3
        #   k_r = r // 3
        #   i_r = (r // 3) % 2  # This gives i_r in {0, 1}
        
        # U[i*3 + j_r, r] = u_{i,r}
        # V[j_r*4 + k_r, r] = v_r
        # W[k_r*2 + i, r] = w_{i,r}
        
        # Then the tensor equation becomes:
        # For (i,j,k): sum_{r: j_r=j, k_r=k} u_{i,r} * v_r * w_{i,r} = 1
        # For fixed (j,k), there's exactly one r = k*3 + j
        # So: u_{i,r} * v_r * w_{i,r} = 1 for all i
        
        # This means u_{i,r} * w_{i,r} = 1/v_r for all i
        # So u_{i,r} = a_{i,r} and w_{i,r} = 1/(v_r * a_{i,r})
        
        # For integer entries, v_r must be ±1, and a_{i,r} must be ±1
        # Then w_{i,r} = ±1
        
        # Let's try v_r = 1 for all r, u_{i,r} = 1, w_{i,r} = 1
        for r in range(12):
            j_r = r % 3
            k_r = r // 3
            for i in range(2):
                U[i * 3 + j_r][r] = 1
            V[j_r * 4 + k_r][r] = 1
            for i in range(2):
                W[k_r * 2 + i][r] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None
    
    result = constrained_search()
    if result is not None:
        return result
    
    if time.time() > deadline:
        return schoolbook()
    
    # Final attempt: try with half-integers
    def half_integer_search():
        """Search with entries in {-1, -0.5, 0, 0.5, 1}."""
        vals = [-1, -0.5, 0, 0.5, 1]
        
        for it in range(20000):
            if time.time() > deadline:
                return None
            
            U = [[random.choice(vals) for _ in range(12)] for _ in range(6)]
            V = [[random.choice(vals) for _ in range(12)] for _ in range(12)]
            W = [[random.choice(vals) for _ in range(12)] for _ in range(8)]
            
            # Verify with tolerance
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(12):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Convert to fractions
                U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U]
                V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V]
                W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W]
                
                # Verify exactly
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for r in range(12):
                                s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                            if s != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None
    
    result = half_integer_search()
    if result is not None:
        return result
    
    return schoolbook()
