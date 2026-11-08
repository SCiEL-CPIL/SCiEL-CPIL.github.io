"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,1,2], k in [0,1,2].
    We need U (6 x r), V (9 x r), W (6 x r) such that:
      T[i,j,k] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*2+i, c]
    
    Schoolbook gives rank 18. We attempt to find a lower rank decomposition.
    """
    n, m, p = 2, 3, 3
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify_decomp(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Generate schoolbook as baseline
    U_sb = np.zeros((6, 18), dtype=int)
    V_sb = np.zeros((9, 18), dtype=int)
    W_sb = np.zeros((6, 18), dtype=int)
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*2+i, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = 18
    
    # Try structured rank-16 construction based on overlapping 2x2 blocks
    # For 2x3 * 3x3, we can try to use Strassen-like ideas on the 3x3 part
    
    # Key insight: Split the 3x3 B matrix into overlapping 2x2 submatrices
    # and use linear combinations to reduce products
    
    # Try a specific algebraic construction for rank 16:
    # Use the fact that we only have 2 rows in A, so we can share structure
    
    # Construction attempt:
    # Let's define 16 products using the following linear combinations:
    
    # For row 0 of A: [a00, a01, a02]
    # For row 1 of A: [a10, a11, a12]
    # B is 3x3: [b00 b01 b02; b10 b11 b12; b20 b21 b22]
    
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    
    # Try rank 16 with a structured pattern:
    # Use 8 products for the "main" terms and 8 for corrections
    
    # Let me try a direct search with constrained random coefficients
    
    import random
    random.seed(42)
    np.random.seed(42)
    
    def try_structured_rank16(max_iters=5000):
        """Try structured search for rank 16 with small integer coefficients."""
        # Strategy: Start with schoolbook and try to merge terms
        # Or: Use random coefficients but with structure
        
        # Try different coefficient ranges
        for coeff_max in [2, 3, 4]:
            for it in range(max_iters // 3):
                if time.time() > deadline:
                    return None
                
                # Generate with some structure: many zeros, few non-zeros
                U = np.zeros((6, 16), dtype=int)
                V = np.zeros((9, 16), dtype=int)
                W = np.zeros((6, 16), dtype=int)
                
                for c in range(16):
                    # Each term should involve a few A entries, B entries, and C entries
                    # Pick random subsets
                    u_indices = random.sample(range(6), random.randint(1, 3))
                    v_indices = random.sample(range(9), random.randint(1, 3))
                    w_indices = random.sample(range(6), random.randint(1, 3))
                    
                    for idx in u_indices:
                        U[idx, c] = random.randint(-coeff_max, coeff_max)
                    for idx in v_indices:
                        V[idx, c] = random.randint(-coeff_max, coeff_max)
                    for idx in w_indices:
                        W[idx, c] = random.randint(-coeff_max, coeff_max)
                
                if verify_decomp(U, V, W):
                    return (U, V, W)
        
        return None
    
    def try_random_rank(r, max_iters=3000, coeff_range=(-2, 2)):
        """Try random integer coefficient search for given rank."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            U = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
            V = np.random.randint(coeff_range[0], coeff_range[1]+1, (9, r))
            W = np.random.randint(coeff_range[0], coeff_range[1]+1, (6, r))
            if verify_decomp(U, V, W):
                return (U, V, W)
        return None
    
    # Try structured rank-16 search
    result = try_structured_rank16(max_iters=3000)
    if result is not None:
        best_rank = 16
        best_decomp = result
    elif time.time() < deadline - 15:
        # Try rank 17
        result = try_random_rank(17, max_iters=2000)
        if result is not None:
            best_rank = 17
            best_decomp = result
        elif time.time() < deadline - 5:
            # Try rank 16 with plain random
            result = try_random_rank(16, max_iters=2000)
            if result is not None:
                best_rank = 16
                best_decomp = result
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
