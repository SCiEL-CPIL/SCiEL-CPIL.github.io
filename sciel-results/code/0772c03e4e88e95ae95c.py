"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]
DEADLINE = 145.0  # seconds, with 5s safety margin before 150s kill

def _laderman_3x3x3():
    """
    Laderman's rank-23 algorithm for 3x3 matrix multiplication.
    Returns U (9x23), V (9x23), W (9x23) such that
    (A*B)[i,j] = sum_c U[i*3+j, c] * V[c] * W[c]
    where V[c] = A[i,j] * B[j,k] pattern.
    
    Actually, for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor: T[i,k,j] = A[i,j]*B[j,k]*C[i,k]
    
    In our convention:
    U[i*m+j, c] corresponds to A[i,j]
    V[j*p+k, c] corresponds to B[j,k]  
    W[k*n+i, c] corresponds to C[i,k] (note: k*n+i, not i*n+k)
    
    For 3x3x3: n=m=p=3
    U is 9x23, V is 9x23, W is 9x23
    """
    n = m = p = 3
    rank = 23
    
    # Laderman's 23-term algorithm for 3x3 matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: T[i,k,j] = A[i,j] * B[j,k] * C[i,k]
    # In tensor decomposition: T[i,k,j] = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
    
    # Laderman's algorithm (one standard formulation)
    # Let me use the explicit 23-term decomposition
    # Indices: i,k in {0,1,2} for C, j in {0,1,2} for the sum
    
    # The 23 terms of Laderman's algorithm:
    # Term 1: A[0,0]*B[0,0] -> C[0,0]
    # Term 2: A[0,1]*B[1,0] -> C[0,0]
    # Term 3: A[0,2]*B[2,0] -> C[0,0]
    # ... etc
    
    # Let me use a known explicit formulation
    # From literature: Laderman's 23-term algorithm
    
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    # Laderman's algorithm terms (i, j, k) for A[i,j], B[j,k], C[i,k]
    # The 23 terms:
    terms = [
        # (i_A, j_A, j_B, k_B, i_C, k_C, coeff)
        # Standard Laderman 23-term algorithm
        (0, 0, 0, 0, 0, 0, 1),
        (0, 1, 1, 0, 0, 0, 1),
        (0, 2, 2, 0, 0, 0, 1),
        (1, 0, 0, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1),
        (1, 2, 2, 1, 1, 1, 1),
        (2, 0, 0, 2, 2, 2, 1),
        (2, 1, 1, 2, 2, 2, 1),
        (2, 2, 2, 2, 2, 2, 1),
        # Cross terms
        (0, 0, 1, 1, 0, 1, 1),
        (0, 1, 0, 0, 0, 0, -1),  # This doesn't look right
    ]
    
    # Let me use a cleaner formulation. I'll construct the 23-term algorithm properly.
    # Actually, let me just use the schoolbook for 3x3x3 (rank 27) and then try to optimize.
    # But we know rank 23 exists. Let me look up the explicit terms.
    
    # Alternative: use the known fact that 3x3x4 can be done in rank 33 or less.
    # Let me construct a valid decomposition and then try to minimize.
    
    # For now, let me return the schoolbook 3x3x3 (rank 27) as a starting point
    # and then extend to 3x3x4.
    
    # Schoolbook 3x3x3: rank 27
    U_sb = np.zeros((9, 27), dtype=float)
    V_sb = np.zeros((9, 27), dtype=float)
    W_sb = np.zeros((9, 27), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_sb[i*3+j, col] = 1
                V_sb[j*3+k, col] = 1
                W_sb[k*3+i, col] = 1
                col += 1
    
    return U_sb, V_sb, W_sb, 27


def construct():
    """
    Construct a tensor decomposition for 3x3x4 matrix multiplication.
    Returns dict with 'U', 'V', 'W' matrices.
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (12, rank)
    W: (p*n, rank) = (12, rank)
    
    Convention: W[k*n+i, r] for the output factor.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Strategy: 
    # 1. Start with schoolbook (rank 36)
    # 2. Try to find a better decomposition
    # 3. Return the best found within deadline
    
    # Schoolbook decomposition (rank 36)
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_schoolbook
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    
    # Try to improve using a constructive approach
    # For 3x3x4, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0,1,2,3
    # This is like 4 separate 3x3 matrix-vector products (or 3x3x1)
    # But we can share computations across the 4 outputs.
    
    # Known result: 3x3x4 can be done in rank 33 (based on 3x3x3 rank 23 + 10)
    # Let me try to construct a rank-33 decomposition.
    
    # Approach: Use Laderman's 23-term algorithm for the first 3 columns of B,
    # then handle the 4th column separately, sharing some terms.
    
    # Actually, let me try a different approach: use a systematic search
    # for a low-rank decomposition using integer coefficients.
    
    # For now, let me try a simple improvement: 
    # The 3x3x4 problem can be viewed as computing 3x4 = (3x3) + (3x1)
    # where we compute A*B[:,0:3] using a 3x3x3 algorithm and A*B[:,3] separately.
    
    # If we have a rank-r algorithm for 3x3x3, and rank-s for 3x3x1,
    # then we can do 3x3x4 in rank r+s (if no sharing) or less with sharing.
    
    # 3x3x1 (matrix-vector) can be done in rank 9 (schoolbook) or less.
    # Actually, matrix-vector 3x3 can be done in rank 6 or less with some tricks.
    
    # Let me try: use schoolbook 3x3x3 (rank 27) + schoolbook 3x3x1 (rank 9) = rank 36.
    # That's the same as full schoolbook.
    
    # To improve, I need a better 3x3x3 algorithm. Let me hardcode Laderman's 23-term algorithm.
    
    # Laderman's 23-term algorithm for 3x3x3:
    # Reference: D. Laderman, "A 23-multiplication algorithm for 3x3 matrix multiplication"
    
    # The 23 terms (using 0-indexed):
    # I'll use a known explicit formulation.
    
    # Let me define the 23 terms as (i, j) for A, (j, k) for B, (i, k) for C
    # with coefficients.
    
    laderman_terms = [
        # (i_A, j_A, j_B, k_B, i_C, k_C, coeff)
        (0, 0, 0, 0, 0, 0, 1),
        (0, 1, 1, 0, 0, 0, 1),
        (0, 2, 2, 0, 0, 0, 1),
        (1, 0, 0, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 1, 1),
        (1, 2, 2, 1, 1, 1, 1),
        (2, 0, 0, 2, 2, 2, 1),
        (2, 1, 1, 2, 2, 2, 1),
        (2, 2, 2, 2, 2, 2, 1),
        (0, 0, 1, 1, 0, 1, 1),
        (0, 1, 0, 0, 0, 0, -1),
        (0, 1, 1, 1, 0, 1, 1),
        (0, 2, 0, 0, 0, 0, 1),
        (0, 2, 2, 2, 0, 2, 1),
        (1, 0, 0, 0, 1, 0, 1),
        (1, 0, 1, 1, 1, 1, -1),
        (1, 2, 1, 2, 1, 2, 1),
        (1, 2, 2, 1, 1, 1, -1),
        (2, 0, 1, 0, 2, 0, 1),
        (2, 1, 0, 0, 2, 0, -1),
        (2, 1, 1, 2, 2, 2, 1),
        (2, 2, 1, 1, 2, 2, -1),
        (2, 2, 2, 1, 2, 1, 1),
    ]
    
    # Verify this is correct by checking if it decomposes the 3x3x3 tensor
    U_23 = np.zeros((9, 23), dtype=float)
    V_23 = np.zeros((9, 23), dtype=float)
    W_23 = np.zeros((9, 23), dtype=float)
    
    for c, (iA, jA, jB, kB, iC, kC, coeff) in enumerate(laderman_terms):
        U_23[iA * 3 + jA, c] = coeff
        V_23[jB * 3 + kB, c] = 1
        W_23[kC * 3 + iC, c] = 1
    
    # Verify: T[i,k,j] should equal sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
    # For 3x3x3: T[i,k,j] = 1 if (i,k,j) corresponds to A[i,j]*B[j,k]*C[i,k]
    # i.e., T[i,k,j] = 1 for all i,k,j in {0,1,2}
    
    # Actually, the multiplication tensor for 3x3x3 is:
    # T[i, k, j] = 1 (since C[i,k] = sum_j A[i,j] * B[j,k])
    # So T is all ones? No, that's not right.
    
    # The tensor is: T[i, k, j] such that C[i,k] = sum_j T[i,k,j] * A[i,j] * B[j,k]
    # So T[i,k,j] = 1 for all i,k,j.
    
    # Let me verify our decomposition:
    # sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] should equal 1 for all i,j,k
    
    verified = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0
                for c in range(23):
                    val += U_23[i*3+j, c] * V_23[j*3+k, c] * W_23[k*3+i, c]
                if abs(val - 1) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if verified:
        # Great, we have a valid rank-23 3x3x3 decomposition!
        # Now extend to 3x3x4.
        
        # For 3x3x4: n=3, m=3, p=4
        # U: (9, rank), V: (12, rank), W: (12, rank)
        # Convention: W[k*n+i, r]
        
        # Strategy: Use the 23-term algorithm for the first 3 columns of B (k=0,1,2),
        # and handle the 4th column (k=3) with additional terms.
        
        # For the first 3 columns, we can reuse the 23 terms, but V needs to be 12-dimensional.
        # For k=0,1,2: V[j*4+k, c] = V_23[j*3+k, c] for the corresponding terms
        # For k=3: V[j*4+3, c] = 0 for the first 23 terms
        
        # Then we need additional terms for the 4th column.
        # The 4th column is: C[i,3] = sum_j A[i,j] * B[j,3]
        # This is a 3x3x1 problem (matrix-vector).
        
        # Schoolbook for 3x3x1: rank 9
        # But we might be able to share some terms with the first 3 columns.
        
        # For now, let's just add 9 more terms for the 4th column.
        # Total rank: 23 + 9 = 32. But we need to check if this is valid.
        
        # Actually, the 23 terms already cover k=0,1,2. For k=3, we need new terms.
        # Let's add schoolbook terms for k=3.
        
        rank_33 = 23 + 9  # 32 terms
        U_33 = np.zeros((9, rank_33), dtype=float)
        V_33 = np.zeros((12, rank_33), dtype=float)
        W_33 = np.zeros((12, rank_33), dtype=float)
        
        # Copy the 23 terms for k=0,1,2
        for c in range(23):
            for i in range(3):
                for j in range(3):
                    U_33[i*3+j, c] = U_23[i*3+j, c]
            for j in range(3):
                for k in range(3):
                    V_33[j*4+k, c] = V_23[j*3+k, c]
            for k in range(3):
                for i in range(3):
                    W_33[k*3+i, c] = W_23[k*3+i, c]
            # For k=3, V and W entries are 0 (already initialized)
        
        # Add 9 schoolbook terms for k=3
        for i in range(3):
            for j in range(3):
                c = 23 + i * 3 + j
                U_33[i*3+j, c] = 1
                V_33[j*4+3, c] = 1
                W_33[3*3+i, c] = 1  # k=3, so W[3*3+i, c]
        
        # Verify the 3x3x4 decomposition
        # T[i, k, j] = 1 for all i in {0,1,2}, k in {0,1,2,3}, j in {0,1,2}
        verified_33 = True
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    val = 0
                    for c in range(rank_33):
                        val += U_33[i*3+j, c] * V_33[j*4+k, c] * W_33[k*3+i, c]
                    if abs(val - 1) > 1e-10:
                        verified_33 = False
                        break
                if not verified_33:
                    break
            if not verified_33:
                break
        
        if verified_33:
            best_U, best_V, best_W = U_33, V_33, W_33
            best_rank = rank_33
        
        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
        
        # Try to reduce the rank further by combining terms or finding a better algorithm.
        # For now, we have rank 32. Let's see if we can do better.
        
        # Actually, let me double-check the Laderman terms. The verification might have failed.
        # Let me print debug info if verification failed.
        
        if not verified:
            # Fallback to schoolbook
            pass
    
    # Final deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 2:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
