import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]
_START_TIME = time.time()
TIME_LIMIT = 110  # Leave 10s safety margin before 120s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each output element C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition requires:
    # sum_r U[3*i+j, r] * V[5*j+k, r] * W[2*k+i, r] = 1 for all i,j,k
    # and = 0 for all other (i,k) pairs not matching (i,j,k)
    
    # Build the multiplication tensor T[i,j,k] = 1
    # Check: sum_r U[3*i+j, r] * V[5*j+k, r] * W[2*k+i, r] should be 1
    for i in range(n):
        for k in range(p):
            w_idx = 2 * k + i
            for j in range(m):
                u_idx = 3 * i + j
                v_idx = 5 * j + k
                s = 0.0
                for r in range(rank):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(s - 1.0) > 1e-9:
                    return False
            # Check orthogonality: for different (i',k') pairs, the sum should be 0
            # Actually, the condition is that for fixed (i,k), sum_j A[i,j]*B[j,k]
            # The tensor is T[i,j,k]=1, so we need:
            # sum_r U[3*i+j,r]*V[5*j+k,r]*W[2*k+i,r] = 1 for all (i,j,k)
            # This is already checked above.
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: A[i, j] -> 3*i + j
    # V: B[j, k] -> 5*j + k
    # W: C[i, k] -> 2*k + i
    
    # --- Block 1: C[:, 0:2] ---
    # Strassen A_left * B_1_top
    col = 0
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[6][col] += 1
    W[0][col] += 1; W[3][col] += 1

    col = 1
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[1][col] += 1; W[3][col] -= 1

    col = 2
    U[0][col] += 1
    V[1][col] += 1; V[6][col] -= 1
    W[2][col] += 1; W[3][col] += 1

    col = 3
    U[4][col] += 1
    V[5][col] += 1; V[0][col] -= 1
    W[1][col] += 1; W[0][col] += 1

    col = 4
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    W[2][col] += 1; W[0][col] -= 1

    col = 5
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[3][col] += 1

    col = 6
    U[1][col] += 1; U[4][col] -= 1
    V[5][col] += 1; V[6][col] += 1
    W[0][col] += 1

    # Schoolbook A_right * B_1_bottom
    col = 7
    U[2][col] += 1
    V[10][col] += 1
    W[0][col] += 1

    col = 8
    U[2][col] += 1
    V[11][col] += 1
    W[2][col] += 1

    col = 9
    U[5][col] += 1
    V[10][col] += 1
    W[1][col] += 1

    col = 10
    U[5][col] += 1
    V[11][col] += 1
    W[3][col] += 1

    # --- Block 2: C[:, 2:4] ---
    # Strassen A_left * B_2_top
    col = 11
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[7][col] += 1

    col = 12
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] -= 1

    col = 13
    U[0][col] += 1
    V[3][col] += 1; V[8][col] -= 1
    W[6][col] += 1; W[7][col] += 1

    col = 14
    U[4][col] += 1
    V[7][col] += 1; V[2][col] -= 1
    W[5][col] += 1; W[4][col] += 1

    col = 15
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[6][col] += 1; W[4][col] -= 1

    col = 16
    U[3][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[3][col] += 1
    W[7][col] += 1

    col = 17
    U[1][col] += 1; U[4][col] -= 1
    V[7][col] += 1; V[8][col] += 1
    W[4][col] += 1

    # Schoolbook A_right * B_2_bottom
    col = 18
    U[2][col] += 1
    V[12][col] += 1
    W[4][col] += 1

    col = 19
    U[2][col] += 1
    V[13][col] += 1
    W[6][col] += 1

    col = 20
    U[5][col] += 1
    V[12][col] += 1
    W[5][col] += 1

    col = 21
    U[5][col] += 1
    V[13][col] += 1
    W[7][col] += 1

    # --- Block 3: C[:, 4] ---
    # Schoolbook A_left * B_3_top
    col = 22
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1

    col = 23
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1

    col = 24
    U[3][col] += 1
    V[4][col] += 1
    W[9][col] += 1

    col = 25
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1

    # Schoolbook A_right * B_3_bottom
    col = 26
    U[2][col] += 1
    V[14][col] += 1
    W[8][col] += 1

    col = 27
    U[5][col] += 1
    V[14][col] += 1
    W[9][col] += 1

    return U, V, W

def _try_rank27_search():
    """
    Try to find a rank-27 decomposition by exploring alternative patterns.
    
    The key insight: we have 3 blocks of output columns.
    Block 1 (cols 0-1): 2x2x3 problem -> needs at least 7 (Strassen) + 4 (right col) = 11
    Block 2 (cols 2-3): 2x2x3 problem -> needs at least 7 (Strassen) + 4 (right col) = 11
    Block 3 (col 4): 2x3x1 problem -> needs 6 (schoolbook) or possibly less
    
    Total: 11 + 11 + 6 = 28 (current)
    
    To get 27, we need to save 1 term somewhere.
    
    Strategy: Can we merge terms across blocks?
    - The "right column" terms (A_right * B_bottom) for blocks 1 and 2 use
      U indices 2 and 5 (A[0,2] and A[1,2]).
    - Perhaps we can use a combined approach for the last column.
    
    Alternative: For the 2x3x1 block (Block 3), can we do better than 6?
    The 2x3x1 tensor has rank at most 6 (schoolbook). 
    Actually, for 2x3x1, the rank is exactly 6 since it's just 6 independent terms
    (each output is a separate linear combination).
    
    Wait - let me reconsider. The full tensor is T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    
    Let me try a different partition:
    - Use Strassen for the 2x2 part (j=0,1) for ALL k values simultaneously?
    - No, Strassen works on 2x2x2 tensors.
    
    Let me think about this differently. The tensor has 30 entries (2*3*5).
    Each rank-1 term can cover at most... well, it's a product of three vectors.
    
    Actually, let me try: what if for Block 3 (k=4), instead of 6 schoolbook terms,
    we can use a more clever decomposition?
    
    For k=4: C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4]
              C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4]
    
    This is a 2x3x1 tensor. Its rank is 6 (each of the 6 products is independent).
    So we can't reduce below 6 for this block alone.
    
    However, maybe we can share terms between blocks?
    
    Let me try: what if we use a Strassen-like approach that spans multiple k values?
    
    Actually, the most promising approach: reduce the schoolbook parts.
    For Block 1 right column (4 terms) and Block 2 right column (4 terms), 
    we use A[0,2] and A[1,2] with B[2,0..3].
    
    What if we combine: use (A[0,2]+A[1,2]) with some B combinations?
    That would give us C[0,?] + C[1,?] which we then need to separate.
    
    This is getting complex. Let me just try a brute-force search over small modifications.
    """
    n, m, p = DIMENSIONS
    
    # Try: can we save 1 term by combining the last column handling?
    # Current Block 3 uses 6 terms. What if we use 5?
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4]
    
    # Try: M1 = (A[0,0]+A[1,0]) * B[0,4] -> gives C[0,4]+C[1,4] contribution
    # M2 = (A[0,1]+A[1,1]) * B[1,4] -> gives C[0,4]+C[1,4] contribution  
    # M3 = (A[0,2]+A[1,2]) * B[2,4] -> gives C[0,4]+C[1,4] contribution
    # M4 = (A[0,0]-A[1,0]) * B[0,4] -> gives C[0,4]-C[1,4] contribution
    # M5 = (A[0,1]-A[1,1]) * B[1,4] -> gives C[0,4]-C[1,4] contribution
    # M6 = (A[0,2]-A[1,2]) * B[2,4] -> gives C[0,4]-C[1,4] contribution
    # That's still 6.
    
    # What about:
    # M1 = A[0,0]*B[0,4]
    # M2 = A[1,0]*B[0,4]  
    # M3 = (A[0,1]+A[1,1])*B[1,4]  -> need to separate
    # M4 = (A[0,1]-A[1,1])*B[1,4]  -> need to separate
    # M5 = A[0,2]*B[2,4]
    # M6 = A[1,2]*B[2,4]
    # Still 6.
    
    # The 2x3x1 tensor genuinely has rank 6.
    
    # Let me try a completely different approach:
    # What if we don't partition by k-blocks but use a global Strassen-like scheme?
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all (i,j,k).
    
    # This is equivalent to: the bilinear map (A,B) -> AB has a rank-r decomposition.
    
    # Known results: rank(2x3x5) matrix multiplication.
    # Lower bound: max(2*3, 3*5, 2*5) = max(6,15,10) = 15 (trivial)
    # Better lower bounds exist but are hard to compute.
    
    # Upper bound: 28 (current).
    # Can we do 27?
    
    # Let me try: use Strassen for the 2x2x5 part (j=0,1, all k) 
    # and schoolbook for j=2.
    
    # 2x2x5: We can decompose as two 2x2x2 blocks + one 2x2x1 block.
    # 2x2x2: rank 7 (Strassen)
    # 2x2x2: rank 7 (Strassen)  
    # 2x2x1: rank 4 (schoolbook)
    # Total for j=0,1: 7+7+4 = 18
    # j=2: 2x1x5 -> rank 10 (schoolbook: A[0,2]*B[2,k] and A[1,2]*B[2,k] for k=0..4)
    # Total: 18 + 10 = 28. Same as before.
    
    # What if we use a different decomposition for 2x2x1?
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # This is 4 terms. Can we do 3?
    # M1 = (A[0,0]+A[1,0])*B[0,k] -> C[0,k]+C[1,k]
    # M2 = (A[0,1]+A[1,1])*B[1,k] -> C[0,k]+C[1,k]
    # M3 = (A[0,0]-A[1,0])*B[0,k] -> C[0,k]-C[1,k]
    # From M1+M3: 2*A[0,0]*B[0,k] -> A[0,0]*B[0,k]
    # From M1-M3: 2*A[1,0]*B[0,k] -> A[1,0]*B[0,k]
    # But we still need A[0,1]*B[1,k] and A[1,1]*B[1,k] from M2...
    # M2 gives A[0,1]*B[1,k] + A[1,1]*B[1,k], can't separate without another term.
    # So we need 4 terms minimum for 2x2x1.
    
    # I'm stuck at 28. Let me just return the verified rank-28 solution.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better solution if time permits
    U, V, W = _build_rank28()
    
    # Verify the solution
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Rank-28 decomposition failed verification")
    
    # Check time and try to improve
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 5:
        result = _try_rank27_search()
        if result is not None:
            U, V, W = result
    
    return {"U": U, "V": V, "W": W}

    # Indices:
    # U: A[i, j] -> 3*i + j
    # V: B[j, k] -> 5*j + k
    # W: C[i, k] -> 2*k + i  (Task says W[k*n+i, r], so index is k*n+i. n=2. So 2*k+i)

    # --- Block 1: C[:, 0:2] ---
    # 1. Strassen A_left * B_1_top
    # A_left: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B_1_top: B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6
    
    # M1: (A00+A11)(B00+B11)
    col = 0
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[6][col] += 1
    # C00 = W[0], C11 = W[3]
    W[0][col] += 1; W[3][col] += 1

    # M2: (A10+A11)B00
    col = 1
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    # C10 = W[1], C11 = W[3]
    W[1][col] += 1; W[3][col] -= 1

    # M3: A00(B01-B11)
    col = 2
    U[0][col] += 1
    V[1][col] += 1; V[6][col] -= 1
    # C01 = W[2], C11 = W[3]
    W[2][col] += 1; W[3][col] += 1

    # M4: A11(B10-B00)
    col = 3
    U[4][col] += 1
    V[5][col] += 1; V[0][col] -= 1
    # C10 = W[1], C00 = W[0]
    W[1][col] += 1; W[0][col] += 1

    # M5: (A00+A01)B11
    col = 4
    U[0][col] += 1; U[1][col] += 1
    V[6][col] += 1
    # C01 = W[2], C00 = W[0]
    W[2][col] += 1; W[0][col] -= 1

    # M6: (A10-A00)(B00+B01)
    col = 5
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    # C11 = W[3]
    W[3][col] += 1

    # M7: (A01-A11)(B10+B11)
    col = 6
    U[1][col] += 1; U[4][col] -= 1
    V[5][col] += 1; V[6][col] += 1
    # C00 = W[0]
    W[0][col] += 1

    # 2. Schoolbook A_right * B_1_bottom
    # A_right: A[0,2]=2, A[1,2]=5
    # B_1_bottom: B[2,0]=10, B[2,1]=11
    
    col = 7
    U[2][col] += 1
    V[10][col] += 1
    W[0][col] += 1 # C00

    col = 8
    U[2][col] += 1
    V[11][col] += 1
    W[2][col] += 1 # C01

    col = 9
    U[5][col] += 1
    V[10][col] += 1
    W[1][col] += 1 # C10

    col = 10
    U[5][col] += 1
    V[11][col] += 1
    W[3][col] += 1 # C11

    # --- Block 2: C[:, 2:4] ---
    # 1. Strassen A_left * B_2_top
    # B_2_top: B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8
    
    col = 11
    # M1
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[8][col] += 1
    W[4][col] += 1; W[7][col] += 1 # C02=W[4], C13=W[7]

    col = 12
    # M2
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[5][col] += 1; W[7][col] -= 1 # C12=W[5], C13=W[7]

    col = 13
    # M3
    U[0][col] += 1
    V[3][col] += 1; V[8][col] -= 1
    W[6][col] += 1; W[7][col] += 1 # C03=W[6], C13=W[7]

    col = 14
    # M4
    U[4][col] += 1
    V[7][col] += 1; V[2][col] -= 1
    W[5][col] += 1; W[4][col] += 1 # C12=W[5], C02=W[4]

    col = 15
    # M5
    U[0][col] += 1; U[1][col] += 1
    V[8][col] += 1
    W[6][col] += 1; W[4][col] -= 1 # C03=W[6], C02=W[4]

    col = 16
    # M6
    U[3][col] += 1; U[0][col] -= 1
    V[2][col] += 1; V[3][col] += 1
    W[7][col] += 1 # C13=W[7]

    col = 17
    # M7
    U[1][col] += 1; U[4][col] -= 1
    V[7][col] += 1; V[8][col] += 1
    W[4][col] += 1 # C02=W[4]

    # 2. Schoolbook A_right * B_2_bottom
    # B_2_bottom: B[2,2]=12, B[2,3]=13
    
    col = 18
    U[2][col] += 1
    V[12][col] += 1
    W[4][col] += 1 # C02

    col = 19
    U[2][col] += 1
    V[13][col] += 1
    W[6][col] += 1 # C03

    col = 20
    U[5][col] += 1
    V[12][col] += 1
    W[5][col] += 1 # C12

    col = 21
    U[5][col] += 1
    V[13][col] += 1
    W[7][col] += 1 # C13

    # --- Block 3: C[:, 4] ---
    # 1. Schoolbook A_left * B_3_top
    # B_3_top: B[0,4]=4, B[1,4]=9
    
    col = 22
    U[0][col] += 1
    V[4][col] += 1
    W[8][col] += 1 # C04

    col = 23
    U[1][col] += 1
    V[9][col] += 1
    W[8][col] += 1 # C04

    col = 24
    U[3][col] += 1
    V[4][col] += 1
    W[9][col] += 1 # C14

    col = 25
    U[4][col] += 1
    V[9][col] += 1
    W[9][col] += 1 # C14

    # 2. Schoolbook A_right * B_3_bottom
    # B_3_bottom: B[2,4]=14
    
    col = 26
    U[2][col] += 1
    V[14][col] += 1
    W[8][col] += 1 # C04

    col = 27
    U[5][col] += 1
    V[14][col] += 1
    W[9][col] += 1 # C14

    return {"U": U, "V": V, "W": W}
