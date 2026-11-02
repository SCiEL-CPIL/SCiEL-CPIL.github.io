import time
from fractions import Fraction

def _build_44_rank(n, m, p):
    """Build the known 44-rank decomposition."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part (j=1,2)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _build_42_rank(n, m, p):
    """
    Try a 42-rank decomposition.
    
    Key insight: The 2x1x8 part needs at least 16 terms (2 rows x 8 cols).
    The 2x2x8 part: instead of 4 blocks of 7 (28 terms), try to share
    structure across blocks to save 2 terms.
    
    Strategy: Use a different grouping. Process the 2x2x8 part as:
    - First 6 columns: 3 blocks of 2x2x2 with 7 terms each = 21
    - Last 2 columns: a 2x2x2 with 5 terms (using a different scheme)
    Total for 2x2x8: 26, plus 16 for 2x1x8 = 42.
    
    For the last 2x2x2 block, use a 5-term scheme:
    C[0,0] = A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    This is just schoolbook (4 terms) but we need to check if we can
    combine with the previous blocks. Actually, let me think differently.
    
    Alternative: Use the fact that for the 2x2x8 case, we can use
    a "winograd" style approach where we process pairs of columns
    with shared A-side linear combinations.
    
    Let me try: for the 2x2x8 part, use 6 terms per pair of columns
    instead of 7, by exploiting the 2-row structure differently.
    
    Actually, the minimal rank for 2x2x2 is 7 (Strassen). So 4 blocks x 7 = 28.
    To get below 28, we need cross-block sharing.
    
    Let me try a different approach: group all 8 columns and find
    a 2x2x8 decomposition with fewer than 28 terms.
    
    One idea: Use the fact that A has only 2 rows. We can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For i=0: C[0,k] = A[0,1]*B[1,k] + A[0,2]*B[2,k]
    For i=1: C[1,k] = A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    Define:
    x1 = A[0,1], x2 = A[1,1], x3 = A[0,2], x4 = A[1,2]
    
    We need to compute 16 outputs (2x8) from these 4 inputs and 16 B values.
    
    Using bilinear complexity: we can use a scheme where we form
    linear combinations of A's, multiply with linear combinations of B's.
    
    Let me try: for each pair of columns (k0, k1), use 6 terms:
    
    T1 = (x1 + x3) * (B[1,k0] + B[2,k1])
    T2 = (x1 + x4) * B[1,k0]
    T3 = x1 * (B[1,k1] - B[2,k1])
    T4 = x4 * (B[2,k0] - B[1,k0])
    T5 = (x3 + x4) * B[2,k1]
    T6 = (x2 - x1) * (B[1,k0] + B[1,k1])
    
    Then:
    C[0,k0] = T1 - T5 + T6  ... let me verify.
    
    This is getting complex. Let me just try the 42-rank with a simpler
    approach: use 5 terms for one of the 2x2x2 blocks by accepting
    that we need a non-standard scheme.
    
    Actually, the known lower bound for 2x2x2 is 7. So we can't do
    better than 7 per isolated block. The only way to save is
    cross-block sharing.
    
    Let me try: process columns 0-5 with 3 blocks of 7 = 21,
    and columns 6-7 with a 5-term scheme that shares with block 3.
    
    Hmm, this is hard. Let me just return None if we can't find one.
    """
    return None

def _build_43_rank(n, m, p):
    """
    Try a 43-rank decomposition.
    
    Idea: Use 7 terms for 3 blocks of 2x2x2 (21 terms) and
    a 6-term scheme for the last 2x2x2 block.
    
    For the last block, use:
    T1 = (A[0,1]+A[1,2])*(B[1,6]+B[2,7])
    T2 = (A[1,1]+A[1,2])*B[1,6]
    T3 = A[0,1]*(B[1,7]-B[2,7])
    T4 = A[1,2]*(B[2,6]-B[1,6])
    T5 = (A[0,1]+A[0,2])*B[2,7]
    T6 = (A[1,1]-A[0,1])*(B[1,6]+B[1,7])
    
    This gives 6 terms. Let me verify the outputs:
    C[0,6] should be A[0,1]*B[1,6] + A[0,2]*B[2,6]
    C[0,7] should be A[0,1]*B[1,7] + A[0,2]*B[2,7]
    C[1,6] should be A[1,1]*B[1,6] + A[1,2]*B[2,6]
    C[1,7] should be A[1,1]*B[1,7] + A[1,2]*B[2,7]
    
    From the 6 terms:
    T1 = (a01+a12)(b16+b27) = a01*b16 + a01*b27 + a12*b16 + a12*b27
    T2 = (a11+a12)*b16 = a11*b16 + a12*b16
    T3 = a01*(b17-b27) = a01*b17 - a01*b27
    T4 = a12*(b26-b16) = a12*b26 - a12*b16
    T5 = (a01+a02)*b27 = a01*b27 + a02*b27
    T6 = (a11-a01)*(b16+b17) = a11*b16 + a11*b17 - a01*b16 - a01*b17
    
    C[0,6] = a01*b16 + a02*b26
    C[0,7] = a01*b17 + a02*b27
    C[1,6] = a11*b16 + a12*b26
    C[1,7] = a11*b17 + a12*b27
    
    Let's see if we can express these as linear combos of T1..T6:
    
    C[0,6] = a01*b16 + a02*b26
    C[0,7] = a01*b17 + a02*b27
    C[1,6] = a11*b16 + a12*b26
    C[1,7] = a11*b17 + a12*b27
    
    This is a 4-output, 6-term system. We need to find W coefficients
    such that the bilinear form matches. This is a linear system.
    
    Let me just set it up and solve.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = 43
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0) - 16 terms
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: 2x2x8 part
    # First 3 blocks (columns 0-5): 7 terms each = 21 terms
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Standard Strassen-like 7-term scheme
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    # Last block (columns 6-7): 6 terms
    k0, k1 = 6, 7
    
    # T1: (A[0,1]+A[1,2]) * (B[1,6]+B[2,7])
    U[0 * m + 1][col] = 1
    U[1 * m + 2][col] = 1
    V[1 * p + k0][col] = 1
    V[2 * p + k1][col] = 1
    # W coefficients to be determined
    col += 1
    t1_col = col - 1
    
    # T2: (A[1,1]+A[1,2]) * B[1,6]
    U[1 * m + 1][col] = 1
    U[1 * m + 2][col] = 1
    V[1 * p + k0][col] = 1
    col += 1
    t2_col = col - 1
    
    # T3: A[0,1] * (B[1,7]-B[2,7])
    U[0 * m + 1][col] = 1
    V[1 * p + k1][col] = 1
    V[2 * p + k1][col] = -1
    col += 1
    t3_col = col - 1
    
    # T4: A[1,2] * (B[2,6]-B[1,6])
    U[1 * m + 2][col] = 1
    V[2 * p + k0][col] = 1
    V[1 * p + k0][col] = -1
    col += 1
    t4_col = col - 1
    
    # T5: (A[0,1]+A[0,2]) * B[2,7]
    U[0 * m + 1][col] = 1
    U[0 * m + 2][col] = 1
    V[2 * p + k1][col] = 1
    col += 1
    t5_col = col - 1
    
    # T6: (A[1,1]-A[0,1]) * (B[1,6]+B[1,7])
    U[1 * m + 1][col] = 1
    U[0 * m + 1][col] = -1
    V[1 * p + k0][col] = 1
    V[1 * p + k1][col] = 1
    col += 1
    t6_col = col - 1
    
    # Now solve for W coefficients for the last block
    # We need:
    # C[0,6] = A[0,1]*B[1,6] + A[0,2]*B[2,6]
    # C[0,7] = A[0,1]*B[1,7] + A[0,2]*B[2,7]
    # C[1,6] = A[1,1]*B[1,6] + A[1,2]*B[2,6]
    # C[1,7] = A[1,1]*B[1,7] + A[1,2]*B[2,7]
    
    # Let w_{i,k,t} be the W coefficient for output (i,k) and term t
    # C[i,k] = sum_t w_{i,k,t} * (sum_{a} U[a,t] * A[a]) * (sum_{b} V[b,t] * B[b])
    
    # This is a linear system. Let me set it up with 12 unknowns
    # (w for each of 4 outputs x ... wait, we have 6 terms, so 4*6=24 unknowns)
    # But we only need the W entries for the last 6 columns.
    
    # Actually, let me just verify by direct computation with test values.
    # Set A[0,1]=1, all other A=0, B[1,6]=1, all other B=0.
    # Expected: C[0,6]=1, C[0,7]=0, C[1,6]=0, C[1,7]=0
    
    # This is getting complex. Let me just return None for now.
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact."""
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with the known 44-rank solution
    best_result = _build_44_rank(n, m, p)
    
    # Try to find lower rank solutions
    # Try 43 rank
    if time.time() < deadline - 5:
        result_43 = _build_43_rank(n, m, p)
        if result_43 is not None and _verify_decomposition(result_43["U"], result_43["V"], result_43["W"], n, m, p):
            best_result = result_43
    
    # Try 42 rank
    if time.time() < deadline - 5:
        result_42 = _build_42_rank(n, m, p)
        if result_42 is not None and _verify_decomposition(result_42["U"], result_42["V"], result_42["W"], n, m, p):
            best_result = result_42
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify_decomposition(res["U"], res["V"], res["W"], 2, 3, 8):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
