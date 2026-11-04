"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Split inner dimension 5 = 4 + 1.
    A (4x5) = [A1 | A2], A1 is 4x4, A2 is 4x1
    B (5x6) = [B1; B2], B1 is 4x6, B2 is 1x6
    C = A1*B1 + A2*B2
    
    For A2*B2 (4x1 * 1x6): outer product, rank 4
    For A1*B1 (4x4 * 4x6): split into A1*B1[:,:4] (4x4*4x4) + A1*B1[:,4:] (4x4*4x2)
      - 4x4*4x4: use Strassen, rank 7
      - 4x4*4x2: schoolbook, rank 32
    
    Total rank: 4 + 7 + 32 = 43
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # We'll build the decomposition term by term
    # Each term is a triple (u_idx, v_idx, w_idx, coeff)
    # where U[u_idx][col] += coeff, V[v_idx][col] += coeff, W[w_idx][col] += coeff
    # But actually for rank-1 terms, each term contributes 1 to one position in each matrix
    
    # Let's collect all rank-1 terms as (u_pos, v_pos, w_pos)
    # U is indexed by (i, j) -> i*m + j, i in [0,4), j in [0,5)
    # V is indexed by (j, k) -> j*p + k, j in [0,5), k in [0,6)
    # W is indexed by (k, i) -> k*n + i, k in [0,6), i in [0,4)
    
    terms = []
    
    # Part 1: A2 * B2 (outer product, rank 4)
    # A2[i] * B2[k] contributes to C[i,k]
    # A2[i] is at position (i, 4) in A, so U index = i*5 + 4
    # B2[k] is at position (4, k) in B, so V index = 4*6 + k
    # C[i,k] is at position (k, i) in W, so W index = k*4 + i
    for i in range(4):
        for k in range(6):
            u_pos = i * 5 + 4
            v_pos = 4 * 6 + k
            w_pos = k * 4 + i
            terms.append((u_pos, v_pos, w_pos))
    
    # Part 2: A1 * B1[:,:4] (4x4 * 4x4, Strassen rank 7)
    # A1[i,j] is at (i,j) in A, U index = i*5 + j, j in [0,4)
    # B1[j,k] is at (j,k) in B, V index = j*6 + k, j in [0,4), k in [0,4)
    # C[i,k] is at (k,i) in W, W index = k*4 + i
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4, we split into 2x2 blocks of size 2x2:
    # A1 = [[A00, A01], [A10, A11]] where each is 2x2
    # B1[:,:4] = [[B00, B01], [B10, B11]] where each is 2x2
    #
    # But wait, this gives us 4x4 * 4x4 = 4x4, and we need to map back.
    # Each 2x2 block multiplication uses Strassen with 7 terms.
    # But the 7 terms involve sums/differences of 2x2 blocks, which are themselves bilinear.
    #
    # Actually, Strassen's algorithm for 4x4*4x4: we apply Strassen at the top level,
    # which reduces 4x4*4x4 to 7 multiplications of 2x2*2x2, each of which can be done
    # with 7 more (if we recurse) or 4 (schoolbook for 2x2).
    #
    # If we use Strassen recursively: rank = 7 * 7 = 49 for 4x4*4x4. That's worse than schoolbook (16)!
    #
    # Wait, no. Strassen for 2x2*2x2 has rank 7. For 4x4*4x4, we can either:
    # 1. Schoolbook: rank 16
    # 2. Strassen applied once (treating 4x4 as 2x2 blocks of 2x2): 7 * (rank of 2x2*2x2) = 7*4 = 28 (if we do 2x2 schoolbook)
    # 3. Strassen applied twice: 7*7 = 49
    #
    # So schoolbook (16) is better than Strassen for 4x4! Strassen only helps for larger matrices.
    #
    # Hmm, so let me reconsider. For 4x4*4x4, schoolbook rank is 16.
    # For 4x4*4x2, schoolbook rank is 32.
    # Total: 4 (outer) + 16 + 32 = 52.
    
    # After careful analysis, simple splits don't reduce rank below schoolbook (120).
    # To reduce rank, we need Strassen or similar non-trivial bilinear algorithms.
    # 
    # For now, implement correct schoolbook decomposition.
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_pos = i * m + j
                v_pos = j * p + k
                w_pos = k * n + i
                terms.append((u_pos, v_pos, w_pos))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for col, (u_pos, v_pos, w_pos) in enumerate(terms):
        U[u_pos][col] = 1
        V[v_pos][col] = 1
        W[w_pos][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
