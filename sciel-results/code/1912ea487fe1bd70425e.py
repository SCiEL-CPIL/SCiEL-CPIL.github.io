"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, full_check=False):
    """Verify that U ⊗ V ⊗ W exactly equals the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all (i,j,k) positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1.0) > 1e-9:
                    return False, f"Pos ({i},{j},{k}): {val}"
    
    if full_check:
        # Check all off-diagonal positions
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m; j = a % m
                    j2 = b // p; k = b % p
                    k2 = c // n; i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        continue
                    val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    if abs(val) > 1e-9:
                        return False, f"Off ({a},{b},{c}): {val}"
    else:
        # Sample off-diagonal
        np.random.seed(42)
        for _ in range(1000):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i = a // m; j = a % m
            j2 = b // p; k = b % p
            k2 = c // n; i2 = c % n
            if j == j2 and k == k2 and i == i2:
                continue
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val) > 1e-9:
                return False, f"Off ({a},{b},{c}): {val}"
    
    return True, "OK"

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns (U, V, W) where each is a 4x7 matrix.
    """
    # Strassen 2x2x2 rank-7
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # Flatten: A=[a,b,c,d], B=[e,f,g,h], C=[i,j,k,l]
    # Positions: A(i,j) -> i*2+j, B(j,k) -> j*2+k, C(k,i) -> k*2+i
    
    # The 7 Strassen products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # i = M1+M4-M5+M7
    # j = M3+M5
    # k = M2+M4
    # l = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1  # a+d
    V[0, 0] = 1; V[3, 0] = 1  # e+h
    W[0, 0] = 1; W[3, 0] = 1  # contributes to i and l
    
    # M2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1  # c+d
    V[0, 1] = 1               # e
    W[2, 1] = 1               # k
    
    # M3: a(f-h)
    U[0, 2] = 1               # a
    V[1, 2] = 1; V[3, 2] = -1 # f-h
    W[1, 2] = 1; W[3, 2] = 1  # j and l
    
    # M4: d(g-e)
    U[3, 3] = 1               # d
    V[2, 3] = 1; V[0, 3] = -1 # g-e
    W[0, 3] = 1; W[2, 3] = 1  # i and k
    
    # M5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1  # a+b
    V[3, 4] = 1               # h
    W[0, 4] = -1; W[1, 4] = 1 # -i + j
    
    # M6: (c-a)(e+f)
    U[0, 5] = -1; U[2, 5] = 1  # c-a
    V[0, 5] = 1; V[1, 5] = 1   # e+f
    W[3, 5] = 1               # l
    
    # M7: (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1  # b-d
    V[2, 6] = 1; V[3, 6] = 1   # g+h
    W[0, 6] = 1               # i
    
    return U, V, W

def _verify_strassen_2x2():
    """Verify the Strassen 2x2 decomposition."""
    U, V, W = _strassen_2x2_decomposition()
    # Check: for 2x2x2, positions are (i,j,k) -> A[i*2+j], B[j*2+k], C[k*2+i]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + j
                b = j * 2 + k
                c = k * 2 + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1.0) > 1e-9:
                    return False, f"({i},{j},{k}): {val}"
    # Check off-diagonal
    for a in range(4):
        for b in range(4):
            for c in range(4):
                i = a // 2; j = a % 2
                j2 = b // 2; k = b % 2
                k2 = c // 2; i2 = c % 2
                if j == j2 and k == k2 and i == i2:
                    continue
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val) > 1e-9:
                    return False, f"Off ({a},{b},{c}): {val}"
    return True, "OK"

def _construct_4x4x4_rank49():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication.
    
    Approach: Use the 2x2 block structure with Strassen.
    
    4x4 matrices as 2x2 blocks of 2x2:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each 2x2*2x2 product uses Strassen (rank 7).
    Each C block is a sum of two 2x2 products, requiring 2*7 = 14 Strassen terms.
    4 C blocks * 14 = 56 terms.
    
    To get to rank 49, we need to share 7 terms between blocks.
    The Larsen construction identifies shared products.
    
    For a verifiable implementation, let me use the 56-term construction
    (which is still better than schoolbook 64) and verify it.
    Actually 56 > 49, so let me try to find the sharing.
    
    Key sharing: A00 appears in C00 and C01. The Strassen products
    involving A00 can potentially be shared.
    
    For C00: A00*B00 uses Strassen products M1-M7 with A00 as the "A" matrix.
    For C01: A00*B01 uses Strassen products with A00 as "A" and B01 as "B".
    
    These are different products (different B matrices), so they can't be
    directly shared. The sharing in Larsen's construction is more subtle.
    
    For reliability, let me implement the 56-term version and verify it.
    56 < 64 (schoolbook), so it's an improvement.
    """
    
    # 4x4 matrix block positions:
    # A00: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    
    # B00: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # B01: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # B10: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # B11: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    
    # C00: rows 0-1, cols 0-1 -> flat indices 0,1,4,5
    # C01: rows 0-1, cols 2-3 -> flat indices 2,3,6,7
    # C10: rows 2-3, cols 0-1 -> flat indices 8,9,12,13
    # C11: rows 2-3, cols 2-3 -> flat indices 10,11,14,15
    
    # In the 4x4x4 tensor:
    # A position (i,j) -> flat index i*4+j
    # B position (j,k) -> flat index j*4+k
    # C position (k,i) -> flat index k*4+i
    
    # For 2x2 block A00 (rows 0-1, cols 0-1):
    #   A00[0,0] = A[0,0] -> flat 0
    #   A00[0,1] = A[0,1] -> flat 1
    #   A00[1,0] = A[1,0] -> flat 4
    #   A00[1,1] = A[1,1] -> flat 5
    
    # For B00 (rows 0-1, cols 0-1):
    #   B00[0,0] = B[0,0] -> flat 0
    #   B00[0,1] = B[0,1] -> flat 1
    #   B00[1,0] = B[1,0] -> flat 4
    #   B00[1,1] = B[1,1] -> flat 5
    
    # For C00 (rows 0-1, cols 0-1):
    #   C00[0,0] = C[0,0] -> flat 0
    #   C00[0,1] = C[0,1] -> flat 1
    #   C00[1,0] = C[1,0] -> flat 4
    #   C00[1,1] = C[1,1] -> flat 5
    
    # The Strassen decomposition for A00*B00 -> C00:
    # In the 2x2 space, A00 flat = [0,1,4,5] (relative to 4x4 flat)
    # B00 flat = [0,1,4,5]
    # C00 flat = [0,1,4,5]
    
    # Let me define the mapping from 2x2 local index to 4x4 global index
    def block_map(block_row, block_col):
        """Map 2x2 local index to 4x4 global flat index."""
        # Local index l = lr*2 + lc
        # Global row = block_row*2 + lr
        # Global col = block_col*2 + lc
        # Global flat = global_row*4 + global_col
        def local_to_global(l):
            lr = l // 2
            lc = l % 2
            gr = block_row * 2 + lr
            gc = block_col * 2 + lc
            return gr * 4 + gc
        return local_to_global
    
    # Get the Strassen 2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2_decomposition()
    
    # Now build the 4x4x4 decomposition
    # For each C block, we need two Strassen decompositions (for the two products)
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Total: 8 Strassen decompositions, each with 7 terms = 56 terms
    
    n4 = 4
    total_terms = 56
    U = np.zeros((n4 * n4, total_terms))
    V = np.zeros((n4 * n4, total_terms))
    W = np.zeros((n4 * n4, total_terms))
    
    term_idx = 0
    
    # Define the 8 products: (A_block, B_block, C_block)
    products = [
        # C00 = A00*B00 + A01*B10
        (0, 0, 0, 0),  # A00*B00 -> C00
        (0, 1, 1, 0),  # A01*B10 -> C00
        # C01 = A00*B01 + A01*B11
        (0, 0, 0, 1),  # A00*B01 -> C01
        (0, 1, 1, 1),  # A01*B11 -> C01
        # C10 = A10*B00 + A11*B10
        (1, 0, 0, 0),  # A10*B00 -> C10
        (1, 1, 1, 0),  # A11*B10 -> C10
        # C11 = A10*B01 + A11*B11
        (1, 0, 0, 1),  # A10*B01 -> C11
        (1, 1, 1, 1),  # A11*B11 -> C11
    ]
    # Each entry: (A_block_row, A_block_col, B_block_row, B_block_col, C_block_row, C_block_col)
    # Wait, I need 6 values. Let me redo:
    products = [
        (0, 0, 0, 0, 0, 0),  # A00*B00 -> C00
        (0, 1, 1, 0, 0, 0),  # A01*B10 -> C00
        (0, 0, 0, 1, 0, 1),  # A00*B01 -> C01
        (0, 1, 1, 1, 0, 1),  # A01*B11 -> C01
        (1, 0, 0, 0, 1, 0),  # A10*B00 -> C10
        (1, 1, 1, 0, 1, 0),  # A11*B10 -> C10
        (1, 0, 0, 1, 1, 1),  # A10*B01 -> C11
        (1, 1, 1, 1, 1, 1),  # A11*B11 -> C11
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        # Get the index mappings
        a_map = block_map(ar, ac)
        b_map = block_map(br, bc)
        c_map = block_map(cr, cc)
        
        for t in range(7):
            # Fill U, V, W for this Strassen term
            for local_idx in range(4):
                U[a_map(local_idx), term_idx] += S_U[local_idx, t]
                V[b_map(local_idx), term_idx] += S_V[local_idx, t]
                W[c_map(local_idx), term_idx] += S_W[local_idx, t]
            term_idx += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Verify Strassen 2x2 works
    valid_s, msg_s = _verify_strassen_2x2()
    if not valid_s:
        print(f"Strassen 2x2 verification failed: {msg_s}")
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_list = [[int(round(x)) for x in row] for row in U]
        V_list = [[int(round(x)) for x in row] for row in V]
        W_list = [[int(round(x)) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Construct 4x4x4 decomposition (rank 56)
    U4, V4, W4 = _construct_4x4x4_rank49()
    rank4 = U4.shape[1]
    
    # Verify 4x4x4 decomposition
    valid4, msg4 = _verify_decomposition(U4, V4, W4, 4, 4, 4, full_check=True)
    if not valid4:
        print(f"4x4x4 verification failed: {msg4}")
        # Fall back to schoolbook for 4x4x4
        rank4 = 64
        U4 = np.zeros((16, rank4))
        V4 = np.zeros((16, rank4))
        W4 = np.zeros((16, rank4))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U4[i * 4 + j, col] = 1
                    V4[j * 4 + k, col] = 1
                    W4[k * 4 + i, col] = 1
                    col += 1
    
    # Build full 4x4x4x8 decomposition
    rank_total = 2 * rank4
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Left half: B columns 0-3
    for col in range(rank4):
        U[:, col] = U4[:, col]
        for j in range(m):
            for k in range(4):
                V[j * p + k, col] = V4[j * 4 + k, col]
        for k in range(4):
            for i in range(n):
                W[k * n + i, col] = W4[k * 4 + i, col]
    
    # Right half: B columns 4-7
    for col in range(rank4, 2 * rank4):
        U[:, col] = U4[:, col - rank4]
        for j in range(m):
            for k in range(4):
                V[j * p + (4 + k), col] = V4[j * 4 + k, col - rank4]
        for k in range(4):
            for i in range(n):
                W[k * n + i, col] = W4[k * 4 + i, col - rank4]
    
    # Verify full decomposition
    valid, msg = _verify_decomposition(U, V, W, n, m, p, full_check=False)
    if not valid:
        print(f"Full verification failed: {msg}")
        # Fall back to direct schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists of integers
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
