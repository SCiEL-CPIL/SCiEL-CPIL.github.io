import numpy as np
import time

DIMENSIONS = [3, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the substitution method: embed 3x3x6 into 4x4x4, use rank-49 decomposition
    try:
        U, V, W = build_rank_49_substitution(n, m, p)
        if time.time() - start_time < DEADLINE and verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 54
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

def build_rank_49_substitution(n, m, p):
    """
    Embed 3x3 x 3x6 into 4x4 x 4x4 using the substitution method.
    
    The idea: A 3x3 matrix times a 3x6 matrix can be viewed as:
    - Split the 3x6 into two 3x3 blocks: C = A * [B1 | B2]
    - This is equivalent to computing two 3x3 multiplications in parallel
    
    We embed this into a 4x4x4 problem by:
    - Creating a 4x4 matrix M_A from A (3x3) by adding a row/column
    - Creating a 4x4 matrix M_B from the 3x6 (viewed as 3x3x2) 
    - Using the rank-49 decomposition of 4x4x4
    - Restricting to the subspace where the extra dimensions are zero
    
    For 4x4x4, we use a known rank-49 decomposition.
    The standard approach uses Strassen-like recursion.
    
    Actually, let me use a simpler approach: use the known fact that
    3x3x3 has rank 19 (or 23), and 3x3x6 can be done with rank 38
    by doing two 3x3x3 multiplications.
    
    But we want rank < 54. Let me try rank 49 via embedding in 4x4x4.
    
    For 4x4x4, the optimal rank is 49. We can construct it using
    Strassen's algorithm recursively.
    
    Let me implement the rank-49 decomposition for 4x4x4 directly.
    """
    
    # The rank-49 decomposition of 4x4x4 matrix multiplication
    # This is based on the known result that R(4x4x4) = 49
    
    # We'll use a construction based on Strassen's 2x2x2 (rank 7)
    # applied hierarchically.
    
    # For 4x4, we can split into 2x2 blocks and use Strassen.
    # 4x4 = 2*(2x2), so we need 7 multiplications for each level.
    # Total: 7 * 7 = 49 for two levels of Strassen.
    
    # Let me construct this explicitly.
    
    # Level 1: 4x4 split into 2x2 blocks
    # Each 2x2 block multiplication uses Strassen (rank 7)
    # We have 4 block positions, but Strassen combines them.
    
    # Actually, the standard Strassen for 4x4:
    # Split A, B, C into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Each Cij is 2x2, computed via Strassen on 2x2 blocks.
    # But that gives 4 * 7 = 28, not 49.
    #
    # Wait, I think I'm confusing things. Let me reconsider.
    #
    # For 4x4x4, the rank is 49. One construction:
    # Use the fact that 4x4 can be decomposed using 7 "super-multiplications"
    # where each super-multiplication is itself a 2x2x2 (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let me implement this. The 7 Strassen multipliers for 4x4 are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Where Aij, Bij are 2x2 blocks.
    
    # Each Mj is a 2x2 matrix, and we compute it via Strassen (rank 7).
    
    # So for each of the 7 Mj's, we need 7 scalar multiplications.
    # Total: 7 * 7 = 49.
    
    # Now, to embed 3x3x6 into this:
    # A is 3x3, we pad to 4x4 with zeros: A4 = [[A, 0], [0, 0]]
    # The 3x6 matrix B, we view as 3x3 x 2 (two 3x3 blocks side by side)
    # Pad each to 4x4: B1_4 = [[B1, 0], [0, 0]], B2_4 = [[B2, 0], [0, 0]]
    
    # But wait, the output is 3x6, which is two 3x3 blocks.
    # So we need to compute C1 = A * B1 and C2 = A * B2, both 3x3.
    
    # Using the 4x4x4 decomposition:
    # C4 = A4 * B4, where B4 is 4x4.
    # But we have two B matrices, so we need to be clever.
    
    # Alternative: pack B1 and B2 into a single 4x4 matrix?
    # B4 = [[B1, B2], [0, 0]]? No, that doesn't work dimensionally.
    
    # Let me think differently. The 3x6 matrix B can be seen as a 3x3 matrix
    # over a "2-dimensional" field, or we can interleave.
    
    # Actually, the simplest approach: just do two separate 3x3x3 multiplications.
    # If 3x3x3 has rank 19, then 3x3x6 has rank at most 38.
    # But 38 < 49, so this is better!
    
    # Let me use the known rank-19 decomposition of 3x3x3.
    
    # The rank-19 decomposition of 3x3x3 is known (Landsberg, 1998).
    # Let me try to recall or reconstruct it.
    
    # Actually, let me try a different approach. Use Strassen's algorithm
    # to get a rank-22 decomposition of 3x3x3, then use 22*2 = 44 for 3x3x6.
    
    # Strassen for 3x3: Split into 2x2 and 1x1 blocks? That's asymmetric.
    
    # Better: Use the known fact that 3x3x3 can be done with rank 22
    # using a specific construction.
    
    # Let me implement a rank-22 decomposition of 3x3x3.
    
    # One known construction uses 22 terms. Let me try to find it.
    
    # Actually, I'll try a different strategy: use the rank-49 4x4x4 decomposition
    # and restrict it properly.
    
    # Embedding: 
    # A (3x3) -> A4 (4x4) by adding row 4 = [0,0,0,0], col 4 = [0,0,0,0]^T
    # B (3x6) -> We need to map this to 4x4.
    
    # Idea: View B as a 3x3 matrix where each entry is a 2-vector.
    # Then B4 (4x4) has entries that are 2-vectors? No, that doesn't work.
    
    # Let me try: B4 = [[B1, 0], [0, 0]] where B1 is the first 3x3 of B.
    # Then C4 = A4 * B4 gives C1 = A * B1 in the top-left 3x3.
    # Similarly for B2.
    
    # So we do two 4x4x4 multiplications, each with rank 49.
    # Total rank: 98. That's worse than 54!
    
    # So the embedding approach doesn't directly help unless we can
    # combine the two multiplications.
    
    # Let me try yet another approach: use the rank-19 3x3x3 decomposition.
    
    # I'll implement a known rank-19 decomposition of 3x3x3.
    
    # The decomposition from Landsberg (1998) "Tensor rank and arithmetical
    # complexity of bilinear mappings" gives rank 19 for 3x3x3.
    
    # Let me try to reconstruct it or use a simpler rank-22 construction.
    
    # For now, let me try to implement a rank-22 3x3x3 decomposition
    # using a modified Strassen approach.
    
    # Strassen for 3x3: 
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # We can split as:
    # A = [[A11, a13], [A21, a23], [a31, a32, a33]]
    # where A11, A21 are 2x1, and the bottom-right is 1x1.
    
    # This is getting complicated. Let me just use the schoolbook 54
    # and return it, since I can't reliably implement a lower-rank decomposition
    # without risking incorrectness.
    
    # Actually, let me try one more thing: use the fact that we can do
    # 3x3x6 by doing one 3x3x3 (rank 22) and one 3x3x3 (rank 22),
    # for total rank 44.
    
    # Let me implement a rank-22 3x3x3 decomposition.
    
    # I'll use a construction based on 2x2 Strassen + corrections.
    
    # Split 3x3 into:
    # Top-left 2x2, top-right 2x1, bottom-left 1x2, bottom-right 1x1
    #
    # A = [[A22, a_r], [b_l, a_br]] where A22 is 2x2, a_r is 2x1, 
    #     b_l is 1x2, a_br is scalar
    #
    # B = [[B22, b_r], [c_l, b_br]]
    #
    # C = A * B = [[A22*B22 + a_r*c_l, A22*b_r + a_r*b_br],
    #              [b_l*B22 + a_br*c_l, b_l*b_r + a_br*b_br]]
    #
    # Using Strassen for the 2x2*2x2 part (rank 7), plus corrections.
    #
    # This is getting too complex to implement correctly without reference.
    
    # Let me just return the schoolbook 54 to be safe.
    raise Exception("No better decomposition found")

def verify(U, V, W, n, m, p):
    rank = len(U[0])
    # Build tensor
    tensor = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        uc = np.array(U[:, c])
        vc = np.array(V[:, c])
        wc = np.array(W[:, c])
        tensor += np.einsum('i,j,k->ijk', uc, vc, wc)
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                target[ui, vj, wk] = 1.0
    
    return np.allclose(tensor, target)
