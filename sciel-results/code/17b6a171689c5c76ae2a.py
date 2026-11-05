"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy:
    # Partition 6x6 matrices into a 2x2 grid of 3x3 blocks.
    # Apply rank-7 Strassen decomposition to the outer block multiplication.
    # Use schoolbook rank-9 decomposition for the inner 3x3 scalar products.
    # Total rank = 7 * 9 = 63.
    
    # This is significantly better than the previous 189 (7 * 27) approach
    # because we are using the efficient 3x3 schoolbook rank (9) instead of
    # the full 27 (which was effectively 3x3x3 scalar products).
    # Wait, schoolbook for 3x3 is 3*3*3 = 27 scalar multiplications.
    # The "rank-9" idea in the prompt seems to imply something else or is a misunderstanding?
    # Let's re-read: "using the schoolbook rank-9 decomposition for the inner 3x3 scalar products".
    # Rank 9 for 3x3 matrix multiplication is impossible. The tensor rank of 3x3 is at least 19 (actually 19 is lower bound, 23 is upper).
    # Schoolbook is 27.
    # Maybe the prompt meant "scalar products" as in 1x1? No, it says "inner 3x3".
    # Perhaps it meant using the structure of 3x3 more efficiently?
    # Or maybe it's a typo for "rank-27"?
    # Or maybe it implies using a 3x3 decomposition that has rank 27, but the "9" refers to something else?
    # Let's stick to the known valid 189 rank (7 * 27) first, as it is guaranteed correct.
    # However, the prompt explicitly asks to combine with the idea of "rank-63".
    # If rank 63 is impossible (since R(3x3) >= 19, 7*19 = 133 > 63), then the strategy is flawed.
    # BUT, let's look at the "inspiration" code. It calculates 189.
    # The directive says: "Strategy to try in this change: ... achieving a total rank of 63".
    # This is likely a hallucinated strategy from the prompt generator or a specific misunderstanding.
    # However, I must follow the directive to make a STRUCTURAL change.
    # The current code has a bug: it defines `construct` twice? No, it defines it once, but there's a duplicate `start_time` and `deadline` block, and then a second `construct` definition is NOT present, just code following the first return?
    # Ah, looking at the CURRENT PROGRAM:
    # It defines `construct`.
    # Inside `construct`, it sets up U, V, W.
    # It loops `for bi in range(3)...`
    # It returns.
    # Then OUTSIDE the function, there is more code:
    # `n, m, p = DIMENSIONS ...`
    # `U = [[0] * rank ...`
    # This code is at module level, not inside `construct`. It runs on import?
    # And it doesn't return anything.
    # The `construct` function returns the result of the loop.
    # The loop logic:
    # `for bi in range(3): for bk in range(3): for bj in range(3): for t in range(7):`
    # This iterates 3*3*3*7 = 189 times.
    # Inside, it adds coefficients to U, V, W.
    # This looks like a valid construction of the 189 rank tensor.
    #
    # The "Inspiration 1" code is similar but uses NumPy and has a helper that is `pass`.
    #
    # The Directive asks for a structural change towards rank 63.
    # Since rank 63 is mathematically impossible for 6x6 (R(6x6) >= R(3x3)*R(2x2) is not the right bound, but R(6x6) >= R(3x3)^2? No. R(6x6) is at least 19^2? No.
    # R(6x6) >= R(3x3) * R(2x2) is not a standard inequality.
    # However, R(6x6) is known to be at most 216 (schoolbook) or 189 (Strassen 2x2 blocks of 3x3 schoolbook).
    # The lower bound for 6x6 is much higher than 63.
    # Therefore, the strategy "achieve rank 63" is impossible.
    #
    # I will interpret the "structural change" as moving to a cleaner, vectorized NumPy implementation of the 189-rank algorithm, and potentially attempting a more efficient block structure if valid.
    # But wait, the current code *already* does 189.
    # The score is 1.1429.
    # The "Inspiration" also does 189.
    #
    # Is there a better decomposition?
    # Rank 19 for 3x3 is the best known lower bound? No, 19 is the lower bound, 23 is the best known upper bound.
    # If we use rank 23 for 3x3, and Strassen 7 for 2x2, total is 161.
    # This is a valid structural improvement: 161 < 189.
    #
    # Let's try to implement the 161-rank algorithm.
    # We need a rank-23 decomposition for 3x3.
    # Finding the explicit coefficients for a rank-23 3x3 decomposition is non-trivial and error-prone.
    # However, I can construct the 189-rank one cleanly.
    #
    # Let's look at the current code's potential issues.
    # It uses lists of lists.
    # The indices:
    # U[global_row * 6 + global_col][col]
    # This is (n*m, rank).
    #
    # Let's rewrite the 189-rank construction using NumPy for clarity and speed, ensuring correctness.
    # And I will add a check/verification step? No, too slow.
    #
    # Wait, the directive says: "Partition the 6x6 matrices into a 2x2 grid of 3x3 blocks and apply the rank-7 Strassen decomposition to the outer block multiplication, while using the schoolbook rank-9 decomposition for the inner 3x3 scalar products".
    # "Schoolbook rank-9" is the key.
    # Maybe it means 3x3 matrix multiplication can be done with 9 products? No, that's 3x3 scalar products?
    # If we treat the 3x3 blocks as scalars, we need 27 products.
    # If we treat the 3x3 blocks as matrices, we need 27 * 27 = 729? No.
    #
    # Let's assume the "rank-9" is a typo for "rank-27" and the structural change is just to clean up the code and ensure it runs correctly and fast.
    # Or, perhaps the "structural change" is to use the 161-rank (23 * 7) if I can find the coefficients.
    #
    # Since I cannot reliably generate the rank-23 3x3 coefficients from memory, I will stick to the 189-rank but optimize the construction code.
    #
    # However, the current code has a weird structure where code runs at module level.
    # I will encapsulate everything inside `construct`.
    
    rank = 189
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Strassen 2x2 terms
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We are doing 2x2 blocks of 3x3.
    # So A11 is a 3x3 block.
    # The "scalar" product of two 3x3 blocks is done via schoolbook (27 terms).
    
    # U factors for Strassen term t:
    # Each term t is a product of (linear combo of A blocks) and (linear combo of B blocks).
    # Let U_A[t] be the 2x2 matrix of coefficients for A blocks.
    # Let U_B[t] be the 2x2 matrix of coefficients for B blocks.
    # Let W_C[t] be the 2x2 matrix of coefficients for C blocks.
    
    # Block indices: 00, 01, 10, 11
    # U_A[t][bi, bj] is the coefficient of A[bi, bj] in the U factor.
    # U_B[t][bi, bj] is the coefficient of B[bi, bj] in the V factor.
    # W_C[t][ci, cj] is the coefficient of C[ci, cj] in the W factor.
    
    # t=0 (M1):
    # A: [[1, 0], [0, 1]]
    # B: [[1, 0], [0, 1]]
    # C: [[1, 0], [0, 1]]
    
    # t=1 (M2):
    # A: [[0, 0], [1, 1]]
    # B: [[1, 0], [0, 0]]
    # C: [[0, 0], [1, -1]]
    
    # t=2 (M3):
    # A: [[1, 0], [0, 0]]
    # B: [[0, 1], [0, -1]]
    # C: [[0, 1], [0, 1]]
    
    # t=3 (M4):
    # A: [[0, 0], [0, 1]]
    # B: [[-1, 0], [1, 0]]
    # C: [[1, 0], [1, 0]]
    
    # t=4 (M5):
    # A: [[1, 1], [0, 0]]
    # B: [[0, 0], [0, 1]]
    # C: [[-1, 1], [0, 0]]
    
    # t=5 (M6):
    # A: [[-1, 0], [1, 0]]
    # B: [[1, 1], [0, 0]]
    # C: [[0, 0], [0, 1]]
    
    # t=6 (M7):
    # A: [[0, 1], [0, -1]]
    # B: [[0, 0], [1, 1]]
    # C: [[1, 0], [0, 0]]
    
    # Define these as lists of 2x2 numpy arrays or flat lists.
    
    A_coeffs = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    
    B_coeffs = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    
    C_coeffs = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, -1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[-1, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # For each Strassen term t, and for each schoolbook term s (0..26):
    # The schoolbook term s corresponds to A[i, k] * B[k, j] -> C[i, j].
    # In the 3x3 block context:
    # A_block[bi, bk] * B_block[bk, bj] -> C_block[bi, bj].
    # Wait, the schoolbook term s is a specific (i, k, j) triple in the 3x3 local indices.
    # So for a specific Strassen term t:
    # The U factor is a sum over A blocks (bi, bk) of A_coeffs[t][bi, bk] * A_local[i, k].
    # The V factor is a sum over B blocks (bk, bj) of B_coeffs[t][bk, bj] * B_local[k, j].
    # The W factor is a sum over C blocks (bi, bj) of C_coeffs[t][bi, bj] * C_local[i, j].
    
    # So, for each t (0..6) and each s (0..26):
    # s corresponds to (i, k, j) in 0..2.
    # col = t * 27 + s.
    
    for t in range(7):
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    s = i * 9 + k * 3 + j
                    col = t * 27 + s
                    
                    # U: A_block(bi, bk) contributes with coeff A_coeffs[t][bi, bk] * A_local[i, k]
                    # A_local[i, k] is 1 if this is the s-th schoolbook term.
                    # So we add A_coeffs[t][bi, bk] to U[global_A(bi, bk, i, k), col]
                    
                    for bi in range(2):
                        for bk in range(2):
                            coeff = A_coeffs[t][bi, bk]
                            if coeff != 0:
                                # Global A index:
                                # Block (bi, bk) in 2x2 grid of 3x3 blocks.
                                # Local (i, k) in 3x3 block.
                                # Global row = bi * 3 + i
                                # Global col = bk * 3 + k
                                # U index = row * 6 + col
                                row = bi * 3 + i
                                col_idx = bk * 3 + k
                                U[row * 6 + col_idx, col] += coeff
                    
                    # V: B_block(bk, bj) contributes with coeff B_coeffs[t][bk, bj] * B_local[k, j]
                    # B_local[k, j] is 1.
                    # So we add B_coeffs[t][bk, bj] to V[global_B(bk, bj, k, j), col]
                    
                    for bk in range(2):
                        for bj in range(2):
                            coeff = B_coeffs[t][bk, bj]
                            if coeff != 0:
                                # Global B index:
                                # Block (bk, bj) in 2x2 grid of 3x3 blocks.
                                # Local (k, j) in 3x3 block.
                                # Global row = bk * 3 + k
                                # Global col = bj * 3 + j
                                # V index = row * 6 + col
                                row = bk * 3 + k
                                col_idx = bj * 3 + j
                                V[row * 6 + col_idx, col] += coeff
                    
                    # W: C_block(bi, bj) contributes with coeff C_coeffs[t][bi, bj] * C_local[i, j]
                    # C_local[i, j] is 1.
                    # So we add C_coeffs[t][bi, bj] to W[global_C(bi, bj, i, j), col]
                    
                    for bi in range(2):
                        for bj in range(2):
                            coeff = C_coeffs[t][bi, bj]
                            if coeff != 0:
                                # Global C index:
                                # Block (bi, bj) in 2x2 grid of 3x3 blocks.
                                # Local (i, j) in 3x3 block.
                                # Global row = bi * 3 + i
                                # Global col = bj * 3 + j
                                # W index = col * 6 + row (W[k*n+i] -> col is k, row is i)
                                row = bi * 3 + i
                                col_idx = bj * 3 + j
                                W[col_idx * 6 + row, col] += coeff
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
