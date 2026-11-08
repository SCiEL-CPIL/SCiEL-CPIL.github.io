"""
Kronecker Block Decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
  1. View A (3x3) as a 3x1 block matrix of 1x3 row vectors: A = [a0; a1; a2], ai in F^3.
  2. View B (3x7) as a 3x3 block matrix of 1x3 column vectors: B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]], b_ij in F^3.
  3. The product C = A * B is a 3x3 block matrix of 1x3 row vectors: C = [c0; c1; c2], where c_i = sum_j a_i * b_ji (dot product of 1x3 vectors).
  4. This is equivalent to multiplying a 3x3 "super-matrix" A' (entries are 1x3 vectors) by a 3x3 "super-matrix" B' (entries are 1x3 vectors) to get a 3x3 "super-matrix" C' (entries are 1x3 vectors).
  5. However, the entries are vectors, not scalars. The multiplication is: C'_{ij} = sum_k A'_{ik} * B'_{kj}, where * is the dot product.
  6. This is not standard matrix multiplication of scalars. But we can embed each 1x3 vector into a 2x2 matrix via a linear isomorphism? No, that's overcomplicating.
  
  Let's rethink:
  The tensor is T_{i,j,k} = 1 if i,j,k are valid indices.
  We want to write T = sum_r U_r * V_r * W_r.
  
  Alternative Kronecker approach:
  The 3x3 * 3x7 multiplication can be seen as:
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  Let's partition the indices:
  i in {0,1,2}, j in {0,1,2}, k in {0,..,6}
  
  We can use the fact that 3 = 2 + 1.
  Split A into A_top (2x3) and A_bottom (1x3).
  Split B into B_left (3x6) and B_right (3x1).
  
  C_top_left = A_top * B_left (2x6)
  C_top_right = A_top * B_right (2x1)
  C_bottom_left = A_bottom * B_left (1x6)
  C_bottom_right = A_bottom * B_right (1x1)
  
  This is just schoolbook.
  
  Let's try the "Strassen on 2x2 block" approach more carefully.
  The known best rank for 3x3 * 3x7 is likely 61 or 62.
  Schoolbook is 63.
  
  The previous run found a rank 60 solution using a hybrid of Strassen 2x2x2 and schoolbook.
  The score for rank 60 is 63/60 = 1.05.
  
  Let's implement the hybrid strategy from the lesson:
  "Replacing the schoolbook 2x2x3 block (12 terms) with Strassen 2x2x2 (7 terms) + schoolbook 2x2x1 (4 terms) = 11 terms, saving 1 rank from 61 to 60"
  
  Wait, 63 - 1 = 62? No, 63 - 12 + 11 = 62?
  Let's re-read: "saving 1 rank from 61 to 60". This implies the base was 61? No, schoolbook is 63.
  Maybe the base was a 61-rank solution?
  
  Actually, let's just implement the best known heuristic:
  1. Use Strassen for the top-left 2x2 * 2x2 block of the multiplication.
  2. Use schoolbook for the rest.
  
  Top-left 2x2 * 2x2 block:
  A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
  This contributes to C[0,0], C[0,1], C[1,0], C[1,1].
  
  The terms involving A[0:2, 0:2] and B[0:2, 0:2] are:
  C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
  C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
  C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
  C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
  
  In schoolbook, these are 4 terms.
  In Strassen, these are 7 terms. This is worse!
  
  Why would Strassen help?
  Strassen helps when you are multiplying large matrices and the overhead is amortized.
  Here, the "block" is small.
  
  Let's look at the dimensions again.
  3x3 * 3x7.
  
  Maybe we can decompose the 3x7 matrix B into B1 (3x3) + B2 (3x4)? No, that doesn't make sense.
  
  Let's try a different partition:
  A is 3x3. B is 3x7.
  Let's split B into B1 (3x4) and B2 (3x3).
  C = A * B1 + A * B2.
  
  A * B1 is 3x4.
  A * B2 is 3x3.
  
  We can compute A * B2 using a rank-19 decomposition for 3x3 * 3x3 (if we can find one, but schoolbook is 27, Strassen-based might be 21 or 22).
  And A * B1 using schoolbook (3*3*4 = 36).
  Total: 36 + 19 = 55? If rank 19 is achievable.
  
  But we don't have a rank-19 3x3*3x3 decomposition.
  The best known for 3x3*3x3 is 23 (Lader).
  
  So 36 + 23 = 59. This is better than 63!
  
  Let's try to implement a rank-23 decomposition for 3x3 * 3x3.
  This is complex.
  
  Alternative:
  Use the fact that 3x3 * 3x7 is isomorphic to 3x3 * 3x3 + 3x3 * 3x4.
  
  Let's try a simpler improvement:
  The current program returns schoolbook (63).
  The lesson says a rank 60 solution was found.
  
  How was rank 60 found?
  "Replacing the schoolbook 2x2x3 block (12 terms) with Strassen 2x2x2 (7 terms) + schoolbook 2x2x1 (4 terms) = 11 terms"
  
  This implies there is a 2x2x3 block in the multiplication.
  2x2x3 block: A[0:2, 0:2] * B[0:2, 0:3]?
  This gives C[0:2, 0:3].
  Terms: 2*2*3 = 12.
  
  If we use Strassen for the 2x2 * 2x2 part (cols 0:2 of B), that's 7 terms.
  And schoolbook for the 2x2 * 2x1 part (col 2 of B), that's 4 terms.
  Total: 11 terms.
  Saving: 1 term.
  
  So the total rank becomes 63 - 1 = 62?
  But the lesson says "saving 1 rank from 61 to 60".
  This suggests the base was 61, not 63.
  
  Maybe there is another saving?
  
  Let's just implement the 62-rank solution first, and see if we can find more savings.
  
  Actually, let's try to implement a rank-60 solution directly if possible.
  
  Another idea:
  Use a rank-12 decomposition for 2x2 * 2x3?
  Schoolbook is 12. Strassen+Schoolbook is 11.
  
  Is there a rank-11 decomposition for 2x2 * 2x3?
  If yes, then we save 1 more term.
  
  Let's assume the best we can do quickly is 62.
  
  But wait, the score is 1.0 for rank 63.
  Rank 62 gives 63/62 = 1.016.
  Rank 60 gives 63/60 = 1.05.
  
  Let's try to implement the 62-rank solution.
  
  Steps:
  1. Initialize U, V, W as zeros of shape (9, 62), (21, 62), (21, 62).
  2. Fill in the Strassen terms for the 2x2 * 2x2 block (A[0:2, 0:2] * B[0:2, 0:2]).
  3. Fill in the schoolbook terms for the 2x2 * 2x1 block (A[0:2, 0:2] * B[0:2, 2:3]).
  4. Fill in the schoolbook terms for the rest of the multiplication.
  
  The "rest" is:
  - A[0:2, 0:2] * B[0:2, 3:7] (2x2 * 2x4 = 16 terms)
  - A[0:2, 2:3] * B[2:3, 0:7] (2x1 * 1x7 = 14 terms)
  - A[2:3, 0:3] * B[0:3, 0:7] (1x3 * 3x7 = 21 terms)
  
  Total: 7 + 4 + 16 + 14 + 21 = 62.
  
  Let's implement this.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
DEADLINE_SECONDS = 115

def _verify(U, V, W):
    """Check that U, V, W exactly decompose the 3x3 * 3x7 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2 * 2x2 multiplication.
    Each term is a tuple (U_row, V_row, W_row, coeff)
    where U_row is a tuple of (index, coeff) for the U matrix, etc.
    """
    # M1 = (a+d)(e+h)
    # U: a(0), d(2); V: e(0), h(1); W: C00(0), C11(3)
    # M2 = (c+d)e
    # U: c(2), d(3); V: e(0); W: C01(2), C11(3)
    # M3 = a(f-h)
    # U: a(0); V: f(1), -h(3); W: C00(0), -C10(2)
    # M4 = d(g-e)
    # U: d(3); V: g(2), -e(0); W: -C00(0), C10(2)
    # M5 = (a+b)h
    # U: a(0), b(1); V: h(3); W: C01(1), C11(3)
    # M6 = (c-a)(e+f)
    # U: c(2), -a(0); V: e(0), f(1); W: C00(0)
    # M7 = (b-d)(g+h)
    # U: b(1), -d(3); V: g(2), h(3); W: -C11(3)
    
    # We need to map these to the global indices.
    # A[0:2, 0:2] -> indices 0,1,2,3 in U (i*m+j)
    # B[0:2, 0:2] -> indices 0,1,2,3 in V (j*p+k)
    # C[0:2, 0:2] -> indices 0,1,2,3 in W (k*n+i)
    
    # Let's define the terms explicitly.
    # U index: i*3 + j
    # V index: j*7 + k
    # W index: k*3 + i
    
    terms = []
    
    # M1: (A00+A11)(B00+B11)
    # U: (0,1), (3,1)  [A00, A11]
    # V: (0,1), (7,1)  [B00, B11]
    # W: (0,1), (12,1) [C00, C11]
    terms.append(((0,1), (3,1), (0,1), (7,1), (0,1), (12,1)))
    
    # M2: (A10+A11)B00
    # U: (2,1), (3,1)  [A10, A11]
    # V: (0,1)         [B00]
    # W: (1,1), (12,1) [C01, C11]
    terms.append(((2,1), (3,1), (0,1), (1,1), (12,1)))
    
    # M3: A00(B10-B00)
    # U: (0,1)         [A00]
    # V: (8,1), (0,-1) [B10, B00]
    # W: (0,1), (10,-1)[C00, C10]
    terms.append(((0,1), (8,1), (0,-1), (0,1), (10,-1)))
    
    # M4: A11(B01-B00)
    # U: (3,1)         [A11]
    # V: (1,1), (0,-1) [B01, B00]
    # W: (0,-1), (10,1)[C00, C10]
    terms.append(((3,1), (1,1), (0,-1), (0,-1), (10,1)))
    
    # M5: (A00+A01)B11
    # U: (0,1), (1,1)  [A00, A01]
    # V: (8,1)         [B11]
    # W: (1,1), (12,1) [C01, C11]
    terms.append(((0,1), (1,1), (8,1), (1,1), (12,1)))
    
    # M6: (A10-A00)(B00+B10)
    # U: (2,1), (0,-1) [A10, A00]
    # V: (0,1), (8,1)  [B00, B10]
    # W: (0,1)         [C00]
    terms.append(((2,1), (0,-1), (0,1), (8,1), (0,1)))
    
    # M7: (A01-A11)(B01+B11)
    # U: (1,1), (3,-1) [A01, A11]
    # V: (1,1), (8,1)  [B01, B11]
    # W: (12,-1)       [C11]
    terms.append(((1,1), (3,-1), (1,1), (8,1), (12,-1)))
    
    return terms

def _build_hybrid():
    """
    Build a hybrid decomposition:
    - Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    - Schoolbook for the rest
    """
    n, m, p = DIMENSIONS
    rank = 62
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen terms
    strassen_terms = _strassen_2x2_terms()
    for term in strassen_terms:
        u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs = term
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] += coeff
        col += 1
    
    # Add schoolbook terms for the rest
    # The Strassen part covered A[0:2, 0:2] * B[0:2, 0:2]
    # We need to cover:
    # 1. A[0:2, 0:2] * B[0:2, 3:7] (cols 3,4,5,6)
    # 2. A[0:2, 0:2] * B[2:3, 0:7] (row 2 of B)
    # 3. A[2:3, 0:3] * B[0:3, 0:7] (row 2 of A)
    
    # 1. A[0:2, 0:2] * B[0:2, 3:7]
    for i in range(2):
        for j in range(2):
            for k in range(3, 7):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # 2. A[0:2, 0:2] * B[2:3, 0:7]
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # 3. A[2:3, 0:3] * B[0:3, 0:7]
    for i in range(2, 3):
        for j in range(3):
            for k in range(7):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Build the hybrid decomposition
    result = _build_hybrid()
    
    # Verify
    if _verify(result['U'], result['V'], result['W']):
        return result
    else:
        # Fallback to schoolbook if verification fails
        n, m, p = DIMENSIONS
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

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Verified: {_verify(result['U'], result['V'], result['W'])}")
