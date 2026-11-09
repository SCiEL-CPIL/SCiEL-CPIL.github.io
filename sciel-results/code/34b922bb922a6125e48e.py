import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    timeout = 100.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target tensor entries
    # T[i*m + j, j*p + k, k*n + i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Schoolbook solution as fallback
    R_school = n * m * p
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((np_, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school

    # Construct Winograd-like decomposition
    # A is 2x3, B is 3x5.
    # Split A into A1 (2x2) and a2 (2x1).
    # Split B into B1 (2x5) and b2 (1x5).
    # B1 split into B11 (2x2), B12 (2x2), b13 (2x1).
    
    # Indices:
    # A[i, j] -> U index i*m + j
    # B[j, k] -> V index j*p + k
    # C[i, k] -> W index k*n + i
    
    U = []
    V = []
    W = []
    
    # 1. A1 B11 (2x2 x 2x2) -> 8 terms
    # A1: A[0,0], A[0,1], A[1,0], A[1,1]
    # B11: B[0,0], B[0,1], B[1,0], B[1,1]
    # C parts: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Term 1: A[0,0] B[0,0]
    U.append([1, 0, 0, 0, 0, 0])
    V.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Term 2: A[0,1] B[1,0]
    U.append([0, 1, 0, 0, 0, 0])
    V.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Term 3: A[0,0] B[0,1]
    U.append([1, 0, 0, 0, 0, 0])
    V.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Term 4: A[0,1] B[1,1]
    U.append([0, 1, 0, 0, 0, 0])
    V.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Term 5: A[1,0] B[0,0]
    U.append([0, 0, 1, 0, 0, 0])
    V.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
    # Term 6: A[1,1] B[1,0]
    U.append([0, 0, 0, 1, 0, 0])
    V.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
    # Term 7: A[1,0] B[0,1]
    U.append([0, 0, 1, 0, 0, 0])
    V.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
    # Term 8: A[1,1] B[1,1]
    U.append([0, 0, 0, 1, 0, 0])
    V.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
    # 2. A1 B12 (2x2 x 2x2) -> 8 terms
    # B12: B[0,2], B[0,3], B[1,2], B[1,3]
    # C parts: C[0,2], C[0,3], C[1,2], C[1,3]
    
    # Term 9: A[0,0] B[0,2]
    U.append([1, 0, 0, 0, 0, 0])
    V.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    
    # Term 10: A[0,1] B[1,2]
    U.append([0, 1, 0, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    
    # Term 11: A[0,0] B[0,3]
    U.append([1, 0, 0, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    
    # Term 12: A[0,1] B[1,3]
    U.append([0, 1, 0, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    
    # Term 13: A[1,0] B[0,2]
    U.append([0, 0, 1, 0, 0, 0])
    V.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    
    # Term 14: A[1,1] B[1,2]
    U.append([0, 0, 0, 1, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    
    # Term 15: A[1,0] B[0,3]
    U.append([0, 0, 1, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    
    # Term 16: A[1,1] B[1,3]
    U.append([0, 0, 0, 1, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    
    # 3. A1 b13 (2x2 x 2x1) -> 4 terms
    # b13: B[0,4], B[1,4]
    # C parts: C[0,4], C[1,4]
    
    # Term 17: A[0,0] B[0,4]
    U.append([1, 0, 0, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    
    # Term 18: A[0,1] B[1,4]
    U.append([0, 1, 0, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    
    # Term 19: A[1,0] B[0,4]
    U.append([0, 0, 1, 0, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    # Term 20: A[1,1] B[1,4]
    U.append([0, 0, 0, 1, 0, 0])
    V.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    W.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    
    # 4. a2 b2^T (2x1 x 1x5) -> 1 term? No, 5 terms?
    # a2: A[0,2], A[1,2]
    # b2: B[2,0], B[2,1], B[2,2], B[2,3], B[2,4]
    # C parts: C[0,0..4], C[1,0..4]
    
    # This is an outer product.
    # A[0,2] B[2,0] -> C[0,0]
    # A[0,2] B[2,1] -> C[0,1]
    # ...
    # This cannot be done in 1 term. It requires 5 terms?
    # T[i, j, k] = A[i, 2] B[2, k].
    # This is a rank 5 tensor?
    # No, it's a sum of 5 rank 1 tensors?
    # A[0,2] B[2,0] is one term.
    # A[0,2] B[2,1] is another.
    # So 5 terms for the first row of A and 5 for the second?
    # No, A[0,2] B[2,k] contributes to C[0,k].
    # A[1,2] B[2,k] contributes to C[1,k].
    # So for each k, we have 2 terms?
    # Total 10 terms?
    # Wait, the schoolbook for this part is 2 * 5 = 10 terms.
    # Can we do better?
    # It's just a matrix-vector product for each row.
    # A[0,2] * B[2, :] -> vector of length 5.
    # A[1,2] * B[2, :] -> vector of length 5.
    # This is rank 10?
    # Yes, because each output C[i,k] depends on A[i,2] and B[2,k].
    # They are independent.
    
    # So we need 10 more terms.
    # Total rank 20 + 10 = 30.
    # Back to schoolbook!
    
    # Ah, the outer product A[0,2] B[2,k] is not a single term.
    # It is 5 separate multiplications for row 0, and 5 for row 1.
    # So rank is 10.
    
    # My previous calculation of rank 19 was wrong because I assumed outer product is rank 1.
    # Outer product of vectors u (size n) and v (size p) has rank n*p.
    # So rank 2 + 10 = 12? No.
    # A1 B11: 8
    # A1 B12: 8
    # A1 b13: 4
    # a2 b2: 10
    # Total 30.
    
    # So this decomposition is not better than schoolbook.
    
    # However, we can use Winograd for the 2x2x2 parts to get rank 7.
    # 7 + 7 + 4 + 10 = 28.
    # 28 < 30.
    
    # Let's implement the rank 7 Winograd for 2x2x2.
    
    U = []
    V = []
    W = []
    
    # Winograd 2x2x2 Rank 7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(e+g+h) -- No
    # Standard Winograd:
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(e+g+h)
    # M3 = (a+b)(f+h)
    # M4 = (c+d)(e+g)
    # M5 = (c+d)(e+g+h)
    # M6 = (c+d)(f+h)
    # M7 = (a-b)(e-g) ? No.
    
    # Let's use the explicit rank 7 decomposition:
    # M1 = (a+b)(e+g)
    # M2 = (a+b)(e+g+h)
    # M3 = (a+b)(f+h)
    # M4 = (c+d)(e+g)
    # M5 = (c+d)(e+g+h)
    # M6 = (c+d)(f+h)
    # M7 = (a-b)(e-g) ?
    
    # Actually, let's just use the 8 terms for simplicity and safety, rank 8.
    # 8 + 8 + 4 + 10 = 30.
    # Still 30.
    
    # We need a better strategy.
    # Maybe use 3x3 Winograd?
    # Or just return the schoolbook result.
    
    return best_result

if __name__ == "__main__":
    construct()
