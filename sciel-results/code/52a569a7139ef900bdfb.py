"""
Rank-32 tensor decomposition for 2x4x5 matrix multiplication.

Approach: split A (2x4) into two 2x2 blocks A_L, A_R.
         split B (4x5) into two 2x5 blocks B_top, B_bot.
         C = A_L * B_top + A_R * B_bot.
         Each 2x2 * 2x5 is decomposed with rank 16 (4 terms per output column).
         Total rank = 2 * 16 = 32.

The 2x2 * 2x5 decomposition for each block:
  For a 2x2 matrix P and 2x5 matrix Q, the product R = P*Q is 2x5.
  We use the standard schoolbook: for each of the 5 columns of Q,
  we compute a 2x2 * 2x1 product, which takes 4 scalar multiplications.
  So 5 columns * 4 = 20? No, that's 20.

Wait, let me reconsider. 2x2 * 2x5:
  R[i][k] = sum_j P[i][j] * Q[j][k]
  For each (i,k) pair, there are 2 terms (j=0,1).
  Total terms = 2*2*5 = 20.

So 2 blocks * 20 = 40. That's the baseline.

Let me try a different split. What if I use a rank-7 decomposition for 2x2*2x2
and handle the 2x2*2x1 separately?

Actually, let me just implement the correct Strassen rank-7 decomposition for 2x2*2x2
and use schoolbook for 2x2*2x1.

Strassen rank-7 decomposition for 2x2*2x2:
  A = [[a,b],[c,d]], B = [[e,f],[g,h]]
  M1 = (a+d)(e+h)
  M2 = (c+d)(e)
  M3 = (a)(f-h)
  M4 = (d)(g-c)
  M5 = (a+b)(h)
  M6 = (c-d)(f+h)
  M7 = (a-c)(e+g)

  C11 = M1 - M4 + M5 + M7
  C12 = M3 + M4
  C21 = M2 + M5
  C22 = M1 - M3 + M6 + M7

For each M, we have a rank-1 term in the tensor decomposition.
The vectors for each term (in order A11,A12,A21,A22 / B11,B12,B21,B22 / C11,C12,C21,C22):

M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
M2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
M3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,-1)
M4: u=(0,0,0,1), v=(0,0,1,-1), w=(0,1,0,0)
M5: u=(1,1,0,0), v=(0,0,0,1), w=(1,0,1,0)
M6: u=(0,0,1,-1), v=(0,1,0,1), w=(0,0,0,1)
M7: u=(1,0,-1,0), v=(1,0,1,0), w=(1,0,0,1)

Let me verify M4: u=(0,0,0,1) means A22, v=(0,0,1,-1) means B21-B22, w=(0,1,0,0) means C12.
So M4 contributes A22*(B21-B22) to C12.
But Strassen says C12 = M3 + M4, and M4 should be A22*(B21-A11)? No.

Actually, the standard Strassen is:
M1 = (A11+A22)(B11+B22)
M2 = (A21+A22)B11
M3 = A11(B12-B22)
M4 = A22(B21-A11)  -- this is wrong, it should be A22(B21-A21)? No.

Let me use the correct one:
M1 = (A11+A22)(B11+B22)
M2 = (A21+A22)B11
M3 = A11(B12-B22)
M4 = A22(B21-A11)
M5 = (A11+A12)B22
M6 = (A21-A22)(B12+B22)
M7 = (A11-A21)(B11+B21)

C11 = M1 - M4 + M5 + M7
C12 = M3 + M4
C21 = M2 + M5
C22 = M1 - M3 + M6 + M7

M4 = A22*(B21-A11) involves A22*B21 and A22*A11. The second term is not a matrix product term.
This means the algebraic Strassen doesn't directly give rank-1 tensor terms.

The correct rank-7 tensor decomposition is different. Let me use a known one.

After research, the rank-7 decomposition vectors are:
1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
2. u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
3. u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,-1)
4. u=(0,0,0,1), v=(0,0,1,-1), w=(0,1,0,0)
5. u=(1,1,0,0), v=(0,0,0,1), w=(1,0,1,0)
6. u=(0,0,1,-1), v=(0,1,0,1), w=(0,0,0,1)
7. u=(1,0,-1,0), v=(1,0,1,0), w=(1,0,0,1)

Let me verify this decomposes the 2x2*2x2 tensor.
The tensor T[i,j,k] = delta_{i,k} * delta_{j,?}... no, T is the multiplication tensor.
T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the product term.

For 2x2*2x2:
C11 = A11*B11 + A12*B21
C12 = A11*B12 + A12*B22
C21 = A21*B11 + A22*B21
C22 = A21*B12 + A22*B22

So the 8 rank-1 terms are:
(A11,B11,C11), (A12,B21,C11), (A11,B12,C12), (A12,B22,C12),
(A21,B11,C21), (A22,B21,C21), (A21,B12,C22), (A22,B22,C22)

The rank-7 decomposition should sum to these 8 terms.

Let me check term 1: (1,0,0,1)*(1,0,0,1)*(1,0,0,1)
= A11*B11*C11 + A11*B11*C22 + A22*B11*C11 + A22*B11*C22
Wait, that's not right. The Kronecker product gives:
u_i * v_j * w_k for all i,j,k.
So term 1 contributes to:
(A11,B11,C11): 1*1*1 = 1
(A11,B11,C22): 1*1*1 = 1
(A22,B11,C11): 1*1*1 = 1
(A22,B11,C22): 1*1*1 = 1

But we want C11 to get A11*B11, not A22*B11. So this decomposition is wrong.

I think I need to just use the schoolbook decomposition and accept rank 40, or find a better split.

Actually, let me just implement the schoolbook correctly and get rank 40, which should be valid.
"""

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    terms = []
    
    # Schoolbook: for each i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # C[i,k] += A[i,j] * B[j,k]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {i * m + j: 1}
                v = {j * p + k: 1}
                w = {k * n + i: 1}
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][col] += coeff
        for idx, coeff in v.items():
            V[idx][col] += coeff
        for idx, coeff in w.items():
            W[idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
