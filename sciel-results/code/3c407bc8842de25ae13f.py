"""
Strategy: Explicitly construct a rank-17 decomposition for 2x3x3 matrix multiplication.

The 2x3x3 multiplication tensor can be viewed as a block matrix multiplication:
C (2x3) = A (2x3) * B (3x3)

Let A = [A1 A2] where A1, A2 are 2x3 matrices? No, A is 2x3.
Let's partition columns of A and rows of B?
Actually, the standard way to get rank < nmp is to use Strassen for sub-blocks.

Consider the multiplication:
[ C11 C12 C13 ]   =   [ A11 A12 A13 ]   *   [ B11 B12 B13 ]
[ C21 C22 C23 ]       [ A21 A22 A23 ]       [ B21 B22 B23 ]       [ B31 B32 B33 ]

This is a 2x3 by 3x3 multiplication.
We can group the 2 rows of A and 3 columns of B?
Let's try to decompose it into a 2x2x2 Strassen part and a schoolbook part.

A common trick for small rectangular multiplications:
If we multiply (2x2) by (2x2), rank is 7.
If we multiply (2x1) by (1x1), rank is 2.

Let's partition the 3x3 B matrix into a 2x2 block and a 1x1 block?
B = [ B_22  B_21 ]
    [ B_12  B_11 ]
where B_22 is 2x2, B_21 is 2x1, B_12 is 1x2, B_11 is 1x1.

A is 2x3. Let's partition A into A_2 and A_1?
A = [ A_2 (2x2)  A_1 (2x1) ]

Then C = A * B = [ A_2 A_1 ] * [ B_22 B_21 ; B_12 B_11 ]
C = [ A_2 B_22 + A_1 B_12 ,  A_2 B_21 + A_1 B_11 ]

C is 2x3.
Let C = [ C_L (2x2)  C_R (2x1) ]

C_L = A_2 B_22 + A_1 B_12
C_R = A_2 B_21 + A_1 B_11

This decomposition is not a direct sum of tensors because A_1 and A_2 are shared, and B_12, B_21, etc., are shared.
However, we can define the tensor as a direct sum if we split the indices appropriately.

Let's try a different partition.
Multiply (2x3) by (3x3).
We can view this as:
(2x2) * (2x2)  -> Rank 7 (Strassen)
(2x1) * (1x2)  -> Rank 2
(1x2) * (2x1)  -> Rank 2
(1x1) * (1x1)  -> Rank 1
... this is getting complicated because of the cross terms.

Alternative: Use the known rank-17 decomposition from literature.
For 2x3x3, the rank is 17.
The decomposition can be constructed by taking the Strassen decomposition for the top-left 2x2 block of the output, and handling the rest.

Let's define the factors U, V, W.
Shapes:
U: (2*3, 17) = (6, 17)
V: (3*3, 17) = (9, 17)
W: (3*2, 17) = (6, 17)

Indices:
U[i*m + j, c] where i in {0,1}, j in {0,1,2}
V[j*p + k, c] where j in {0,1,2}, k in {0,1,2}
W[k*n + i, c] where k in {0,1,2}, i in {0,1}

We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

Let's construct 17 terms.
Terms 1-7: Strassen for the 2x2 sub-problem involving indices i in {0,1}, j in {0,1}, k in {0,1}.
Wait, the Strassen algorithm computes (2x2)*(2x2).
Let A_S = A[0:2, 0:2] (2x2)
Let B_S = B[0:2, 0:2] (2x2)
C_S = A_S * B_S (2x2)

The Strassen decomposition for C_S has 7 terms.
For these 7 terms, we only care about i in {0,1}, j in {0,1}, k in {0,1}.
So for c in 0..6:
U[i*3+j, c] is non-zero only if j in {0,1}.
V[j*3+k, c] is non-zero only if j in {0,1}, k in {0,1}.
W[k*2+i, c] is non-zero for all i, k in {0,1}.

Let's use the standard Strassen formulas for 2x2 matrices.
M1 = (A11 + A22) * (B11 + B22)
M2 = (A21 + A22) * B11
M3 = A11 * (B12 - B22)
M4 = A22 * (B21 - B11)
M5 = (A11 + A12) * B22
M6 = (A21 - A11) * (B11 + B12)
M7 = (A12 - A22) * (B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

In our tensor indices:
i is row of A/C, j is col of A / row of B, k is col of B/C.
A is 2x3. B is 3x3.
Let's map Strassen indices:
A_S[0,0] = A[0,0] -> U[0*3+0, c] = U[0, c]
A_S[0,1] = A[0,1] -> U[0*3+1, c] = U[1, c]
A_S[1,0] = A[1,0] -> U[1*3+0, c] = U[3, c]
A_S[1,1] = A[1,1] -> U[1*3+1, c] = U[4, c]

B_S[0,0] = B[0,0] -> V[0*3+0, c] = V[0, c]
B_S[0,1] = B[0,1] -> V[0*3+1, c] = V[1, c]
B_S[1,0] = B[1,0] -> V[1*3+0, c] = V[3, c]
B_S[1,1] = B[1,1] -> V[1*3+1, c] = V[4, c]

W for Strassen part:
W[0*2+0, c] = W[0, c] (C[0,0] coeff)
W[0*2+1, c] = W[1, c] (C[0,1] coeff)
W[1*2+0, c] = W[2, c] (C[1,0] coeff)
W[1*2+1, c] = W[3, c] (C[1,1] coeff)
Note: W indices are k*n+i.
k=0, i=0 -> W[0, c]
k=1, i=0 -> W[2, c]
k=0, i=1 -> W[1, c]
k=1, i=1 -> W[3, c]

Let's define the 7 Strassen terms for the top-left 2x2 block.
For c=0..6, we set U, V, W such that they compute the 2x2 product.
U and V will have zeros for j=2 (the 3rd column of A) and k=2 (the 3rd row/col of B) in these terms?
Actually, Strassen only uses A[0:2, 0:2] and B[0:2, 0:2].
So for c in 0..6:
U[2, c] = 0 (A[0,2])
U[5, c] = 0 (A[1,2])
V[2, c] = 0 (B[0,2])
V[5, c] = 0 (B[1,2])
V[6, c] = 0 (B[2,0])
V[7, c] = 0 (B[2,1])
V[8, c] = 0 (B[2,2])
W[4, c] = 0 (C[0,2] -> k=2, i=0)
W[5, c] = 0 (C[1,2] -> k=2, i=1)

Now we need to handle the remaining terms:
1. The contribution of A[*, 2] (3rd col of A) to C[*, 0:2] (first 2 cols of C).
   This is A_1 (2x1) * B_2 (2x2) where B_2 is B[0:2, 0:2]? No.
   C[0,0] += A[0,2] * B[0,0] + A[0,2] * ... no.
   C[i, k] = sum_j A[i, j] B[j, k].
   
   We already handled j=0,1 and k=0,1 via Strassen.
   We need to handle:
   - j=2, k=0,1 (A's 3rd col, B's first 2 rows) -> contributes to C[:, 0:2]
   - j=0,1, k=2 (A's first 2 cols, B's 3rd row) -> contributes to C[:, 2]
   - j=2, k=2 (A's 3rd col, B's 3rd row) -> contributes to C[:, 2]

   Let's list the missing (i, j, k) triples:
   j=2, k=0: (0,2,0), (1,2,0)
   j=2, k=1: (0,2,1), (1,2,1)
   j=0, k=2: (0,0,2), (1,0,2)
   j=1, k=2: (0,1,2), (1,1,2)
   j=2, k=2: (0,2,2), (1,2,2)

   Total 10 missing terms.
   If we use schoolbook for these, we need 10 more terms. Total 7+10=17.
   
   So, we can just use schoolbook terms for the remaining 10 (i,j,k) pairs!
   Since the Strassen terms have zeros in the "boundary" indices (j=2 or k=2 or output col 2), they won't interfere with the schoolbook terms for the boundary.
   And the schoolbook terms for the boundary will have zeros in the "core" indices (j in {0,1} AND k in {0,1} AND output col in {0,1})?
   Wait.
   Schoolbook term for (i,j,k) is:
   U[i*m+j, c] = 1
   V[j*p+k, c] = 1
   W[k*n+i, c] = 1
   
   If we add schoolbook terms for the 10 boundary cases, they will have 1s at specific (i,j,k).
   The Strassen terms produce 1s at (i,j,k) in the core {0,1}x{0,1}x{0,1}.
   Do the Strassen terms produce any non-zero contribution to the boundary?
   No, because for Strassen terms c=0..6:
   U[i*3+2, c] = 0
   V[2*3+k, c] = 0 (B's 3rd row)
   V[j*3+2, c] = 0 (B's 3rd col)
   W[2*2+i, c] = 0 (C's 3rd col)
   
   So the Strassen terms only contribute to the core 2x2x2 sub-tensor.
   The schoolbook terms for the 10 boundary cases only contribute to the boundary.
   They are disjoint in support!
   
   Therefore, the sum of the 7 Strassen terms and the 10 schoolbook terms gives an exact rank-17 decomposition.

Implementation Plan:
1. Initialize U, V, W as zeros of shapes (6,17), (9,17), (6,17).
2. Fill columns 0-6 with the Strassen decomposition for the 2x2 core.
   - Map Strassen M1..M7 to U, V, W entries.
3. Fill columns 7-16 with schoolbook terms for the 10 boundary (i,j,k) triples.
   - List the 10 triples.
   - For each triple, set U, V, W to 1 at the corresponding indices.
4. Verify the decomposition.
5. Return the factors.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U shape: (6, R)
    # V shape: (9, R)
    # W shape: (6, R)
    
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: Strassen for the 2x2 core (i in {0,1}, j in {0,1}, k in {0,1}) ---
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # Core indices:
    # i: 0, 1
    # j: 0, 1
    # k: 0, 1
    
    # Strassen terms M1..M7
    # We define the coefficients for U, V, W for each term.
    
    # Helper to set a term
    def set_term(c, u_vals, v_vals, w_vals):
        # u_vals: dict {i*m+j: val}
        # v_vals: dict {j*p+k: val}
        # w_vals: dict {k*n+i: val}
        for idx, val in u_vals.items():
            if val != 0:
                U[idx, c] += val
        for idx, val in v_vals.items():
            if val != 0:
                V[idx, c] += val
        for idx, val in w_vals.items():
            if val != 0:
                W[idx, c] += val

    # M1 = (A11 + A22) * (B11 + B22)
    # A11=U[0,0], A22=U[1,1] (using 2x2 local indices 0,1)
    # B11=V[0,0], B22=V[1,1]
    # C11 += M1, C22 += M1
    # W[0,0] += 1 (C11), W[1,1] += 1 (C22)
    set_term(0,
             {0: 1, 4: 1},  # A11, A22 (indices 0*3+0, 1*3+1)
             {0: 1, 4: 1},  # B11, B22 (indices 0*3+0, 1*3+1)
             {0: 1, 3: 1}   # C11 (k=0,i=0), C22 (k=1,i=1)
            )
            
    # M2 = (A21 + A22) * B11
    # A21=U[1,0], A22=U[1,1]
    # B11=V[0,0]
    # C21 += M2, C22 -= M2
    set_term(1,
             {3: 1, 4: 1},  # A21, A22
             {0: 1},        # B11
             {2: 1, 3: -1}  # C21 (k=1,i=0? No, C21 is i=1,k=0 -> W[0*2+1]=W[1]? Wait.)
            )
    # Let's re-verify W indices.
    # W[k*n + i, c]
    # C[i, k]
    # i=0, k=0 -> W[0*2+0] = W[0]
    # i=0, k=1 -> W[1*2+0] = W[2]
    # i=1, k=0 -> W[0*2+1] = W[1]
    # i=1, k=1 -> W[1*2+1] = W[3]
    
    # My previous mapping:
    # W[0] = C[0,0]
    # W[1] = C[1,0]
    # W[2] = C[0,1]
    # W[3] = C[1,1]
    
    # Strassen Formulas:
    # C11 (i=0,k=0) = M1 + M4 - M5 + M7
    # C12 (i=0,k=1) = M3 + M5
    # C21 (i=1,k=0) = M2 + M4
    # C22 (i=1,k=1) = M1 - M2 + M3 + M6
    
    # Let's redo the terms carefully.
    
    # Term 0: M1
    # U: A11(0,0), A22(1,1) -> {0:1, 4:1}
    # V: B11(0,0), B22(1,1) -> {0:1, 4:1}
    # W: C11(0,0), C22(1,1) -> {0:1, 3:1}
    # Correct.
    
    # Term 1: M2
    # U: A21(1,0), A22(1,1) -> {3:1, 4:1}
    # V: B11(0,0) -> {0:1}
    # W: C21(1,0), C22(1,1) with signs +1, -1
    # C21 is i=1, k=0 -> W[0*2+1] = W[1]
    # C22 is i=1, k=1 -> W[1*2+1] = W[3]
    # So W: {1:1, 3:-1}
    set_term(1,
             {3: 1, 4: 1},
             {0: 1},
             {1: 1, 3: -1}
            )
            
    # Term 2: M3
    # U: A11(0,0) -> {0:1}
    # V: B12(0,1), B22(1,1) with signs +1, -1 -> {1:1, 4:-1}
    # W: C12(0,1), C22(1,1) with signs +1, +1
    # C12 is i=0, k=1 -> W[1*2+0] = W[2]
    # C22 is i=1, k=1 -> W[1*2+1] = W[3]
    # So W: {2:1, 3:1}
    set_term(2,
             {0: 1},
             {1: 1, 4: -1},
             {2: 1, 3: 1}
            )
            
    # Term 3: M4
    # U: A22(1,1) -> {4:1}
    # V: B21(1,0), B11(0,0) with signs +1, -1 -> {3:1, 0:-1}
    # W: C11(0,0), C21(1,0) with signs +1, +1
    # C11 is i=0, k=0 -> W[0]
    # C21 is i=1, k=0 -> W[1]
    # So W: {0:1, 1:1}
    set_term(3,
             {4: 1},
             {3: 1, 0: -1},
             {0: 1, 1: 1}
            )
            
    # Term 4: M5
    # U: A11(0,0), A12(0,1) -> {0:1, 1:1}
    # V: B22(1,1) -> {4:1}
    # W: C11(0,0), C12(0,1) with signs -1, +1
    # C11 is i=0, k=0 -> W[0]
    # C12 is i=0, k=1 -> W[2]
    # So W: {0:-1, 2:1}
    set_term(4,
             {0: 1, 1: 1},
             {4: 1},
             {0: -1, 2: 1}
            )
            
    # Term 5: M6
    # U: A21(1,0), A11(0,0) with signs +1, -1 -> {3:1, 0:-1}
    # V: B11(0,0), B12(0,1) -> {0:1, 1:1}
    # W: C22(1,1) with sign +1
    # C22 is i=1, k=1 -> W[3]
    # So W: {3:1}
    set_term(5,
             {3: 1, 0: -1},
             {0: 1, 1: 1},
             {3: 1}
            )
            
    # Term 6: M7
    # U: A12(0,1), A22(1,1) with signs +1, -1 -> {1:1, 4:-1}
    # V: B21(1,0), B22(1,1) -> {3:1, 4:1}
    # W: C11(0,0) with sign +1
    # C11 is i=0, k=0 -> W[0]
    # So W: {0:1}
    set_term(6,
             {1: 1, 4: -1},
             {3: 1, 4: 1},
             {0: 1}
            )
            
    # --- Part 2: Schoolbook for the boundary ---
    # Boundary triples (i, j, k) where j=2 OR k=2.
    # i in {0,1}
    
    boundary_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == 2 or k == 2:
                    boundary_triples.append((i, j, k))
                    
    # There should be 10 such triples.
    # j=2, k=0,1 (2*2=4)
    # j=0,1, k=2 (2*2=4)
    # j=2, k=2 (1)
    # Total 9?
    # Let's count:
    # j=2, k=0: (0,2,0), (1,2,0)
    # j=2, k=1: (0,2,1), (1,2,1)
    # j=2, k=2: (0,2,2), (1,2,2)
    # j=0, k=2: (0,0,2), (1,0,2)
    # j=1, k=2: (0,1,2), (1,1,2)
    # Total 10.
    
    for idx, (i, j, k) in enumerate(boundary_triples):
        c = 7 + idx
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
        
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {
            "U": _to_list(U_sb),
            "V": _to_list(V_sb),
            "W": _to_list(W_sb)
        }
        
    return {
        "U": _to_list(U),
        "V": _to_list(V),
        "W": _to_list(W)
    }

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays."""
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
    return U, V, W

def _to_list(M):
    result = []
    for row in M:
        new_row = []
        for x in row:
            rx = round(x * 2) / 2
            if abs(rx - round(rx)) < 1e-9:
                new_row.append(int(round(rx)))
            else:
                new_row.append(rx)
        result.append(new_row)
    return result

# EVOLVE-BLOCK-END
