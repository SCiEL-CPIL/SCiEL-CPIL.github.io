"""Tensor decomposition for 2x4x6 matrix multiplication.

Approach: Direct rank-42 construction via 6 independent 2x2x2 Strassen blocks.
The 2x4x6 multiplication tensor can be partitioned into 6 sub-tensors, each
corresponding to a 2x2x2 multiplication (since 4 = 2*2 and 6 = 2*3, but we
can pair columns differently).

Actually, let me think more carefully. The 2x4x6 tensor T[i,j,k] = 1 for all
i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.

One approach: partition the 4 columns of A into two groups of 2, and the 6 
columns of B into three groups of 2. This gives 6 blocks, each a 2x2x2 
multiplication. But that doesn't quite work because the indices don't align 
simply.

Better approach: Use the fact that a 2x4 @ 4x6 multiplication can be seen as:
- Split the 4x4 middle dimension: j = 2*j1 + j2 where j1 in {0,1}, j2 in {0,1}
- Split the 6 output dimension: k = 2*k1 + k2 where k1 in {0,1,2}, k2 in {0,1}
- The 2 input rows: i in {0,1}

For each (j1, k1) pair (6 pairs), we have a 2x2x2 multiplication:
  C[i, 2*k1 + k2] = sum_{j2} A[i, 2*j1 + j2] * B[2*j1 + j2, 2*k1 + k2]

Wait, that's not right either. Let me reconsider.

C[i,k] = sum_j A[i,j] * B[j,k]

If I write j = 2*j1 + j2 and k = 2*k1 + k2:
C[i, 2*k1 + k2] = sum_{j1, j2} A[i, 2*j1 + j2] * B[2*j1 + j2, 2*k1 + k2]

This couples all j1 values for each (i, k1, k2). So it's not simply 6 independent 
2x2x2 blocks.

Alternative: Use a block-structured approach. Consider the 4x6 matrix B split as:
B = [B_top | B_bottom] where B_top is 2x6 and B_bottom is 2x6.

Then C = A_top * B_top + A_bottom * B_bottom where A_top is 2x2 and A_bottom is 2x2.

Each of A_top * B_top and A_bottom * B_bottom is a 2x2 @ 2x6 = 2x6 multiplication.

A 2x2 @ 2x6 multiplication: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
This can be decomposed using Strassen-like methods.

Actually, let me just use the simple block approach with rank 42:
- Split the 4 intermediate dimension into 2 groups of 2
- For each group, do a 2x2 @ 2x6 multiplication
- Each 2x2 @ 2x6 has rank... let's compute.

For 2x2 @ 2x6: We need to compute C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0..5}.
That's 12 outputs, each is a sum of 2 products. Naive rank is 24.

Using Strassen on the 2x2 part: We can express the 2x2 multiplication with 7 scalar multiplications, but here the "scalar" is actually a 1x6 vector (the row of B). So we need 7 vector multiplications of size 1x6, which means 7*6 = 42 scalar multiplications total for one 2x2@2x6 block.

Wait no. Let me think again.

For 2x2 @ 2x6:
C[i,k] = sum_j A[i,j] B[j,k]

Using Strassen for the 2x2 part, we compute 7 intermediate products, each of which is a product of a linear combination of A entries (scalar) and a linear combination of B entries (which is a 1x6 vector). So each "product" is actually 6 scalar multiplications. Total: 7 * 6 = 42.

For the full 2x4 @ 4x6, we split into two 2x2 @ 2x6 blocks (top and bottom halves of the 4 intermediate dimension), each requiring rank 42. But they share the same output C, so we sum them. The total rank is 42 + 42 = 84? No, that's not right because the two blocks contribute to the same C entries.

Actually, each block computes a partial C, and we add them. The tensor decomposition sums over all rank terms, so if block 1 uses rank-42 terms and block 2 uses rank-42 terms, the total rank is 84. That's worse than schoolbook (48).

Let me try a different partitioning. Split the 6 output columns into 3 groups of 2:
For k1 in {0,1,2}, k = 2*k1 + k2:
C[i, 2*k1+k2] = sum_j A[i,j] B[j, 2*k1+k2]

For fixed k1, this is a 2x4 @ 4x2 multiplication. We can further split the 4 intermediate dimension into 2 groups of 2.

For fixed k1, C[i, 2*k1+k2] = sum_{j=0}^{3} A[i,j] B[j, 2*k1+k2]
= sum_{j1=0}^{1} sum_{j2=0}^{1} A[i, 2*j1+j2] B[2*j1+j2, 2*k1+k2]

Let A_top = A[:, 0:2] (2x2), A_bot = A[:, 2:4] (2x2)
Let B_top = B[0:2, 2*k1:2*k1+2] (2x2), B_bot = B[2:4, 2*k1:2*k1+2] (2x2)

C_part = A_top @ B_top + A_bot @ B_bot (2x2 matrix)

Each 2x2 @ 2x2 has Strassen rank 7. So for each k1, we need 7+7=14 rank terms.
For 3 values of k1, total rank = 3 * 14 = 42.

This gives a valid rank-42 decomposition!

Let me construct it explicitly.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# Strassen factors for 2x2x2 (rank 7)
# A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# C = [[a*e + c*g, a*f + c*h], [b*e + d*g, b*f + d*h]]
# Strassen:
# m1 = (a+d)(e+h)
# m2 = (c+d)e
# m3 = a(f-h)
# m4 = d(g-e)
# m5 = (a+b)h
# m6 = (c-b)(e+f)
# m7 = (a-d)(g+h)
# C00 = m1 + m4 - m5 + m7
# C01 = m3 + m5
# C10 = m2 + m4
# C11 = m1 - m2 + m3 + m6

STRASSEN_U = np.array([
    [1, 0, 0, 1],   # a+d
    [0, 0, 1, 1],   # c+d
    [1, 0, 0, 0],   # a
    [0, 0, 0, 1],   # d
    [1, 1, 0, 0],   # a+b
    [-1, 0, 1, 0],  # c-b (note: -1*0*a + 1*c + 0*b... wait)
    [0, 1, 0, -1],  # a-d... hmm, let me recheck
], dtype=float)

# Let me redo Strassen more carefully.
# A entries indexed as [a, b, c, d] where A = [[a,b],[c,d]]
# B entries indexed as [e, f, g, h] where B = [[e,f],[g,h]]

# m1 = (a+d)*(e+h): U=[1,0,0,1], V=[1,1,0,0]
# m2 = (c+d)*e:     U=[0,0,1,1], V=[1,0,0,0]
# m3 = a*(f-h):     U=[1,0,0,0], V=[0,1,0,-1]
# m4 = d*(g-e):     U=[0,0,0,1], V=[0,0,1,-1]
# m5 = (a+b)*h:     U=[1,1,0,0], V=[0,0,0,1]
# m6 = (c-b)*(e+f): U=[0,-1,1,0], V=[1,1,0,0]
# m7 = (a-d)*(g+h): U=[1,0,0,-1], V=[0,0,1,1]

# C00 = m1 + m4 - m5 + m7: W=[1,0,0,1,0,-1,1]
# C01 = m3 + m5:            W=[0,0,1,0,1,0,0]
# C10 = m2 + m4:            W=[0,1,0,1,0,0,0]
# C11 = m1 - m2 + m3 + m6:  W=[1,-1,1,0,0,1,0]

STRASSEN_U = np.array([
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, -1, 1, 0],
    [1, 0, 0, -1],
], dtype=float)

STRASSEN_V = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, -1],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
], dtype=float)

STRASSEN_W = np.array([
    [1, 0, 0, 1, 0, -1, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, 1, 0],
], dtype=float)


def _build_rank42():
    """Build a rank-42 decomposition of the 2x4x6 multiplication tensor.
    
    Partition: for each k1 in {0,1,2}, compute the 2x4 @ 4x2 sub-problem
    for columns 2*k1 and 2*k1+1 of B, split into top/bottom 2x2 blocks.
    Each 2x2 @ 2x2 uses Strassen (rank 7), so each k1 uses 2*7=14 terms.
    Total: 3 * 14 = 42.
    """
    n, m, p = 2, 4, 6
    total_rank = 42
    U = np.zeros((n * m, total_rank), dtype=float)  # (8, 42)
    V = np.zeros((m * p, total_rank), dtype=float)  # (24, 42)
    W = np.zeros((p * n, total_rank), dtype=float)  # (12, 42)
    
    col_offset = 0
    for k1 in range(3):  # 3 groups of 2 output columns
        for block in range(2):  # top (j1=0) and bottom (j1=1) 2x2 blocks
            for s in range(7):  # Strassen rank
                col = col_offset + s
                # j1 = block (0 or 1), so intermediate indices are 2*block + j2
                # k1 is fixed, output columns are 2*k1 + k2
                
                # U: rows are i*m + j = i*4 + (2*block + j2)
                # U[i, 2*block+j2, col] = STRASSEN_U[s, i*2+j2]... 
                # Wait, the 2x2 A block has entries A[i, 2*block+j2] for i in {0,1}, j2 in {0,1}
                # So in the 2x2 Strassen, A_local[i,j2] = A[i, 2*block+j2]
                # U_local[i, j2, s] = STRASSEN_U[s, i*2+j2]
                # U[i*m + (2*block+j2), col] = STRASSEN_U[s, i*2+j2]
                for i in range(2):
                    for j2 in range(2):
                        j = 2 * block + j2
                        U[i * m + j, col] = STRASSEN_U[s, i * 2 + j2]
                
                # V: rows are j*p + k = (2*block+j2)*6 + (2*k1+k2)
                # V_local[j2, k2, s] = STRASSEN_V[s, j2*2+k2]
                # V[(2*block+j2)*6 + (2*k1+k2), col] = STRASSEN_V[s, j2*2+k2]
                for j2 in range(2):
                    for k2 in range(2):
                        j = 2 * block + j2
                        k = 2 * k1 + k2
                        V[j * p + k, col] = STRASSEN_V[s, j2 * 2 + k2]
                
                # W: rows are k*n + i = (2*k1+k2)*2 + i
                # W_local[k2, i, s] = STRASSEN_W[s, k2*2+i]
                # W[(2*k1+k2)*2 + i, col] = STRASSEN_W[s, k2*2+i]
                for k2 in range(2):
                    for i in range(2):
                        k = 2 * k1 + k2
                        W[k * n + i, col] = STRASSEN_W[s, k2 * 2 + i]
        
        col_offset += 14
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    U, V, W = _build_rank42()

    # Verify correctness
    if _verify(U, V, W, n, m, p):
        # Convert to list of lists with int where possible
        U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
        V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
        W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback: schoolbook rank 48
    schoolbook_rank = n * m * p
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_fb[ii * m + jj][col] = 1
                V_fb[jj * p + kk][col] = 1
                W_fb[kk * n + ii][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
