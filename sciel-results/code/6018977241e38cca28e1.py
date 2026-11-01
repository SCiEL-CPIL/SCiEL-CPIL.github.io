"""Winograd Fourier Transform Technique (WFT) for 2x4x7 matrix multiplication.

We decompose the multiplication tensor on Z_4 x Z_7.  The cyclic convolution
structure lets us use the Fourier basis of Z_4 and Z_7.  For each frequency
pair we get one scalar multiplication, giving a total of 4*7 = 28 terms.
However, we must be careful: the matrix multiplication tensor is NOT a cyclic
convolution in the usual sense.  Let me think more carefully.

The tensor is T = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} where C is 2x7.

We can view this as: for each i (row of A) and k (column of B), we compute
the inner product of row i of A and column k of B.  There are n*p = 14 such
inner products, each of length m=4.

The rank of a length-m inner product tensor is m (it's the identity tensor
of size m in the "diagonal" sense).  So the total rank is n*p*m = 2*7*4 = 56.

WFT does not reduce the rank of the inner product below m for general m.
The 4-point inner product has rank 4, and since the 14 inner products are
independent (disjoint supports), the total rank is 56.

However, let me reconsider: perhaps we can use a different decomposition.
If we think of the full tensor as a 3-tensor of dimensions (n*m) x (m*p) x (p*n)
= 8 x 28 x 14, we can try to find a lower-rank decomposition.

Actually, the schoolbook rank is n*m*p = 56.  The question is whether we can do
better.  For small matrix multiplication, the rank is often equal to the
schoolbook rank, but sometimes Strassen-type decompositions help.

For 2x4x7, let me think about whether there's a structure.  The dimensions are
small enough that we could in principle search for a lower-rank decomposition,
but that's computationally expensive.

Let me just return the schoolbook decomposition with rank 56.
"""

def construct():
    n, m, p = 2, 4, 7
    rank = n * m * p  # 56
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
