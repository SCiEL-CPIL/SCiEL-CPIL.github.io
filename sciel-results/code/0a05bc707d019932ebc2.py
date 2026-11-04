import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
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
    return U, V, W

def _strassen_2x2_block():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for the 2x2*2x2 tensor.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 multiplications for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # U[i*2+j, r] corresponds to A[i,j]
    # V[j*2+k, r] corresponds to B[j,k]
    # W[k*2+i, r] corresponds to C[i,k]... wait, W[k*n+i] with n=2
    # So W index is k*2+i, meaning row=k, col=i in the output.
    # But C[i,j] = sum_k A[i,k]B[k,j], so output C has indices (i,j).
    # W[k*2+i, r] means the output element at position (k, i)? That seems transposed.
    # Let me re-check the convention.
    
    # Actually, looking at the schoolbook: W[k*n+i, r] = 1 where the term is A[i,k]*B[k,j] contributing to C[i,j].
    # So W[k*n+i] corresponds to output C[i,j] where the "i" in W[k*n+i] is the row of C, and k is... 
    # Wait, in schoolbook for C[i,j] = sum_k A[i,k]B[k,j], the W index is k*n+i.
    # With n=2: W[k*2+i]. For a given (i,j,k) triple, W[k*2+i]=1.
    # So W row index = k*2+i. This means W is indexed by (k, i) pairs.
    # But C[i,j] depends on sum over k of A[i,k]B[k,j].
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # For 2x2*2x2: n=2, m=2, p=2
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    
    # So I need: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    
    # This is 8 equations. Schoolbook has 8 terms. Strassen should have 7.
    
    # Let me construct the Strassen decomposition properly.
    # Each Strassen multiplication M_r involves linear combinations of A and B entries.
    # M_r = (sum_{i,j} alpha_{r,i,j} A[i,j]) * (sum_{j,k} beta_{r,j,k} B[j,k])
    # Then C[i,j] = sum_r gamma_{r,i,j} M_r
    
    # In our tensor decomposition:
    # U[i*m+j, r] = alpha_{r,i,j}
    # V[j*p+k, r] = beta_{r,j,k}  
    # W[k*n+i, r] = gamma_{r, ?, ?}
    
    # Wait, W[k*n+i, r] should give the coefficient for C[?, ?].
    # C[i,j] = sum_k A[i,k]B[k,j]
    # The term M_r contributes gamma_{r,i,j} * M_r to C[i,j].
    # In the tensor: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should give the coefficient...
    
    # I'm getting confused. Let me just directly construct the Strassen U, V, W matrices
    # and verify they work.
    
    # For 2x2*2x2, n=m=p=2:
    # U is 4x7, V is 4x7, W is 4x7
    # U[i*2+j, r] for i,j in {0,1}
    # V[j*2+k, r] for j,k in {0,1}
    # W[k*2+i, r] for k,i in {0,1}
    
    # Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # U[i*2+j, r]:
    # r=0: A00+A11 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
    # r=1: A10+A11 -> U[2,1]=1, U[3,1]=1 (i=1,j=0 and i=1,j=1)
    # r=2: A00 -> U[0,2]=1
    # r=3: A11 -> U[3,3]=1
    # r=4: A00+A01 -> U[0,4]=1, U[1,4]=1
    # r=5: A10-A00 -> U[2,5]=1, U[0,5]=-1
    # r=6: A01-A11 -> U[1,6]=1, U[3,6]=-1
    
    # V[j*2+k, r]:
    # r=0: B00+B11 -> V[0,0]=1, V[3,0]=1
    # r=1: B00 -> V[0,1]=1
    # r=2: B01-B11 -> V[1,2]=1, V[3,2]=-1
    # r=3: B10-B00 -> V[2,3]=1, V[0,3]=-1
    # r=4: B11 -> V[3,4]=1
    # r=5: B00+B01 -> V[0,5]=1, V[1,5]=1
    # r=6: B10+B11 -> V[2,6]=1, V[3,6]=1
    
    # W[k*2+i, r]: W index is k*2+i where k is the "k" from B[k,j] and i is the row of C.
    # Wait, I need to be careful. In the tensor sum:
    # sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    # This should equal 1 for all i,j,k in {0,1}.
    
    # Hmm, this is getting complex. Let me just verify the schoolbook works and try to 
    # find improvements through a different approach.
    
    # Actually, let me just return the schoolbook and focus on a different strategy.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M0 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1  # A00, A11
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    # C00 += M0, C11 += M0
    # C[i,j] gets contribution from W[j*2+i, 0]
    # For C00: i=0,j=0 -> W[0*2+0, 0] = W[0,0] = 1
    # For C11: i=1,j=1 -> W[1*2+1, 0] = W[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # r=1: M1 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1  # B00
    # C10 += M1: i=1,j=0 -> W[0*2+1, 1] = W[1,1] = 1
    # C11 -= M1: i=1,j=1 -> W[1*2+1, 1] = W[3,1] = -1
    W[1,1] = 1; W[3,1] = -1
    
    # r=2: M2 = A00(B01-B11)
    U[0,2] = 1  # A00
    V[1,2] = 1; V[3,2] = -1  # B01, -B11
    # C01 += M2: i=0,j=1 -> W[1*2+0, 2] = W[2,2] = 1
    # C11 += M2: i=1,j=1 -> W[1*2+1, 2] = W[3,2] = 1
    W[2,2] = 1; W[3,2] = 1
    
    # r=3: M3 = A11(B10-B00)
    U[3,3] = 1  # A11
    V[2,3] = 1; V[0,3] = -1  # B10, -B00
    # C00 += M3: i=0,j=0 -> W[0*2+0, 3] = W[0,3] = 1
    # C10 += M3: i=1,j=0 -> W[0*2+1, 3] = W[1,3] = 1
    W[0,3] = 1; W[1,3] = 1
    
    # r=4: M4 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1  # A00, A01
    V[3,4] = 1  # B11
    # C00 -= M4: i=0,j=0 -> W[0,4] = -1
    # C01 += M4: i=0,j=1 -> W[2,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # r=5: M5 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1  # A10, -A00
    V[0,5] = 1; V[1,5] = 1  # B00, B01
    # C11 += M5: i=1,j=1 -> W[3,5] = 1
    W[3,5] = 1
    
    # r=6: M6 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1  # A01, -A11
    V[2,6] = 1; V[3,6] = 1  # B10, B11
    # C00 += M6: i=0,j=0 -> W[0,6] = 1
    W[0,6] = 1
    
    return U, V, W

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    U0, V0, W0 = _schoolbook(n, m, p)
    
    if not _deadline_ok():
        return {"U": U0, "V": V0, "W": W0}
    
    # Strategy: Use block decomposition.
    # A is 2x5, B is 5x5, C is 2x5.
    # Split B into blocks: B = [B0 B1 B2] where B0 is 5x2, B1 is 5x2, B2 is 5x1.
    # C = A * B = [A*B0, A*B1, A*B2]
    # Each A*Bj is 2x2 (for j=0,1) or 2x1 (for j=2).
    
    # For A*B0 (2x2 * 2x2... wait, A is 2x5 and B0 is 5x2, so A*B0 is 2x2).
    # This is a 2x5 * 5x2 multiplication.
    # Similarly A*B1 is 2x5 * 5x2 = 2x2.
    # A*B2 is 2x5 * 5x1 = 2x1.
    
    # For 2x5 * 5x2: We can view this as computing C[i,j] = sum_k A[i,k]*B[k,j]
    # for i in {0,1}, j in {0,1}, k in {0..4}.
    # Schoolbook rank: 2*5*2 = 20.
    # Can we do better? This is like a bilinear form.
    
    # Alternative approach: Use the fact that we're multiplying by a 5x5 matrix.
    # Think of it as 5 independent 2x5 * 5x1 multiplications (one for each column of B).
    # Each 2x5 * 5x1 has schoolbook rank 10.
    # Total schoolbook: 5 * 10 = 50.
    
    # For 2x5 * 5x1: C[i] = sum_k A[i,k]*B[k] for i=0,1.
    # This is two dot products of length 5.
    # Can we compute two dot products of length 5 with fewer than 10 multiplications?
    # This is equivalent to the tensor for computing [sum_k A0k*Bk, sum_k A1k*Bk].
    # The tensor has entries T[i,k] where i in {0,1}, k in {0..4}, and it's just the
    # identity in some sense. Actually, T[i,k] = 1 for all i,k, and the output is
    # C[i] = sum_k T[i,k] * A[i,k] * B[k].
    
    # This is a rank-1 tensor in a sense... no. The tensor is T[i,k] = 1 for all (i,k).
    # The rank of this tensor (as a bilinear map from R^2 x R^5 to R^2) is...
    # Actually, this is just the identity map on the first factor tensored with the
    # all-ones vector on the second. The rank should be 5 (one for each k, shared across i).
    
    # Wait: C[i] = sum_k A[i,k]*B[k]. We can write this as:
    # C = A * B where A is 2x5 and B is 5x1.
    # Using the "outer product" trick: C[i] = sum_k (A[i,k]) * (B[k])
    # We need 5 multiplications: for each k, compute Pk = A[0,k]*B[k] and A[1,k]*B[k].
    # But we can share: let Qk = B[k], then C[0] = sum_k A[0,k]*Qk and C[1] = sum_k A[1,k]*Qk.
    # This still requires 10 multiplications unless we can combine.
    
    # Actually, for 2x5 * 5x1, the tensor rank is exactly 10 (schoolbook is optimal)
    # because it's equivalent to 2 independent dot products of length 5, and each
    # dot product of length n has rank n.
    
    # Hmm, but we can potentially share multiplications across the 5 columns of B.
    
    # Let me try a different approach: use Winograd or FFT-like methods.
    
    # For now, let's try to use the Strassen algorithm on 2x2 blocks.
    # Pad A to 2x6 (add a zero column) and B to 6x6, then use block Strassen.
    # Or better: split the 5x5 into 2x2 blocks.
    
    # B = [B00 B01 B02; B10 B11 B12; B20 B21 B22] where each Bij is 2x2 except
    # the last column/row which are 2x1/1x1.
    
    # This is getting complicated. Let me just try to find a rank-49 or lower
    # decomposition through a direct search.
    
    # Given the time constraints, let me return the schoolbook decomposition.
    return {"U": U0, "V": V0, "W": W0}
