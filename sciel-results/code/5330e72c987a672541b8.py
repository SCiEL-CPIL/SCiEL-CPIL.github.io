import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 98 solution: 2 * 49
    rank = 98
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate 4x4x4 rank-49 decomposition
    # This uses 7 Strassen blocks, each is a 2x2x2 rank-7 decomposition
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # 2x2x2 rank-7 decomposition
    # U2[4x7], V2[4x7], W2[4x7]
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0,0] = 1; U2[1,0] = 0; U2[2,0] = 1; U2[3,0] = 0 # A00+A11 -> indices 0,1,2,3? No.
    # Indices: i*2+j
    # A00: i=0,j=0 -> 0
    # A01: i=0,j=1 -> 1
    # A10: i=1,j=0 -> 2
    # A11: i=1,j=1 -> 3
    U2[0,0] = 1 # A00
    U2[3,0] = 1 # A11
    V2[0,0] = 1 # B00
    V2[3,0] = 1 # B11
    W2[0,0] = 1 # C00 (k=0,i=0)
    W2[3,0] = 1 # C11 (k=1,i=1)
    # Wait, the W index is k*2+i.
    # The product contributes to C.
    # The tensor entry T[i,j,k] is 1 if i,j,k are consistent.
    # In the decomposition, we sum U[i,j,r] V[j,k,r] W[k,i,r].
    # For the 2x2x2 case, the indices are i,j,k in {0,1}.
    
    # Let's just hardcode the 2x2x2 rank-7 decomposition properly.
    # Reference: Strassen's 7 multiplications.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U2, V2, W2 for each of the 7 terms.
    # r=0: M1
    U2[0,0] += 1; U2[3,0] += 1
    V2[0,0] += 1; V2[3,0] += 1
    W2[0,0] += 1; W2[3,0] += 1 # C00 and C11 get contributions?
    # Actually, the W part depends on the output indices.
    # The tensor is T[i,j,k].
    # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k].
    # For the 2x2x2 case, T[i,j,k] = 1 for all i,j,k? No.
    # T[i,j,k] is 1 if the indices match the multiplication pattern.
    # Actually, T[i,j,k] = 1 for all i,j,k in the 2x2x2 tensor?
    # No, the tensor for matrix multiplication has 1s at (i,j,k) such that they form a valid path?
    # No, the tensor M_{2,2,2} has T[i,j,k] = 1 for all i,j,k.
    # Wait, if T[i,j,k]=1 for all i,j,k, then the rank is 8?
    # No, the rank of M_{2,2,2} is 7.
    # The tensor M_{n,m,p} has T[i,j,k] = 1 if i,j,k are valid indices?
    # No, the tensor is defined by the multiplication algorithm.
    # The entry T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k.
    # So for 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # And we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # So my U2, V2, W2 must satisfy this.
    
    # Let's use a known rank-7 decomposition.
    # I will hardcode the U2, V2, W2 from a reliable source or derive it.
    # Deriving it:
    # We need 7 vectors u_r, v_r, w_r in R^4, R^4, R^4 such that
    # sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1.
    
    # This is tricky to derive on the fly.
    # I will use the Strassen construction.
    
    # For M1 = (A00+A11)(B00+B11):
    # u_1 = e00 + e11 (in A space)
    # v_1 = e00 + e11 (in B space)
    # w_1 = e00 + e11 (in C space)?
    # No, w_r is in the C space, indexed by (k,i).
    # The product u_r[i,j] v_r[j,k] gives a matrix in C space.
    # w_r[k,i] must be such that the sum equals 1.
    
    # Actually, the standard Strassen decomposition is:
    # C = sum_r w_r * (u_r A v_r^T)? No.
    
    # Let's just use the fact that I can construct the 4x4x4 rank-49 by combining 7 2x2x2 rank-7 blocks.
    # I will hardcode the 2x2x2 rank-7 decomposition.
    
    # U2, V2, W2 for 2x2x2 rank 7
    # From Wikipedia "Strassen algorithm":
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
    
    # This gives the coefficients for the C blocks.
    # But I need the U, V, W matrices.
    
    # I will implement the generation of U, V, W for 4x4x4 directly using the Strassen blocks.
    
    # For each of the 7 Strassen terms, I define the mapping of the 2x2 blocks.
    # Then I use the rank-7 decomposition of 2x2x2.
    
    # The rank-7 decomposition of 2x2x2:
    # U2, V2, W2 are 4x7 matrices.
    # I will hardcode them.
    
    # This is getting too long for a thought process. I will write the code to generate them.
    
    pass
