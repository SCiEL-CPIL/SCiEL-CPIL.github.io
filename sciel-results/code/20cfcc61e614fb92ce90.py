import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 28, 14

    # 1. Construct the base tensors
    # Strassen 2x2x2 (rank 7)
    # Indices: U is 4x7, V is 4x7, W is 4x7
    # Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int64)
    
    V_s = np.array([
        [0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, -1, 0, 0, 1],
        [0, 0, -1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=np.int64)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int64)

    # Schoolbook 2x2x1 (rank 2)
    # Indices: U is 4x2, V is 2x2, W is 2x2
    # Output is 2x1, so W index is k*2+i where k=0.
    # U[i*2+j, r], V[j*1+0, r], W[0*2+i, r]
    U_1 = np.zeros((4, 2), dtype=np.int64)
    V_1 = np.zeros((2, 2), dtype=np.int64)
    W_1 = np.zeros((2, 2), dtype=np.int64)
    for r in range(2):
        # i=r, j=0 and i=r, j=1 contribute to output i=r
        U_1[r * 2 + 0, r] = 1
        U_1[r * 2 + 1, r] = 1
        V_1[0, r] = 1
        V_1[1, r] = 1
        W_1[r, r] = 1

    # 2. Construct the 2x4x8 tensor decomposition
    # We view the 2x4 matrix A as two 2x2 blocks horizontally: A = [A0 | A1]
    # We view the 4x8 matrix B as four 2x2 blocks vertically: B = [B0; B1] where B0, B1 are 2x4, 
    # but wait. B is 4x8.
    # Let's align indices carefully.
    # A is 2x4. i in {0,1}, j in {0,1,2,3}.
    # B is 4x8. j in {0,1,2,3}, k in {0..7}.
    # C is 2x8. i in {0,1}, k in {0..7}.
    
    # We split j into two blocks of size 2: j_block 0 (j=0,1) and j_block 1 (j=2,3).
    # We split k into four blocks of size 2: k_block 0 (k=0,1), 1 (k=2,3), 2 (k=4,5), 3 (k=6,7).
    
    # For each j_block (2 blocks) and each k_block (4 blocks), we have a 2x2x2 multiplication.
    # Total 8 blocks of 2x2x2.
    # Rank per block = 7. Total rank = 8 * 7 = 56.
    # This is the schoolbook rank for 2x4x8.
    # Can we do better?
    # The "inspiration" mentioned rank 49 for 2x4x8.
    # Rank 49 is 7*7. This suggests a different structure.
    # Actually, 2x4x8 is equivalent to 2x4x4 * 2 (in output dim)? No.
    # Let's look at the dimensions: 2 * 4 * 8 = 64.
    # Strassen for 2x2x2 is 7.
    # If we use Strassen for all 2x2x2 sub-blocks, we get 56.
    # Is there a known decomposition for 2x4x8 with rank < 56?
    # The prompt says "constructing an explicit rank-49 decomposition for the 2x4x8 multiplication tensor".
    # How?
    # Maybe it uses the fact that 4 = 2+2 and 8 = 4+4?
    # Or maybe it exploits the specific structure of 2x4x8.
    # Let's try to construct a rank 49 decomposition for 2x4x8.
    # 49 = 7 * 7.
    # This looks like a tensor product of two Strassen decompositions?
    # If we have M2 (2x2x2) with rank 7.
    # M4 (4x4x4) with rank 49? No, M4 rank is usually 49 (Strassen^2 is 49? No, Strassen is 7 for 2. 7^2=49 for 4x4x4? Yes, if we use Kronecker product of Strassen).
    # But we have 2x4x8.
    # Let's reshape 2x4x8 to (2x2) x (4x2) x (2x4)? No.
    # Let's reshape to (2*2) x (4*2) x (2*4) = 4x8x8? No.
    
    # Let's look at the target: 2x4x7.
    # If we have a rank 49 decomposition for 2x4x8, we can just drop the terms corresponding to the 8th column (k=7).
    # The terms for k=7 are those where W has non-zero entries in row k=7 (which is index 7 in the 2x4 output, i.e., k=3, i=1 in the 2x4 block structure? No, k is 0..7).
    # If the decomposition is block-structured such that the k=7 terms are independent and can be simply removed, we get rank 49 - (rank of k=7 part).
    # If the k=7 part is a 2x4x1 multiplication, its rank is 4 (schoolbook) or maybe less?
    # 2x4x1 rank is 4 (each of the 4 j's contributes to each of the 2 i's? No, sum over j. C[i] = sum_j A[i,j] B[j]. Rank is 4).
    # So if we remove the k=7 part, we might get rank 49 - 4 = 45? Or just 49 if the k=7 part was "free" or shared?
    # Actually, if we construct the 2x4x8 tensor as a sum of 7 "Strassen-like" terms that cover all 8 k's, maybe some terms cover multiple k's.
    
    # Let's try a different approach: Direct construction for 2x4x7.
    # Split j into 2+2. Split k into 2+2+2+1.
    # Blocks:
    # j0 (0,1) * k0 (0,1): 2x2x2 (Rank 7)
    # j0 (0,1) * k1 (2,3): 2x2x2 (Rank 7)
    # j0 (0,1) * k2 (4,5): 2x2x2 (Rank 7)
    # j0 (0,1) * k3 (6):   2x2x1 (Rank 2)
    # j1 (2,3) * k0 (0,1): 2x2x2 (Rank 7)
    # j1 (2,3) * k1 (2,3): 2x2x2 (Rank 7)
    # j1 (2,3) * k2 (4,5): 2x2x2 (Rank 7)
    # j1 (2,3) * k3 (6):   2x2x1 (Rank 2)
    
    # Total Rank = 6 * 7 + 2 * 2 = 42 + 4 = 46.
    # This is better than 56.
    # Can we do better?
    # What if we split k into 1+1+1+1+1+1+1?
    # 2x2x1 blocks. Rank 2 each.
    # j0 has 7 such blocks. j1 has 7 such blocks. Total 14 blocks. Rank 14*2 = 28.
    # Wait, 2x2x1 rank is 2.
    # So 14 * 2 = 28?
    # Let's check 2x2x1 rank.
    # C[i] = A[i,0]B[0] + A[i,1]B[1].
    # This is a linear combination.
    # Tensor T[i,j,k] = delta_{i,j} ... no.
    # T_{i,j,k} for 2x2x1 (k is fixed to 0).
    # T[i, j, 0] = 1 if we are multiplying? No.
    # The tensor for multiplication M_{n,m,p} has T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # For 2x2x1: i in {0,1}, j in {0,1}, k=0.
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    # This tensor is the all-ones tensor of size 2x2x1.
    # The rank of the all-ones tensor of size 2x2x1 is 2?
    # U[0,0]=1, U[0,1]=0, U[1,0]=1, U[1,1]=0
    # V[0,0]=1, V[0,1]=1, V[1,0]=1, V[1,1]=1
    # W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
    # Term 0: U0 V0 W0 = 1 * 1 * 1 = 1 at (0,0,0), (1,0,0).
    # Term 1: U1 V1 W1 = 1 * 1 * 1 = 1 at (0,1,0), (1,1,0).
    # Sum: 1 at (0,0,0), (1,0,0), (0,1,0), (1,1,0). Correct.
    # So rank is 2.
    
    # So if we use 2x2x1 blocks for all 7 k's:
    # j0: 7 blocks of rank 2 -> 14
    # j1: 7 blocks of rank 2 -> 14
    # Total rank 28.
    
    # Wait, is 2x2x1 rank really 2?
    # Yes, because it's a sum of 2 rank-1 tensors.
    # Can it be rank 1? No, because T[0,0,0]T[1,1,0] = 1*1 = 1, but T[0,1,0]T[1,0,0] = 1*1 = 1.
    # For rank 1, T[i,j,k] = u_i v_j w_k.
    # T[0,0,0] = u0 v0 w0 = 1
    # T[1,1,0] = u1 v1 w0 = 1
    # T[0,1,0] = u0 v1 w0 = 1
    # T[1,0,0] = u1 v0 w0 = 1
    # u0 v0 = u0 v1 = u1 v0 = u1 v1 = 1/w0.
    # This implies u0=u1 and v0=v1.
    # Then T[0,0,0] = u0 v0 w0 = 1.
    # This works?
    # If u=[1,1], v=[1,1], w=[1], then T[i,j,0] = 1 for all i,j.
    # Yes! The rank is 1?
    # Let's check.
    # U = [[1], [1]] (2x1)
    # V = [[1, 1]] (1x2) -- wait, V is m x rank. m=2.
    # V = [[1], [1]] (2x1)
    # W = [[1], [1]] (1x1? No, p=1, n=2. W is p*n x rank = 2x1)
    # W = [[1], [1]]
    # Term: U[i,0] V[j,0] W[k*2+i, 0].
    # k=0.
    # i=0, j=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1.
    # i=0, j=1: U[0,0]V[1,0]W[0,0] = 1*1*1 = 1.
    # i=1, j=0: U[1,0]V[0,0]W[2,0] = 1*1*1 = 1.
    # i=1, j=1: U[1,0]V[1,0]W[2,0] = 1*1*1 = 1.
    # Yes, rank 1 is sufficient for 2x2x1?
    # Wait, the output C is 2x1.
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # My decomposition gives:
    # C[0] = 1 * (A[0,0] + A[0,1]) * B[0,0] ?? No.
    # The tensor decomposition is T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For 2x2x1:
    # r=0:
    # U[0,0]=1, U[1,0]=1, U[2,0]=1, U[3,0]=1 (flattened i*2+j)
    # V[0,0]=1, V[1,0]=1 (flattened j*1+k=0)
    # W[0,0]=1, W[1,0]=1 (flattened k*2+i)
    # Contribution to T[i,j,0]:
    # i=0, j=0: U[0,0]V[0,0]W[0,0] = 1.
    # i=0, j=1: U[1,0]V[1,0]W[0,0] = 1.
    # i=1, j=0: U[2,0]V[0,0]W[1,0] = 1.
    # i=1, j=1: U[3,0]V[1,0]W[1,0] = 1.
    # This matches the target tensor (all ones).
    # So rank of 2x2x1 is 1.
    
    # If rank of 2x2x1 is 1, then:
    # 7 k's * 2 j_blocks * rank 1 = 14.
    # Total rank 14?
    # Let's verify if we can combine them.
    # For each k, we have a 2x2x1 block.
    # For k=0: U_k, V_k, W_k.
    # For k=1: U_{k+1}, V_{k+1}, W_{k+1}.
    # They are independent because they affect different output indices k.
    # So we can just sum them.
    # Total rank = 7 (k's) * 2 (j blocks) * 1 = 14.
    
    # Is this correct?
    # Let's check the dimensions.
    # U: 8 x 14
    # V: 28 x 14
    # W: 14 x 14
    
    # For each j_block (0,1) and k (0..6):
    # We add a term.
    # Total 14 terms.
    
    # Let's build this.
    
    U = np.zeros((8, 14), dtype=np.int64)
    V = np.zeros((28, 14), dtype=np.int64)
    W = np.zeros((14, 14), dtype=np.int64)
    
    col = 0
    for jb in range(2): # j block 0, 1
        for k in range(7):
            # j indices: jb*2, jb*2+1
            # k index: k
            # i indices: 0, 1
            
            # U part: 2x2x1 rank 1
            # U[i*4 + j, col]
            # i=0, j=jb*2: U[0*4 + jb*2, col] = 1
            # i=0, j=jb*2+1: U[0*4 + jb*2+1, col] = 1
            # i=1, j=jb*2: U[1*4 + jb*2, col] = 1
            # i=1, j=jb*2+1: U[1*4 + jb*2+1, col] = 1
            
            U[jb*2, col] = 1
            U[jb*2+1, col] = 1
            U[4 + jb*2, col] = 1
            U[4 + jb*2+1, col] = 1
            
            # V part: 2x1x1 rank 1
            # V[j*7 + k, col]
            # j=jb*2: V[jb*2*7 + k, col] = 1
            # j=jb*2+1: V[(jb*2+1)*7 + k, col] = 1
            
            V[(jb*2)*7 + k, col] = 1
            V[(jb*2+1)*7 + k, col] = 1
            
            # W part: 1x2x1 rank 1
            # W[k*2 + i, col]
            # i=0: W[k*2 + 0, col] = 1
            # i=1: W[k*2 + 1, col] = 1
            
            W[k*2 + 0, col] = 1
            W[k*2 + 1, col] = 1
            
            col += 1
            
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook if verification fails (should not happen)
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}


def _build_schoolbook(nn, mm, pp):
    rank = nn * mm * pp
    U = np.zeros((nn * mm, rank), dtype=np.int64)
    V = np.zeros((mm * pp, rank), dtype=np.int64)
    W = np.zeros((pp * nn, rank), dtype=np.int64)
    col = 0
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                U[i * mm + j, col] = 1
                V[j * pp + k, col] = 1
                W[k * nn + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
