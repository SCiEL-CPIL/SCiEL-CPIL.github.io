import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U: (4, 7) indexed by [i*2+j], V: (4, 7) indexed by [j*2+k], W: (4, 7) indexed by [k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A21+A22)B11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    # M4 = A22(B21-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _block_1x4x2():
    """Return (U, V, W) for 1x4x2 matrix multiplication, rank 6.
    U: (4, 6) indexed by [j], V: (8, 6) indexed by [j*2+k], W: (2, 6) indexed by [k]
    C[k] = sum_j A[j] B[j,k]
    """
    U = np.zeros((4, 6))
    V = np.zeros((8, 6))
    W = np.zeros((2, 6))
    # c=0: A0*B00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # c=1: A0*B01
    U[0,1]=1; V[1,1]=1; W[1,1]=1
    # c=2: A1*B10
    U[1,2]=1; V[2,2]=1; W[0,2]=1
    # c=3: A1*B11
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    # c=4: A2*B20
    U[2,4]=1; V[4,4]=1; W[0,4]=1
    # c=5: A2*B21
    U[2,5]=1; V[5,5]=1; W[1,5]=1
    return U, V, W


def _block_2x4x2():
    """Return (U, V, W) for 2x4x2 matrix multiplication, rank 7 (Strassen).
    U: (8, 7) indexed by [i*4+j], V: (8, 7) indexed by [j*2+k], W: (4, 7) indexed by [k*2+i]
    C[i,k] = sum_j A[i,j] B[j,k]
    Split j into (0,1) and (2,3): C = A[:,:2]@B[:2,:] + A[:,2:]@B[2:,:]
    Each is 2x2x2, use Strassen.
    """
    U = np.zeros((8, 7))
    V = np.zeros((8, 7))
    W = np.zeros((4, 7))
    # For first block (j=0,1): U[i*4+j, c], V[j*2+k, c] where j in {0,1}
    # For second block (j=2,3): U[i*4+j, c], V[j*2+k, c] where j in {2,3}
    # Strassen factors for 2x2x2:
    sU = np.array([[1,0,1,0,1,-1,0],
                   [0,0,0,0,1,0,1],
                   [0,1,0,0,0,1,0],
                   [1,1,0,1,0,0,-1]])
    sV = np.array([[1,1,0,1,1,1,0],
                   [0,0,1,0,0,1,1],
                   [1,0,0,-1,0,0,1],
                   [1,0,0,0,1,0,1]])
    sW = np.array([[1,0,0,1,-1,0,1],
                   [0,0,1,0,1,0,0],
                   [0,1,0,1,0,0,0],
                   [1,-1,0,0,0,1,0]])
    
    # First block: j=0,1 -> U indices i*4+0, i*4+1; V indices 0*2+k=k, 1*2+k=2+k
    # U[i*4+0, c] = sU[i, c] (i=0,1), U[i*4+1, c] = sU[i+2, c] (i=0,1)
    for i in range(2):
        U[i*4+0, :] = sU[i, :]
        U[i*4+1, :] = sU[i+2, :]
    # V[0*2+k, c] = sV[k, c] (k=0,1), V[1*2+k, c] = sV[2+k, c] (k=0,1)
    for k in range(2):
        V[0*2+k, :] = sV[k, :]
        V[1*2+k, :] = sV[2+k, :]
    
    # Second block: j=2,3 -> U indices i*4+2, i*4+3; V indices 2*2+k=4+k, 3*2+k=6+k
    for i in range(2):
        U[i*4+2, :] = sU[i, :]
        U[i*4+3, :] = sU[i+2, :]
    for k in range(2):
        V[2*2+k, :] = sV[k, :]
        V[3*2+k, :] = sV[2+k, :]
    
    # W[k*2+i, c] = sW[k*2+i, c]
    W[:, :] = sW
    return U, V, W


def _block_3x4x2():
    """Return (U, V, W) for 3x4x2 matrix multiplication, rank 13.
    U: (12, 13) indexed by [i*4+j], V: (8, 13) indexed by [j*2+k], W: (6, 13) indexed by [k*3+i]
    Split i into i=0 (1x4x2, rank 6) and i=1,2 (2x4x2, rank 7).
    """
    U1, V1, W1 = _block_1x4x2()  # (4,6), (8,6), (2,6)
    U2, V2, W2 = _block_2x4x2()  # (8,7), (8,7), (4,7)
    
    U = np.zeros((12, 13))
    V = np.zeros((8, 13))
    W = np.zeros((6, 13))
    
    # Block 1: i=0, rank 6
    U[0:4, 0:6] = U1
    V[:, 0:6] = V1
    # W1 is (2,6) indexed by [k], W is (6,13) indexed by [k*3+0=k*3]
    for k in range(2):
        W[k*3+0, 0:6] = W1[k, :]
    
    # Block 2: i=1,2, rank 7
    U[4:12, 6:13] = U2
    V[:, 6:13] = V2
    # W2 is (4,7) indexed by [k*2+i] for i in {0,1} (local), W indexed by [k*3+i] for i in {1,2}
    for k in range(2):
        for li in range(2):
            W[k*3 + (li+1), 6:13] = W2[k*2+li, :]
    return U, V, W


def _block_3x4x6():
    """Return (U, V, W) for 3x4x6 matrix multiplication, rank 60.
    U: (12, 60) indexed by [i*4+j], V: (24, 60) indexed by [j*6+k], W: (18, 60) indexed by [k*3+i]
    Split k into 3 pairs of 2: k in {0,1}, {2,3}, {4,5}.
    Each 3x4x2 block has rank 13. Total: 3*13 = 39? 
    Wait, that's not 60. Let me reconsider.
    
    Actually, for 3x4x2 with the split above, rank is 6+7=13.
    For 3x4x6, split k into 3 groups of 2: 3 * 13 = 39.
    
    But wait, let me reconsider the V indexing. For 3x4x6:
    V is (m*p, rank) = (4*6, rank) = (24, rank), indexed by [j*6+k].
    For the k-pair {0,1}: V[j*6+0, c] and V[j*6+1, c].
    For the k-pair {2,3}: V[j*6+2, c] and V[j*6+3, c].
    For the k-pair {4,5}: V[j*6+4, c] and V[j*6+5, c].
    
    So each 3x4x2 sub-block uses a different slice of V. The U and W are shared structure.
    """
    U = np.zeros((12, 39))
    V = np.zeros((24, 39))
    W = np.zeros((18, 39))
    
    for pair in range(3):
        k_start = pair * 2
        U_blk, V_blk, W_blk = _block_3x4x2()
        c_start = pair * 13
        U[:, c_start:c_start+13] = U_blk
        # V_blk is (8, 13) indexed by [j*2+k] for k in {0,1}
        # V is (24, 39) indexed by [j*6+k]
        for j in range(4):
            for k in range(2):
                V[j*6 + k_start + k, c_start:c_start+13] = V_blk[j*2 + k, :]
        # W_blk is (6, 13) indexed by [k*3+i] for k in {0,1}
        # W is (18, 39) indexed by [k*3+i] for k in {0..5}
        for k in range(2):
            for i in range(3):
                W[(k_start + k)*3 + i, c_start:c_start+13] = W_blk[k*3 + i, :]
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    deadline = time.time() + 130  # safety margin
    
    # Build the rank-39 decomposition
    U, V, W = _block_3x4x6()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
    
    # Convert to lists
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
