"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U shape (4,7), V shape (4,7), W shape (4,7)
    Indexing: U[i*m+j] where n=2,m=2 -> U[i*2+j]
             V[j*p+k] where m=2,p=2 -> V[j*2+k]
             W[k*n+i] where p=2,n=2 -> W[k*2+i]
    """
    rank = 7
    U = np.zeros((4, rank), dtype=object)
    V = np.zeros((4, rank), dtype=object)
    W = np.zeros((4, rank), dtype=object)
    
    # Strassen formulas for C = A * B where A,B are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (a+d)(e+h)  -> U[0]=a, U[3]=d, V[0]=e, V[3]=h
    # Actually let's use the standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map indices:
    # A11 -> U[0*2+0] = U[0]
    # A12 -> U[0*2+1] = U[1]
    # A21 -> U[1*2+0] = U[2]
    # A22 -> U[1*2+1] = U[3]
    
    # B11 -> V[0*2+0] = V[0]
    # B12 -> V[0*2+1] = V[1]
    # B21 -> V[1*2+0] = V[2]
    # B22 -> V[1*2+1] = V[3]
    
    # C11 -> W[0*2+0] = W[0] (k=0, i=0)
    # C12 -> W[0*2+1] = W[1] (k=0, i=1) -- Wait, W[k*n+i]. k=0, i=1 -> W[1]
    # C21 -> W[1*2+0] = W[2] (k=1, i=0)
    # C22 -> W[1*2+1] = W[3] (k=1, i=1)
    
    # Let's double check W indexing: W[k*n+i].
    # For n=2, p=2:
    # k=0, i=0 -> W[0]
    # k=0, i=1 -> W[1]
    # k=1, i=0 -> W[2]
    # k=1, i=1 -> W[3]
    
    # M1:
    # U: a, d  -> U[0]=1, U[3]=1
    # V: e, h  -> V[0]=1, V[3]=1
    # W: C11, C22 -> W[0]=1, W[3]=1
    # But M1 contributes to C11 and C22 with +1.
    # Let's fill column by column.
    
    # Col 0: M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # Col 1: M2 = (A21+A22)B11
    # U: A21, A22 -> U[2]=1, U[3]=1
    # V: B11 -> V[0]=1
    # W: C21, C22 -> W[2]=1, W[3]=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # Col 2: M3 = A11(B21-B22)
    # U: A11 -> U[0]=1
    # V: B21, -B22 -> V[2]=1, V[3]=-1
    # W: C12, C22 -> W[1]=1, W[3]=1
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # Col 3: M4 = A22(B12-B11)
    # U: A22 -> U[3]=1
    # V: B12, -B11 -> V[1]=1, V[0]=-1
    # W: C11, C21 -> W[0]=1, W[2]=1
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # Col 4: M5 = (A11+A12)B22
    # U: A11, A12 -> U[0]=1, U[1]=1
    # V: B22 -> V[3]=1
    # W: C11, C12 -> W[0]=-1, W[1]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # Col 5: M6 = (A21-A11)(B11+B12)
    # U: A21, -A11 -> U[2]=1, U[0]=-1
    # V: B11, B12 -> V[0]=1, V[1]=1
    # W: C22 -> W[3]=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # Col 6: M7 = (A12-A22)(B21+B22)
    # U: A12, -A22 -> U[1]=1, U[3]=-1
    # V: B21, B22 -> V[2]=1, V[3]=1
    # W: C11 -> W[0]=1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, i_offset=0, j_offset=0, k_offset=0):
    """
    Combine two decompositions.
    U1, V1, W1 are for (n1, m1, p1)
    U2, V2, W2 are for (n2, m2, p2)
    We assume they are block diagonal in i, j, k respectively.
    i_offset is for U's i-index and W's i-index.
    j_offset is for U's j-index and V's j-index.
    k_offset is for V's k-index and W's k-index.
    
    Actually, let's just build the full matrices directly in construct() to avoid complex offset logic.
    """
    pass

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    
    n, m, p = DIMENSIONS
    # Hybrid: 1x4x8 schoolbook + 2x4x8 Strassen-like
    # 1x4x8 schoolbook: rank 32
    # 2x4x8 Strassen:
    #   Split m=4 into 2+2
    #   2x2x8 Strassen:
    #     Split p=8 into 4+4
    #     2x2x4 Strassen:
    #       Split p=4 into 2+2
    #       2x2x2 Strassen: rank 7
    #       So 2x2x4: 2 * 7 = 14? No.
    #       Wait, 2x2x4 is two 2x2x2 blocks. So rank 14.
    #       Then 2x2x8 is two 2x2x4 blocks. So rank 28.
    #       Then 2x4x8 is two 2x2x8 blocks. So rank 56.
    # Total rank: 32 + 56 = 88.
    
    # Let's implement the recursive Strassen for 2 x 2^a x 2^b.
    # Base: 2x2x2 -> Strassen (rank 7)
    # Recursion:
    #   If m > 2: split m into two halves. (2 x m/2 x p) + (2 x m/2 x p)
    #   Else if p > 2: split p into two halves. (2 x m x p/2) + (2 x m x p/2)
    #   Else: base case 2x2x2.
    
    def build_strassen_2x2x2(i_off, j_off, k_off):
        # Returns U, V, W for the 2x2x2 block at offsets
        U, V, W = _strassen_222()
        # Apply offsets to indices
        # U[i*2+j, col] -> U[(i+i_off)*2 + (j+j_off), col]
        # V[j*2+k, col] -> V[(j+j_off)*2 + (k+k_off), col]
        # W[k*2+i, col] -> W[(k+k_off)*2 + (i+i_off), col]
        U_new = np.zeros_like(U)
        V_new = np.zeros_like(V)
        W_new = np.zeros_like(W)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_new[(i+i_off)*2 + (j+j_off), c] = U[i*2+j, c]
            for j in range(2):
                for k in range(2):
                    V_new[(j+j_off)*2 + (k+k_off), c] = V[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    W_new[(k+k_off)*2 + (i+i_off), c] = W[k*2+i, c]
        return U_new, V_new, W_new

    def build_strassen(n, m, p, i_off, j_off, k_off):
        # n must be 2.
        if m == 2 and p == 2:
            return build_strassen_2x2x2(i_off, j_off, k_off)
        
        U_list = []
        V_list = []
        W_list = []
        
        if m > 2:
            m1 = m // 2
            U1, V1, W1 = build_strassen(n, m1, p, i_off, j_off, k_off)
            U2, V2, W2 = build_strassen(n, m1, p, i_off, j_off + m1, k_off)
            U_list.extend([U1, U2])
            V_list.extend([V1, V2])
            W_list.extend([W1, W2])
        else: # p > 2
            p1 = p // 2
            U1, V1, W1 = build_strassen(n, m, p1, i_off, j_off, k_off)
            U2, V2, W2 = build_strassen(n, m, p1, i_off, j_off, k_off + p1)
            U_list.extend([U1, U2])
            V_list.extend([V1, V2])
            W_list.extend([W1, W2])
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        return U, V, W

    # 1x4x8 schoolbook
    U1, V1, W1 = _schoolbook(1, 4, 8)
    
    # 2x4x8 Strassen
    U2, V2, W2 = build_strassen(2, 4, 8, i_off=1, j_off=0, k_off=0)
    # Wait, the 2x4x8 block corresponds to i in [1,2] (since 1x4x8 took i=0).
    # So i_off=1.
    
    U = np.vstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    rank = U.shape[1]
    
    # Convert to lists of Fractions or integers as required?
    # "integer/half-integer entries"
    # Let's just return lists of floats or ints. The prompt says "integer/half-integer entries".
    # I will convert to lists of ints/floats.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
