"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (4,7) maps A entries [a11,a12,a21,a22] to 7 multipliers
    V: (4,7) maps B entries [b11,b12,b21,b22] to 7 multipliers  
    W: (4,7) maps 7 multipliers to C entries [c11,c12,c21,c22]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (a21+a22)b11 -> c21 += M2, c22 -= M2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = a11(b12-b22) -> c12 += M3, c22 += M3
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = a22(b21-b11) -> c11 += M4, c21 += M4
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (a11+a12)b22 -> c11 -= M5, c12 += M5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> c22 += M6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> c11 += M7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split 4x4x9 into 4x4x4 + 4x4x4 + 4x4x1
    # Use Strassen for each 4x4x4 (rank 49 each), schoolbook for 4x4x1 (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    sU, sV, sW = _strassen_2x2x2()
    
    def build_4x4x4():
        """Build 4x4x4 decomposition using Strassen at 4x4 level (rank 49)."""
        total_rank = 49
        U = np.zeros((16, total_rank))
        V = np.zeros((16, total_rank))
        W = np.zeros((16, total_rank))
        
        # A block indices in flattened 4x4 (row-major):
        # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
        # B same layout, C same layout
        
        A11 = [0,1,4,5]
        A12 = [2,3,6,7]
        A21 = [8,9,12,13]
        A22 = [10,11,14,15]
        B11 = [0,1,4,5]
        B12 = [2,3,6,7]
        B21 = [8,9,12,13]
        B22 = [10,11,14,15]
        C11 = [0,1,4,5]
        C12 = [2,3,6,7]
        C21 = [8,9,12,13]
        C22 = [10,11,14,15]
        
        # M1 = (A11+A22)(B11+B22)
        for col in range(7):
            for idx, flat in enumerate(A11):
                U[flat, col] += sU[idx, col]
            for idx, flat in enumerate(A22):
                U[flat, col] += sU[idx, col]
            for idx, flat in enumerate(B11):
                V[flat, col] += sV[idx, col]
            for idx, flat in enumerate(B22):
                V[flat, col] += sV[idx, col]
            for idx, flat in enumerate(C11):
                W[flat, col] += sW[idx, col]
            for idx, flat in enumerate(C22):
                W[flat, col] += sW[idx, col]
        
        # M2 = (A21+A22)B11
        for col in range(7, 14):
            sc = col - 7
            for idx, flat in enumerate(A21):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(A22):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(B11):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(C21):
                W[flat, col] += sW[idx, sc]
            for idx, flat in enumerate(C22):
                W[flat, col] += -sW[idx, sc]
        
        # M3 = A11(B12-B22)
        for col in range(14, 21):
            sc = col - 14
            for idx, flat in enumerate(A11):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(B12):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(B22):
                V[flat, col] += -sV[idx, sc]
            for idx, flat in enumerate(C12):
                W[flat, col] += sW[idx, sc]
            for idx, flat in enumerate(C22):
                W[flat, col] += sW[idx, sc]
        
        # M4 = A22(B21-B11)
        for col in range(21, 28):
            sc = col - 21
            for idx, flat in enumerate(A22):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(B21):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(B11):
                V[flat, col] += -sV[idx, sc]
            for idx, flat in enumerate(C11):
                W[flat, col] += sW[idx, sc]
            for idx, flat in enumerate(C21):
                W[flat, col] += sW[idx, sc]
        
        # M5 = (A11+A12)B22
        for col in range(28, 35):
            sc = col - 28
            for idx, flat in enumerate(A11):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(A12):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(B22):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(C11):
                W[flat, col] += -sW[idx, sc]
            for idx, flat in enumerate(C12):
                W[flat, col] += sW[idx, sc]
        
        # M6 = (A21-A11)(B11+B12)
        for col in range(35, 42):
            sc = col - 35
            for idx, flat in enumerate(A21):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(A11):
                U[flat, col] += -sU[idx, sc]
            for idx, flat in enumerate(B11):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(B12):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(C22):
                W[flat, col] += sW[idx, sc]
        
        # M7 = (A12-A22)(B21+B22)
        for col in range(42, 49):
            sc = col - 42
            for idx, flat in enumerate(A12):
                U[flat, col] += sU[idx, sc]
            for idx, flat in enumerate(A22):
                U[flat, col] += -sU[idx, sc]
            for idx, flat in enumerate(B21):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(B22):
                V[flat, col] += sV[idx, sc]
            for idx, flat in enumerate(C11):
                W[flat, col] += sW[idx, sc]
        
        return U, V, W
    
    def build_4x4x1():
        """Build 4x4x1 decomposition (rank 16)."""
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        for i in range(4):
            for j in range(4):
                col = i * 4 + j
                U[i * 4 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return U, V, W
    
    # Build components
    U1, V1, W1 = build_4x4x4()
    U2, V2, W2 = build_4x4x4()
    U3, V3, W3 = build_4x4x1()
    
    r1, r2, r3 = U1.shape[1], U2.shape[1], U3.shape[1]
    total_rank = r1 + r2 + r3
    
    # Assemble
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((16, total_rank))
    W_full = np.zeros((36, total_rank))
    
    # Block 1: columns 0-3 of output
    U_full[:, :r1] = U1
    V_full[:, :r1] = V1
    W_full[:16, :r1] = W1  # C entries for columns 0-3: W[k*4+i] for k=0,1,2,3
    
    # Block 2: columns 4-7 of output
    U_full[:, r1:r1+r2] = U2
    V_full[:, r1:r1+r2] = V2
    W_full[16:32, r1:r1+r2] = W2  # k=4,5,6,7
    
    # Block 3: column 8 of output
    U_full[:, r1+r2:] = U3
    V_full[:, r1+r2:] = V3
    W_full[32:36, r1+r2:] = W3  # k=8, i=0,1,2,3
    
    # Verify
    if time.time() < deadline:
        # Check that U^T V W^T gives the correct tensor
        # Tensor T[i,j,k] = 1 for all valid i,j,k
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: U indexed by [i*m+j], V by [j*p+k], W by [k*n+i]
        
        # Spot check a few entries
        valid = True
        for i in [0, 1, 3]:
            for j in [0, 2, 3]:
                for k in [0, 4, 8]:
                    expected = 1.0
                    actual = 0.0
                    for r in range(total_rank):
                        actual += U_full[i*4+j, r] * V_full[j*9+k, r] * W_full[k*4+i, r]
                    if abs(actual - expected) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U_fb = [[0] * rank for _ in range(n * m)]
            V_fb = [[0] * rank for _ in range(m * p)]
            W_fb = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to integer lists
    U_list = [[int(round(x)) for x in row] for row in U_full]
    V_list = [[int(round(x)) for x in row] for row in V_full]
    W_list = [[int(round(x)) for x in row] for row in W_full]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
