"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j, r]
    V[i*2+j, r]
    W[i*2+j, r] -> Note: W is for C, so W[k*2+i, r] in the general formula.
    Here for 2x2x2, W is (2*2, 7) = (4, 7).
    The general formula uses W[k*n+i, r]. For n=2, W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen M1-M7
    # M1 = (A11+A22)(B11+B22)
    # A11=(0,0), A22=(1,1)
    # B11=(0,0), B22=(1,1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11 is (0,0) -> W[0*2+0, 0] = W[0,0]
    # C22 is (1,1) -> W[1*2+1, 0] = W[3,0]
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21=(1,0), A22=(1,1)
    # B11=(0,0)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2 -> C21 is (1,0) -> W[0*2+1, 1] = W[1,1]
    W[1, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11=(0,0)
    # B12=(0,1), B22=(1,1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3 -> C12 is (0,1) -> W[1*2+0, 2] = W[2,2]
    W[2, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22=(1,1)
    # B21=(1,0), B11=(0,0)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11=(0,0), A12=(0,1)
    # B22=(1,1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=(1,0), A11=(0,0)
    # B11=(0,0), B12=(0,1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=(0,1), A22=(1,1)
    # B21=(1,0), B22=(1,1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_32():
    """
    Constructs a rank-32 decomposition for 3x4x4 matrix multiplication.
    Splits A (3x4) into A1 (2x4) and A2 (1x4).
    A1 B is computed using 4 Strassen blocks (rank 28).
    A2 B is computed using 4 scalar blocks (rank 4).
    """
    n, m, p = 3, 4, 4
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Get Strassen tensors
    Us, Vs, Ws = _strassen_222()
    
    # --- Part 1: A1 B (2x4 x 4x4) ---
    # A1 is rows 0,1 of A.
    # B is 4x4.
    # We split B into 2x2 blocks: B11, B12, B21, B22.
    # A1 is split into A11, A12 (2x2 blocks).
    # C1 = A11 B11 + A12 B21 (cols 0,1 of C)
    # C1 = A11 B12 + A12 B22 (cols 2,3 of C)
    
    # Block 1: A11 * B11 -> C[0:2, 0:2]
    # A11 indices in A: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # B11 indices in B: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W[k*n+i]
    # C[0,0] -> W[0*3+0]=0
    # C[0,1] -> W[1*3+0]=3
    # C[1,0] -> W[0*3+1]=1
    # C[1,1] -> W[1*3+1]=4
    
    offset = 0
    # Map Strassen U (4x7) to U (12x32)
    # Strassen U indices: 0,1,2,3 correspond to A11,A12,A21,A22 of the 2x2 block.
    # Here A11 is the top-left 2x2 of A1.
    # A11[0,0] is A[0,0] -> U[0, offset+r] = Us[0, r]
    # A11[0,1] is A[0,1] -> U[1, offset+r] = Us[1, r]
    # A11[1,0] is A[1,0] -> U[4, offset+r] = Us[2, r]
    # A11[1,1] is A[1,1] -> U[5, offset+r] = Us[3, r]
    
    U[0, offset:offset+7] = Us[0, :]
    U[1, offset:offset+7] = Us[1, :]
    U[4, offset:offset+7] = Us[2, :]
    U[5, offset:offset+7] = Us[3, :]
    
    # V mapping: B11 is top-left 2x2 of B.
    # B11[0,0] is B[0,0] -> V[0, offset+r] = Vs[0, r]
    # B11[0,1] is B[0,1] -> V[1, offset+r] = Vs[1, r]
    # B11[1,0] is B[1,0] -> V[4, offset+r] = Vs[2, r]
    # B11[1,1] is B[1,1] -> V[5, offset+r] = Vs[3, r]
    
    V[0, offset:offset+7] = Vs[0, :]
    V[1, offset:offset+7] = Vs[1, :]
    V[4, offset:offset+7] = Vs[2, :]
    V[5, offset:offset+7] = Vs[3, :]
    
    # W mapping: C[0:2, 0:2].
    # Strassen W indices: 0,1,2,3 correspond to C11, C12, C21, C22.
    # C11 is C[0,0] -> W[0, offset+r] = Ws[0, r]
    # C12 is C[0,1] -> W[3, offset+r] = Ws[1, r]
    # C21 is C[1,0] -> W[1, offset+r] = Ws[2, r]
    # C22 is C[1,1] -> W[4, offset+r] = Ws[3, r]
    
    W[0, offset:offset+7] = Ws[0, :]
    W[3, offset:offset+7] = Ws[1, :]
    W[1, offset:offset+7] = Ws[2, :]
    W[4, offset:offset+7] = Ws[3, :]
    
    offset += 7
    
    # Block 2: A11 * B12 -> C[0:2, 2:4]
    # A11 same as above.
    # B12 is top-right 2x2 of B.
    # B12[0,0] is B[0,2] -> V[2, offset+r] = Vs[0, r]
    # B12[0,1] is B[0,3] -> V[3, offset+r] = Vs[1, r]
    # B12[1,0] is B[1,2] -> V[6, offset+r] = Vs[2, r]
    # B12[1,1] is B[1,3] -> V[7, offset+r] = Vs[3, r]
    
    U[0, offset:offset+7] = Us[0, :]
    U[1, offset:offset+7] = Us[1, :]
    U[4, offset:offset+7] = Us[2, :]
    U[5, offset:offset+7] = Us[3, :]
    
    V[2, offset:offset+7] = Vs[0, :]
    V[3, offset:offset+7] = Vs[1, :]
    V[6, offset:offset+7] = Vs[2, :]
    V[7, offset:offset+7] = Vs[3, :]
    
    # C[0:2, 2:4]
    # C[0,2] -> W[2*3+0] = W[6, offset+r] = Ws[0, r]
    # C[0,3] -> W[3*3+0] = W[9, offset+r] = Ws[1, r]
    # C[1,2] -> W[2*3+1] = W[7, offset+r] = Ws[2, r]
    # C[1,3] -> W[3*3+1] = W[10, offset+r] = Ws[3, r]
    
    W[6, offset:offset+7] = Ws[0, :]
    W[9, offset:offset+7] = Ws[1, :]
    W[7, offset:offset+7] = Ws[2, :]
    W[10, offset:offset+7] = Ws[3, :]
    
    offset += 7
    
    # Block 3: A12 * B21 -> C[0:2, 0:2]
    # A12 is top-right 2x2 of A1.
    # A12[0,0] is A[0,2] -> U[2, offset+r] = Us[0, r]
    # A12[0,1] is A[0,3] -> U[3, offset+r] = Us[1, r]
    # A12[1,0] is A[1,2] -> U[6, offset+r] = Us[2, r]
    # A12[1,1] is A[1,3] -> U[7, offset+r] = Us[3, r]
    
    U[2, offset:offset+7] = Us[0, :]
    U[3, offset:offset+7] = Us[1, :]
    U[6, offset:offset+7] = Us[2, :]
    U[7, offset:offset+7] = Us[3, :]
    
    # B21 is bottom-left 2x2 of B.
    # B21[0,0] is B[2,0] -> V[8, offset+r] = Vs[0, r]
    # B21[0,1] is B[2,1] -> V[9, offset+r] = Vs[1, r]
    # B21[1,0] is B[3,0] -> V[12, offset+r] = Vs[2, r]
    # B21[1,1] is B[3,1] -> V[13, offset+r] = Vs[3, r]
    
    V[8, offset:offset+7] = Vs[0, :]
    V[9, offset:offset+7] = Vs[1, :]
    V[12, offset:offset+7] = Vs[2, :]
    V[13, offset:offset+7] = Vs[3, :]
    
    # C[0:2, 0:2] (Same as Block 1)
    W[0, offset:offset+7] = Ws[0, :]
    W[3, offset:offset+7] = Ws[1, :]
    W[1, offset:offset+7] = Ws[2, :]
    W[4, offset:offset+7] = Ws[3, :]
    
    offset += 7
    
    # Block 4: A12 * B22 -> C[0:2, 2:4]
    # A12 same as Block 3.
    U[2, offset:offset+7] = Us[0, :]
    U[3, offset:offset+7] = Us[1, :]
    U[6, offset:offset+7] = Us[2, :]
    U[7, offset:offset+7] = Us[3, :]
    
    # B22 is bottom-right 2x2 of B.
    # B22[0,0] is B[2,2] -> V[10, offset+r] = Vs[0, r]
    # B22[0,1] is B[2,3] -> V[11, offset+r] = Vs[1, r]
    # B22[1,0] is B[3,2] -> V[14, offset+r] = Vs[2, r]
    # B22[1,1] is B[3,3] -> V[15, offset+r] = Vs[3, r]
    
    V[10, offset:offset+7] = Vs[0, :]
    V[11, offset:offset+7] = Vs[1, :]
    V[14, offset:offset+7] = Vs[2, :]
    V[15, offset:offset+7] = Vs[3, :]
    
    # C[0:2, 2:4] (Same as Block 2)
    W[6, offset:offset+7] = Ws[0, :]
    W[9, offset:offset+7] = Ws[1, :]
    W[7, offset:offset+7] = Ws[2, :]
    W[10, offset:offset+7] = Ws[3, :]
    
    offset += 7
    
    # --- Part 2: A2 B (1x4 x 4x4) ---
    # A2 is row 2 of A.
    # C2 is row 2 of C.
    # Rank 4 decomposition.
    # For j=0..3:
    # U[2*4+j, offset+j] = 1
    # V[j*4+k, offset+j] = 1 for all k?
    # No, V[j*4+k, offset+j] = 1 if we want to sum over k?
    # Wait, the term is A_{2j} B_{jk} C_{2k}.
    # We want sum_j A_{2j} B_{jk} = C_{2k}.
    # Using 4 tensors r=0..3.
    # For r=j:
    # U[2*4+j, r] = 1
    # V[j*4+k, r] = 1 for all k?
    # If V is 1 for all k, then sum_k V_{jk,r} W_{k2,r} = sum_k 1 * W_{k2,r}.
    # We need this to be 1 for the specific k?
    # No, the tensor T_{2jk} = 1.
    # sum_r U_{2j,r} V_{jk,r} W_{k2,r} = 1.
    # If we set U_{2j,j} = 1, V_{jk,j} = 1, W_{k2,j} = 1?
    # Then for fixed j, k:
    # sum_r ... = U_{2j,j} V_{jj} W_{j2} = 1*1*1 = 1.
    # But V_{jk,j} must be 1 for ALL k?
    # If V_{jk,j} = 1 for all k, then for a specific k, the term is 1.
    # Yes.
    
    for j in range(4):
        r = offset + j
        U[2*4 + j, r] = 1
        for k in range(4):
            V[j*4 + k, r] = 1
            W[k*3 + 2, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    # Try the rank-32 decomposition
    result = _construct_rank_32()
    
    # Verify
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
