import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    U: (n*m, rank) - coefficients for A
    V: (m*p, rank) - coefficients for B  
    W: (p*n, rank) - coefficients for C (trace convention W[k*n+i, r])
    
    The tensor is T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r]
    and must equal sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] for each (i,j,k)
    which should be 1 if j is the shared index (i.e., the standard matmul tensor).
    
    Actually, the matrix multiplication tensor M[i,j,k] = 1 if j==j (always true in our case? No)
    Wait: for multiplying A (n x m) by B (m x p) to get C (n x p):
    C[i,k] = sum_j A[i,j] B[j,k]
    
    So the tensor T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    Let me re-think. The matrix multiplication tensor is a 3-tensor T of shape (n, m, p) where
    T[i,j,k] = 1 if the index j is the "middle" index. Actually, the standard formulation:
    
    The tensor M_matmul[i,j,k] = 1 for all valid i,j,k? No.
    
    Actually, the matrix multiplication tensor T has entries T[i,j,k] = 1 if we're computing
    C[i,k] which involves A[i,j] and B[j,k]. So T[i,j,k] = 1 for all i, j, k? That doesn't make sense.
    
    Let me reconsider. The tensor decomposition of matrix multiplication:
    We want to write the bilinear map (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    As a tensor, this is a 3-tensor T of shape (n*p, m, m) or similar. But in the standard
    "tensor rank" formulation for matrix multiplication:
    
    The tensor T is of shape (n*m, m*p, p*n) where:
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j', i==i', k==k'
    
    So T is the "identity" in the middle dimension. A rank-r decomposition gives:
    T[a,b,c] = sum_{r=1}^rank U[a,r] * V[b,r] * W[c,r]
    
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    And the condition is:
    U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] summed over r should equal
    1 if j==j' and i==i' and k==k', else 0.
    """
    rank = len(U[0])
    n, m, p = n, m, p
    
    # Build the target tensor: T[i,j,k] = 1 if j==j (trivially true), but actually:
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if j==j', i==i', k==k'
    
    # Verify by checking all entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                            
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _strassen_2x2_decomposition():
    """Return Strassen 2x2x2 decomposition.
    
    Returns U, V, W as lists of lists.
    U: shape (4, 7) - for A indices (i,j) flattened to i*2+j
    V: shape (4, 7) - for B indices (j,k) flattened to j*2+k
    W: shape (4, 7) - for C indices (k,i) flattened to k*2+i (trace convention)
    """
    # Strassen formulas:
    # M1 = (A11 - A10)(B00 + B01)
    # M2 = (A10 + A11)(B10 + B11)
    # M3 = (A00 + A10)(B00)
    # M4 = (A11 + A01)(B11)
    # M5 = (A00 - A10)(B01 + B11)
    # M6 = (A01 + A11)(B10 - B00)
    # M7 = (A00 + A01)(B00 - B10)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5 - M1 - M3
    # C10 = M3 + M6 - M4 - M7
    # C11 = M1 + M2 - M6 + M3
    #
    # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B indices: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C indices: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    #
    # U row index: i*2+j
    # V row index: j*2+k
    # W row index: k*2+i (trace convention: W[k*n+i, r] with n=2)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 - A10)(B00 + B01)
    # A11 = U[3], A10 = U[2]
    # B00 = V[0], B01 = V[1]
    U[3][0] = 1; U[2][0] = -1
    V[0][0] = 1; V[1][0] = 1
    # C00 += M1, C10 -= M1, C11 += M1
    # C00: k=0,i=0 -> W[0*2+0]=W[0]
    # C10: k=1,i=0 -> W[1*2+0]=W[2]
    # C11: k=1,i=1 -> W[1*2+1]=W[3]
    W[0][0] = 1; W[2][0] = -1; W[3][0] = 1
    
    # M2 = (A10 + A11)(B10 + B11)
    # A10 = U[2], A11 = U[3]
    # B10 = V[2], B11 = V[3]
    U[2][1] = 1; U[3][1] = 1
    V[2][1] = 1; V[3][1] = 1
    # C01 += M2, C11 += M2
    # C01: k=1,i=0 -> W[1*2+0]=W[2]
    # C11: k=1,i=1 -> W[1*2+1]=W[3]
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00 + A10)(B00)
    # A00 = U[0], A10 = U[2]
    # B00 = V[0]
    U[0][2] = 1; U[2][2] = 1
    V[0][2] = 1
    # C10 += M3, C11 += M3
    # C10: k=1,i=0 -> W[2]
    # C11: k=1,i=1 -> W[3]
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = (A11 + A01)(B11)
    # A11 = U[3], A01 = U[1]
    # B11 = V[3]
    U[3][3] = 1; U[1][3] = 1
    V[3][3] = 1
    # C00 += M4, C10 -= M4
    # C00: k=0,i=0 -> W[0]
    # C10: k=1,i=0 -> W[2]
    W[0][3] = 1; W[2][3] = -1
    
    # M5 = (A00 - A10)(B01 + B11)
    # A00 = U[0], A10 = U[2]
    # B01 = V[1], B11 = V[3]
    U[0][4] = 1; U[2][4] = -1
    V[1][4] = 1; V[3][4] = 1
    # C00 -= M5, C01 += M5
    # C00: k=0,i=0 -> W[0]
    # C01: k=1,i=0 -> W[2]
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A01 + A11)(B10 - B00)
    # A01 = U[1], A11 = U[3]
    # B10 = V[2], B00 = V[0]
    U[1][5] = 1; U[3][5] = 1
    V[2][5] = 1; V[0][5] = -1
    # C10 -= M6, C11 -= M6
    # C10: k=1,i=0 -> W[2]
    # C11: k=1,i=1 -> W[3]
    W[2][5] = -1; W[3][5] = -1
    
    # M7 = (A00 + A01)(B00 - B10)
    # A00 = U[0], A01 = U[1]
    # B00 = V[0], B10 = V[2]
    U[0][6] = 1; U[1][6] = 1
    V[0][6] = 1; V[2][6] = -1
    # C00 += M7, C10 -= M7
    # C00: k=0,i=0 -> W[0]
    # C10: k=1,i=0 -> W[2]
    W[0][6] = 1; W[2][6] = -1
    
    return U, V, W

def _build_4x4_from_2x2_strassen():
    """Build 4x4x4 Strassen decomposition from 2x2x2 Strassen.
    
    Split A (4x4), B (4x4), C (4x4) into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    
    C = A*B gives:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Use Strassen for the 2x2x2 block multiplication (where each "element" is 2x2):
    M1 = (A11 - A10)(B00 + B01)
    M2 = (A10 + A11)(B10 + B11)
    M3 = (A00 + A10)(B00)
    M4 = (A11 + A01)(B11)
    M5 = (A00 - A10)(B01 + B11)
    M6 = (A01 + A11)(B10 - B00)
    M7 = (A00 + A01)(B00 - B10)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M2 + M5 - M1 - M3
    C10 = M3 + M6 - M4 - M7
    C11 = M1 + M2 - M6 + M3
    
    Each Mi is a 2x2 matrix product, which we compute using the 2x2x2 Strassen decomposition.
    So total rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # U4: shape (16, 49) - A is 4x4, flattened as i*4+j
    # V4: shape (16, 49) - B is 4x4, flattened as j*4+k
    # W4: shape (16, 49) - C is 4x4, trace convention: k*4+i
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # For each of the 7 Strassen multipliers Mi, we need to compute a 2x2 matrix product.
    # The 2x2 product uses U2, V2, W2 with rank 7.
    
    # A block indices:
    # A00: rows 0-1, cols 0-1 -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # A01: rows 0-1, cols 2-3 -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # A10: rows 2-3, cols 0-1 -> U indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # A11: rows 2-3, cols 2-3 -> U indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # B block indices:
    # B00: rows 0-1, cols 0-1 -> V indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B01: rows 0-1, cols 2-3 -> V indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B10: rows 2-3, cols 0-1 -> V indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # B11: rows 2-3, cols 2-3 -> V indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # C block indices (trace convention: k*4+i):
    # C00: k=0-1, i=0-1 -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C01: k=2-3, i=0-1 -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C10: k=0-1, i=2-3 -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C11: k=2-3, i=2-3 -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # For each Strassen multiplier r (0..6), we have a 2x2x2 sub-problem.
    # The sub-problem has its own U2, V2, W2 (each 4x7).
    # We need to map the sub-problem indices to the full 4x4 indices.
    
    # A sub-block mapping:
    # Sub A[si, sj] where si,sj in {0,1} maps to full A[2*si + si2, 2*sj + sj2]
    # Full U index: (2*si + si2)*4 + (2*sj + sj2)
    # Sub U index: si*2 + sj
    
    # B sub-block mapping:
    # Sub B[sk, sl] where sk,sl in {0,1} maps to full B[2*sk + sk2, 2*sl + sl2]
    # Full V index: (2*sk + sk2)*4 + (2*sl + sl2)
    # Sub V index: sk*2 + sl
    
    # C sub-block mapping (trace convention):
    # Sub C[sk, si] where sk,si in {0,1} maps to full C[2*sk + sk2, 2*si + si2]
    # Full W index: (2*sk + sk2)*4 + (2*si + si2)
    # Sub W index: sk*2 + si
    
    for r in range(7):
        base = r * 7
        for sub_u_idx in range(4):
            si = sub_u_idx // 2
            sj = sub_u_idx % 2
            for si2 in range(2):
                for sj2 in range(2):
                    full_u_idx = (2*si + si2)*4 + (2*sj + sj2)
                    U4[full_u_idx][base + sub_u_idx] = U2[sub_u_idx][r]
        
        for sub_v_idx in range(4):
            sk = sub_v_idx // 2
            sl = sub_v_idx % 2
            for sk2 in range(2):
                for sl2 in range(2):
                    full_v_idx = (2*sk + sk2)*4 + (2*sl + sl2)
                    V4[full_v_idx][base + sub_v_idx] = V2[sub_v_idx][r]
        
        for sub_w_idx in range(4):
            sk = sub_w_idx // 2
            si = sub_w_idx % 2
            for sk2 in range(2):
                for si2 in range(2):
                    full_w_idx = (2*sk + sk2)*4 + (2*si + si2)
                    W4[full_w_idx][base + sub_w_idx] = W2[sub_w_idx][r]
    
    return U4, V4, W4

def _build_4x4x8_from_strassen():
    """Build 4x4x8 decomposition by splitting B into two 4x4 blocks.
    
    A (4x4) * B (4x8) = C (4x8)
    B = [B1 | B2] where B1, B2 are 4x4
    C = [A*B1 | A*B2] = [C1 | C2] where C1, C2 are 4x4
    
    Use 4x4x4 Strassen for both A*B1 and A*B2.
    Total rank = 49 + 49 = 98.
    
    U: shape (16, 98)
    V: shape (32, 98) - B is 4x8, flattened as j*8+k
    W: shape (32, 98) - C is 4x8, trace convention: k*4+i
    """
    U4, V4, W4 = _build_4x4_from_2x2_strassen()
    
    U98 = [[0]*98 for _ in range(16)]
    V98 = [[0]*98 for _ in range(32)]
    W98 = [[0]*98 for _ in range(32)]
    
    # First half: A * B1 where B1 is columns 0-3 of B
    # B1: j*8+k for j in 0-3, k in 0-3
    # B2: j*8+k for j in 0-3, k in 4-7
    
    # C1: k*4+i for k in 0-3, i in 0-3
    # C2: k*4+i for k in 4-7, i in 0-3
    
    for r in range(49):
        # U is the same for both halves
        for i in range(16):
            U98[i][r] = U4[i][r]
            U98[i][r + 49] = U4[i][r]
        
        # V for B1: column r in V4 maps to V98 column r
        # V4[j*4+k] where k in 0-3 -> V98[j*8+k]
        for j in range(4):
            for k in range(4):
                v4_idx = j*4 + k
                v98_idx = j*8 + k
                V98[v98_idx][r] = V4[v4_idx][r]
        
        # V for B2: column r+49 in V98
        # V4[j*4+k] where k in 0-3 -> V98[j*8+(k+4)]
        for j in range(4):
            for k in range(4):
                v4_idx = j*4 + k
                v98_idx = j*8 + (k + 4)
                V98[v98_idx][r + 49] = V4[v4_idx][r]
        
        # W for C1: column r in W4 maps to W98 column r
        # W4[k*4+i] where k in 0-3 -> W98[k*4+i]
        for k in range(4):
            for i in range(4):
                w4_idx = k*4 + i
                w98_idx = k*4 + i
                W98[w98_idx][r] = W4[w4_idx][r]
        
        # W for C2: column r+49 in W98
        # W4[k*4+i] where k in 0-3 -> W98[(k+4)*4+i]
        for k in range(4):
            for i in range(4):
                w4_idx = k*4 + i
                w98_idx = (k + 4)*4 + i
                W98[w98_idx][r + 49] = W4[w4_idx][r]
    
    return U98, V98, W98

def schoolbook_decomposition(n, m, p):
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Best known: Schoolbook (rank 128)
    best_result = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based decomposition for 4x4x8 (rank 98)
    try:
        U98, V98, W98 = _build_4x4x8_from_strassen()
        result_strassen = {"U": U98, "V": V98, "W": W98}
        
        # Verify correctness
        if _verify_decomposition(U98, V98, W98, n, m, p):
            best_result = result_strassen
            best_rank = 98
        else:
            # Strassen decomposition is incorrect, keep schoolbook
            pass
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
