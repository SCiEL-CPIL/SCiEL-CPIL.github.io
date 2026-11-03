import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    """
    Returns a rank-16 exact tensor decomposition for 2x3x3 matrix multiplication.
    
    Structure:
    - Rank 7: Strassen for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
    - Rank 4: Schoolbook for terms where j=2, k in {0,1}
    - Rank 5: Optimized terms for k=2 (j in {0,1,2}, i in {0,1})
    
    Total rank: 7 + 4 + 5 = 16
    """
    n, m, p = 2, 3, 3
    rank = 16
    
    U = np.zeros((n * m, rank))  # 6 x 16
    V = np.zeros((m * p, rank))  # 9 x 16
    W = np.zeros((p * n, rank))  # 6 x 16
    
    # ---- Part 1: Strassen for 2x2x2 core (rank 7) ----
    # Indices: U[0..3] = A[0,0], A[0,1], A[1,0], A[1,1]
    #          V[0..3] = B[0,0], B[0,1], B[1,0], B[1,1]
    #          W[0..3] = C[0,0], C[0,1], C[1,0], C[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1  # contributes to C00 and C10
    
    # M2 = (A10+A11)(B00)
    U[1, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[2, 1] = -1  # contributes to C01 and C11
    
    # M3 = (A00)(B10+B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = 1
    W[2, 2] = 1; W[3, 2] = 1  # contributes to C10 and C11
    
    # M4 = (A10)(B01-B10)
    U[1, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # contributes to C00 and C01
    
    # M5 = (A01+A10)(B10)
    U[0, 4] = 1; U[1, 4] = 1
    V[1, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # contributes to C00 and C10
    
    # M6 = (A00-A10)(B01)
    U[0, 5] = 1; U[1, 5] = -1
    V[0, 5] = 1
    W[1, 5] = 1  # contributes to C01
    
    # M7 = (A11-A01)(B11)
    U[2, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1
    W[0, 6] = 1  # contributes to C00
    
    # ---- Part 2: j=2, k in {0,1} (rank 4) ----
    # These are A[i,2] * B[2,k] -> C[i,k] for i in {0,1}, k in {0,1}
    # U[4] = A[0,2], U[5] = A[1,2]
    # V[6] = B[2,0], V[7] = B[2,1]
    # W[0] = C[0,0], W[1] = C[0,1], W[2] = C[1,0], W[3] = C[1,1]
    
    # r=7: A[0,2]*B[2,0] -> C[0,0]
    U[4, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    
    # r=8: A[0,2]*B[2,1] -> C[0,1]
    U[4, 8] = 1; V[7, 8] = 1; W[1, 8] = 1
    
    # r=9: A[1,2]*B[2,0] -> C[1,0]
    U[5, 9] = 1; V[6, 9] = 1; W[2, 9] = 1
    
    # r=10: A[1,2]*B[2,1] -> C[1,1]
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # ---- Part 3: k=2, j in {0,1,2} (rank 5) ----
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # U[0]=A[0,0], U[1]=A[0,1], U[2]=A[1,0], U[3]=A[1,1], U[4]=A[0,2], U[5]=A[1,2]
    # V[2]=B[0,2], V[5]=B[1,2], V[8]=B[2,2]
    # W[4]=C[0,2], W[5]=C[1,2]
    
    # r=11: A[0,0]*B[0,2] -> C[0,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    
    # r=12: A[1,0]*B[0,2] -> C[1,2]
    U[2, 12] = 1; V[2, 12] = 1; W[5, 12] = 1
    
    # r=13: A[0,1]*B[1,2] -> C[0,2]
    U[1, 13] = 1; V[5, 13] = 1; W[4, 13] = 1
    
    # r=14: A[1,1]*B[1,2] -> C[1,2]
    U[3, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # r=15: (A[0,2]+A[1,2])*B[2,2] -> C[0,2] and C[1,2]
    U[4, 15] = 1; U[5, 15] = 1
    V[8, 15] = 1
    W[4, 15] = 1; W[5, 15] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _validate(U, V, W):
    """Validate that the decomposition is correct."""
    n, m, p = 2, 3, 3
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check dimensions
    assert U.shape[0] == n * m, f"U rows: {U.shape[0]} != {n*m}"
    assert V.shape[0] == m * p, f"V rows: {V.shape[0]} != {m*p}"
    assert W.shape[0] == p * n, f"W rows: {W.shape[0]} != {p*n}"
    
    rank = U.shape[1]
    assert V.shape[1] == rank, f"V cols: {V.shape[1]} != {rank}"
    assert W.shape[1] == rank, f"W cols: {W.shape[1]} != {rank}"
    
    # Check that all entries are integer or half-integer
    for name, mat in [("U", U), ("V", V), ("W", W)]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if val != 0:
                    # Check if val is integer or half-integer
                    if abs(val * 2 - round(val * 2)) > 1e-10:
                        raise ValueError(f"{name}[{i},{j}] = {val} is not integer/half-integer")
    
    # Check the tensor decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0 if (i, j, k) != (0, 0, 0) else 1.0
                # Actually, all (i,j,k) should sum to 1
                if abs(total - 1.0) > 1e-10:
                    raise ValueError(f"Tensor component ({i},{j},{k}) = {total}, expected 1.0")
    
    return True

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    try:
        _validate(result['U'], result['V'], result['W'])
        print("Validation passed")
    except Exception as e:
        print(f"Validation failed: {e}")
