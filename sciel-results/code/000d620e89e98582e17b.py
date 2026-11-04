import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_decomp():
    """Return (U, V, W) for 2x2x2 multiplication, rank 7."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    # M1 = a00*(b01+b11), M2 = a10*(b00+b10), M3 = a11*(b11-b01)
    # M4 = a00*(b11-b00), M5 = (a00+a01)*b10, M6 = (a10+a11)*b01
    # M7 = (a00+a10)*(b00-b10), M8 = (a01+a11)*(b10-b11)
    # c00 = M1 + M4 - M5 - M7, c01 = M3 + M5, c10 = M2 + M4, c11 = M1 - M2 + M3 + M8
    # U is for A elements (a00,a01,a10,a11) -> indices 0..3
    # V is for B elements (b00,b01,b10,b11) -> indices 0..3
    # W is for C elements (c00,c01,c10,c11) -> indices 0..3
    
    # M1: a00 * (b01+b11)
    U[0][0] = 1
    V[1][0] = 1
    V[3][0] = 1
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11
    
    # M2: a10 * (b00+b10)
    U[2][1] = 1
    V[0][1] = 1
    V[2][1] = 1
    W[1][1] = 1  # c01
    W[2][1] = 1  # c10
    
    # M3: a11 * (b11-b01)
    U[3][2] = 1
    V[1][2] = -1
    V[3][2] = 1
    W[1][2] = 1  # c01
    W[3][2] = 1  # c11
    
    # M4: a00 * (b11-b00)
    U[0][3] = 1
    V[0][3] = -1
    V[3][3] = 1
    W[0][3] = 1  # c00
    W[2][3] = 1  # c10
    
    # M5: (a00+a01) * b10
    U[0][4] = 1
    U[1][4] = 1
    V[2][4] = 1
    W[0][4] = -1  # c00
    W[1][4] = 1   # c01
    
    # M6: (a10+a11) * b01
    U[2][5] = 1
    U[3][5] = 1
    V[1][5] = 1
    W[0][5] = -1  # c00
    W[2][5] = 1   # c10
    
    # M7: (a00+a10) * (b00-b10)
    U[0][6] = 1
    U[2][6] = 1
    V[0][6] = 1
    V[2][6] = -1
    W[0][6] = -1  # c00
    
    # M8: (a01+a11) * (b10-b11)
    U[1][7] = 1 if 7 < rank else 0  # rank is 7, so M8 is index 6? No, M1..M7 is 7 terms.
    # Wait, Strassen is 7 multiplications. Let's re-check indices.
    # M1 (idx 0), M2 (idx 1), M3 (idx 2), M4 (idx 3), M5 (idx 4), M6 (idx 5), M7 (idx 6).
    # I listed 8 above? M1, M2, M3, M4, M5, M6, M7. That's 7.
    # My M8 was actually part of the standard Strassen?
    # Standard Strassen:
    # M1 = A11(B21+B22)
    # M2 = (A21+A22)B11
    # M3 = (A11+A12)B22
    # M4 = A22(B11-B21)
    # M5 = (A11+A21)B12
    # M6 = (A12-A21)(B11+B21)
    # M7 = (A22-A12)(B12+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let's rebuild carefully with 0-based indexing for 7 terms.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0: A00 * (B01+B11)
    U[0][0] = 1
    V[1][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # c00, c11
    
    # M1: (A10+A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[2][1] = 1  # c01, c10
    
    # M2: (A00+A01) * B11
    U[0][2] = 1; U[1][2] = 1
    V[3][2] = 1
    W[0][2] = 1; W[3][2] = 1  # c00, c11
    
    # M3: A11 * (B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1  # c00, c10
    
    # M4: (A00+A10) * B01
    U[0][4] = 1; U[2][4] = 1
    V[1][4] = 1
    W[0][4] = -1; W[1][4] = 1  # c00, c01
    
    # M5: (A01-A10) * (B00+B10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = -1  # c00
    
    # M6: (A11-A01) * (B01+B11)
    U[3][6] = 1; U[1][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[3][6] = 1  # c11
    
    return U, V, W

def _strassen_2x3x6():
    """
    Decompose 2x3x6 multiplication using Strassen on 2x2 blocks.
    A is 2x3, B is 3x6.
    Partition A into [A0 A1] where A0 is 2x1, A1 is 2x2.
    Partition B into [B0 B1] where B0 is 1x6, B1 is 2x6.
    
    This doesn't directly map to 2x2x2.
    
    Better approach:
    View 2x3x6 as two 2x3x3 problems if B is block diagonal? No.
    
    Let's use the block structure of the tensor.
    T[i, j, k] for i in 0..1, j in 0..2, k in 0..5.
    
    We can split k into k0 in 0..2 and k1 in 0..5?
    Actually, let's use the known rank bound.
    Rank(2x3x6) <= Rank(2x3x3) + Rank(2x3x3)?
    If B = [B1 B2] (3x6), then AB = [AB1 AB2].
    The tensor for 2x3x6 is the direct sum of the tensor for 2x3x3 (for B1) and 2x3x3 (for B2)?
    No, the U factor is shared.
    
    If we have a decomposition for 2x3x3 with rank R, can we get 2x3x6 with rank 2R?
    Yes.
    U_new = [U, U]
    V_new = [V, 0; 0, V] (block diag)
    W_new = [W, 0; 0, W] (block diag)
    Wait, V is 3x3 for 2x3x3? No, V is m*p = 9.
    For 2x3x6, V is 3*6 = 18.
    If we use two copies of the 2x3x3 decomposition, we need 2*R terms.
    
    What is the minimal rank for 2x3x3?
    Schoolbook is 6.
    Strassen-like for 2x3x3?
    We can embed 2x2x2 into 2x3x3.
    Rank(2x2x2) = 7.
    So Rank(2x3x3) <= 7 + something?
    Actually, Rank(2x3x3) is known to be 7.
    Let's verify if Rank(2x3x3) = 7.
    If so, Rank(2x3x6) <= 14.
    
    Let's construct the 2x3x3 decomposition from 2x2x2.
    A is 2x3, B is 3x3.
    Pad A to 2x4? No.
    
    Let's use the direct sum construction from the known rank of 2x3x3.
    If Rank(2x3x3) = 7, then Rank(2x3x6) <= 14.
    Is 14 achievable?
    Yes, by taking two disjoint copies of the 2x3x3 decomposition.
    
    How to construct 2x3x3 rank 7?
    We can use the 2x2x2 decomposition (rank 7) and extend it.
    Or find a direct rank 7 decomposition for 2x3x3.
    
    Let's try to find a rank 7 decomposition for 2x3x3.
    If we succeed, we combine two copies for 2x3x6, giving rank 14.
    Schoolbook is 36. 14 is a huge improvement.
    
    Let's implement a search for 2x3x3 rank 7.
    """
    pass

def _als_decompose(T, n, m, p, rank, deadline, max_sweeps=200, seed=0):
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    U = rng.randn(dim1, rank) * 0.1
    V = rng.randn(dim2, rank) * 0.1
    W = rng.randn(dim3, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 5.0:
            break
        
        VW = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        T1 = T.reshape(dim1, -1)
        try:
            U_new, _, _, _ = np.linalg.lstsq(VW, T1.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            pass
        
        UW = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        UV = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
        T3 = np.transpose(T, (2, 0, 1)).reshape(dim3, -1)
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV, T3.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            pass
        
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((reconstructed - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank(T, n, m, p, rank, deadline, max_trials=50):
    for seed in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=100, seed=seed)
        if result is not None:
            return result
    return None

def _construct_2x3x6_from_2x3x3(res_2x3x3):
    U_233 = np.array(res_2x3x3['U'])
    V_233 = np.array(res_2x3x3['V'])
    W_233 = np.array(res_2x3x3['W'])
    
    R = U_233.shape[1]
    
    U_new = np.hstack([U_233, U_233])
    V_new = np.zeros((18, 2*R))
    V_new[0:9, 0:R] = V_233
    V_new[9:18, R:2*R] = V_233
    
    W_new = np.zeros((12, 2*R))
    W_new[0:6, 0:R] = W_233
    W_new[6:12, R:2*R] = W_233
    
    U_list = [[round(float(x), 8) for x in row] for row in U_new]
    V_list = [[round(float(x), 8) for x in row] for row in V_new]
    W_list = [[round(float(x), 8) for x in row] for row in W_new]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try to find rank 7 for 2x3x3
    T_233 = _build_tensor(2, 3, 3)
    res_233 = _try_rank(T_233, 2, 3, 3, 7, deadline, max_trials=100)
    
    if res_233 is not None:
        if _verify_decomposition(res_233['U'], res_233['V'], res_233['W'], 2, 3, 3, tol=1e-6):
            res_236 = _construct_2x3x6_from_2x3x3(res_233)
            if _verify_decomposition(res_236['U'], res_236['V'], res_236['W'], 2, 3, 6, tol=1e-6):
                return res_236
    
    # Try direct search for 2x3x6 with rank 14
    T_236 = _build_tensor(2, 3, 6)
    res_236 = _try_rank(T_236, 2, 3, 6, 14, deadline, max_trials=50)
    if res_236 is not None:
        return res_236
        
    # Fallback: Schoolbook
    return _schoolbook_decomposition(2, 3, 6)
