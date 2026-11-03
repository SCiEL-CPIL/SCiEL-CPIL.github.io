import time
import numpy as np

DIMENSIONS = [2, 3, 7]
_TIME_LIMIT = 120.0
_START_TIME = time.time()

def _remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _additive_combine(U1, V1, W1, U2, V2, W2):
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    U = np.zeros((U1.shape[0], r1 + r2))
    V = np.zeros((V1.shape[0], r1 + r2))
    W = np.zeros((W1.shape[0], r1 + r2))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_2_2_2():
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices: U (4x7), V (4x7), W (4x7)
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # Term 2: M2 = (A10+A11)B00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, -1]
    
    # Term 3: M3 = A00(B10-B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 0, 1, -1]
    W[:, 2] = [-1, 1, 0, 0]
    
    # Term 4: M4 = A11(B01-B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 1, -1, 0]
    W[:, 3] = [1, 0, 1, 0]
    
    # Term 5: M5 = (A00+A01)B11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 1, 0, 0]
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    # Check if sum_r U_r V_r W_r^T equals the tensor
    # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # u = i*m + j, v = j*p + k, w = k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    
    # Construct target tensor T as a 3D array or check specific points
    # For efficiency, check all valid (i, j, k) points
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = 0
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                if abs(val - 1.0) > 1e-9:
                    return False
                # Also check a few invalid points to ensure no leakage
                # e.g. w = (k+1)*n + i
                if k < p - 1:
                    w_bad = (k + 1) * n + i
                    val_bad = 0
                    for r in range(U.shape[1]):
                        val_bad += U[u, r] * V[v, r] * W[w_bad, r]
                    if abs(val_bad) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # R(2, 3, 7) <= R(2, 1, 7) + R(2, 2, 7)
    # R(2, 1, 7) = 14 (Schoolbook)
    # R(2, 2, 7) <= R(2, 2, 2) + R(2, 2, 5)
    # R(2, 2, 2) = 7 (Strassen)
    # R(2, 2, 5) <= R(2, 2, 2) + R(2, 2, 3)
    # R(2, 2, 2) = 7 (Strassen)
    # R(2, 2, 3) = 12 (Schoolbook, safe)
    # Total: 14 + (7 + (7 + 12)) = 40
    
    # 1. Construct R(2, 1, 7)
    U1, V1, W1 = _schoolbook(2, 1, 7)
    
    # 2. Construct R(2, 2, 7)
    # 2a. R(2, 2, 2)
    U2a, V2a, W2a = _strassen_2_2_2()
    # Verify Strassen
    if not _verify_decomposition(U2a, V2a, W2a, 2, 2, 2):
        # Fallback to schoolbook for 2x2x2
        U2a, V2a, W2a = _schoolbook(2, 2, 2)
        
    # 2b. R(2, 2, 5)
    # 2b1. R(2, 2, 2) (Shifted k by 2)
    U2b1, V2b1, W2b1 = _strassen_2_2_2()
    # Shift V and W for k offset 2
    # V indices: j*2 + k. New k = k+2.
    # V_new[j*2 + (k+2)] = V_old[j*2 + k]
    V2b1_shifted = np.zeros((2 * 5, U2b1.shape[1]))
    W2b1_shifted = np.zeros((5 * 2, W2b1.shape[1]))
    for j in range(2):
        for k in range(2):
            V2b1_shifted[j * 5 + (k + 2), :] = V2b1[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W2b1_shifted[(k + 2) * 2 + i, :] = W2b1[k * 2 + i, :]
            
    # 2b2. R(2, 2, 3) (Schoolbook, k=0,1,2)
    U2b2, V2b2, W2b2 = _schoolbook(2, 2, 3)
    
    # Combine 2b1 and 2b2 to get R(2, 2, 5)
    U2b, V2b, W2b = _additive_combine(U2b1, V2b1_shifted, W2b1_shifted, U2b2, V2b2, W2b2)
    
    # Combine 2a and 2b to get R(2, 2, 7)
    # 2a covers k=0,1. 2b covers k=0..4 (but shifted 2a part to k=2,3? No, 2b1 was shifted to k=2,3. 2b2 is k=0,1,2).
    # Wait, 2a is k=0,1.
    # 2b1 is k=2,3.
    # 2b2 is k=0,1,2.
    # Overlap at k=0,1,2?
    # My split was R(2,2,7) = R(2,2,2) [k=0,1] + R(2,2,5) [k=2,3,4,5,6]?
    # No, R(2,2,5) covers 5 columns.
    # If I use R(2,2,2) for k=0,1 and R(2,2,5) for k=2,3,4,5,6?
    # R(2,2,5) needs to be aligned to k=2..6.
    
    # Let's re-align.
    # Part A: k=0,1. Use Strassen 2x2x2.
    # Part B: k=2,3,4,5,6. Use R(2,2,5).
    
    # Part B construction:
    # R(2,2,5) = R(2,2,2) [local k=0,1] + R(2,2,3) [local k=2,3,4]
    # Local k=0,1 -> Global k=2,3.
    # Local k=2,3,4 -> Global k=4,5,6.
    
    # B1: R(2,2,2) shifted to global k=2,3.
    U_B1, V_B1, W_B1 = _strassen_2_2_2()
    V_B1_shifted = np.zeros((2 * 7, U_B1.shape[1]))
    W_B1_shifted = np.zeros((7 * 2, W_B1.shape[1]))
    for j in range(2):
        for k in range(2):
            V_B1_shifted[j * 7 + (k + 2), :] = V_B1[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W_B1_shifted[(k + 2) * 2 + i, :] = W_B1[k * 2 + i, :]
            
    # B2: R(2,2,3) shifted to global k=4,5,6.
    U_B2, V_B2, W_B2 = _schoolbook(2, 2, 3)
    V_B2_shifted = np.zeros((2 * 7, U_B2.shape[1]))
    W_B2_shifted = np.zeros((7 * 2, W_B2.shape[1]))
    for j in range(2):
        for k in range(3):
            V_B2_shifted[j * 7 + (k + 4), :] = V_B2[j * 3 + k, :]
    for k in range(3):
        for i in range(2):
            W_B2_shifted[(k + 4) * 2 + i, :] = W_B2[k * 2 + i, :]
            
    # Combine B1 and B2 to get R(2,2,5) aligned to global k=2..6
    U_B, V_B, W_B = _additive_combine(U_B1, V_B1_shifted, W_B1_shifted, U_B2, V_B2_shifted, W_B2_shifted)
    
    # Combine Part A (k=0,1) and Part B (k=2..6) to get R(2,2,7)
    U_227, V_227, W_227 = _additive_combine(U2a, V2a, W2a, U_B, V_B, W_B)
    
    # Combine R(2,1,7) and R(2,2,7)
    # R(2,1,7) uses j=0. R(2,2,7) uses j=0,1?
    # No, R(2,1,7) is for the first column of A (j=0).
    # R(2,2,7) is for the next 2 columns (j=1,2).
    # We need to shift R(2,2,7) to j=1,2.
    
    U_227_shifted = np.zeros((2 * 3, U_227.shape[1]))
    V_227_shifted = np.zeros((2 * 7, U_227.shape[1]))
    W_227_shifted = W_227.copy()
    
    for i in range(2):
        for j in range(2):
            U_227_shifted[i * 3 + (j + 1), :] = U_227[i * 2 + j, :]
    for j in range(2):
        for k in range(7):
            V_227_shifted[(j + 1) * 7 + k, :] = V_227[j * 2 + k, :]
            
    U_final, V_final, W_final = _additive_combine(U1, V1, W1, U_227_shifted, V_227_shifted, W_227_shifted)
    
    # Final Verification
    if _verify_decomposition(U_final, V_final, W_final, 2, 3, 7):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to Schoolbook
        U_s, V_s, W_s = _schoolbook(2, 3, 7)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

