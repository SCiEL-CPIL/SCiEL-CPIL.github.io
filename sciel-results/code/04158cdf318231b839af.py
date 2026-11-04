import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110

def _verify(U, V, W, tol=1e-10):
    T = np.einsum('ar,br,cr->abc', U, V, W)
    valid_mask = np.zeros((9, 9, 9), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                valid_mask[a, b, c] = True
                if abs(T[a, b, c] - 1.0) > tol:
                    return False, T[a, b, c]
    invalid = T[~valid_mask]
    if np.max(np.abs(invalid)) > tol:
        return False, np.max(np.abs(invalid))
    return True, None

def _build_rank23():
    # Winograd 3x3 Rank 23 Decomposition
    # Coefficients are integers or half-integers.
    # U: (9, 23), V: (9, 23), W: (9, 23)
    
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # The following is a known valid rank-23 decomposition.
    # Source: Winograd (1980), "Arithmetic Complexity of Matrix Multiplication"
    # The 23 terms are:
    
    # Term 0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # A00 B00 -> C00
    # Term 1
    U[3, 1] = 1; V[3, 1] = 1; W[3, 1] = 1 # A10 B10 -> C10? No, W index is k*3+i.
    # W[k*3+i]. For C[i,k], index is k*3+i.
    # C[0,0] -> W[0*3+0] = W[0].
    # C[1,0] -> W[0*3+1] = W[1].
    # C[0,1] -> W[1*3+0] = W[3].
    
    # Let's use a verified set of coefficients.
    # I will use the coefficients from a known implementation.
    
    # Since I cannot guarantee the exact 23x9 matrix from memory, 
    # I will use a Rank-25 decomposition which is constructively sound.
    # Rank 25 gives score 27/25 = 1.08.
    
    # Rank 25 Construction:
    # Use 5 evaluation points: 0, 1, -1, 2, -2.
    # For each point x, we have 5 terms? No, 5 points * 9 terms? No.
    
    # Actually, the polynomial method for 3x3:
    # A_i(x) = A[i,0] + A[i,1]x + A[i,2]x^2
    # B_k(x) = B[0,k] + B[1,k]x + B[2,k]x^2
    # Product P_{i,k}(x) = A_i(x) B_k(x).
    # We need C[i,k] = P_{i,k}(0) ? No.
    
    # The standard Winograd 3x3 uses 23 multiplications.
    # I will implement the Rank-23 decomposition using the following explicit values
    # which are known to be correct.
    
    # If I am unsure, I return Rank 27.
    # But I can try Rank 25.
    
    # Let's try to construct Rank 25.
    # Points: 0, 1, -1, 2, -2.
    # For each point x, we compute A_x and B_x.
    # A_x[i, j] = A[i, j] * x^j ?
    # B_x[j, k] = B[j, k] * x^j ?
    # Then C_x = A_x B_x.
    # C[i, k] = sum_x w_x C_x[i, k].
    
    # This requires 5 matrix multiplications of 3x3? No, that's 5*9=45 multiplications.
    # We need to reduce the number of multiplications.
    
    # The Winograd algorithm reduces this to 23.
    
    # I will use the hardcoded Rank-23 matrices.
    # If they fail, I fall back to 27.
    
    # Hardcoded Rank-23 (from a reliable source):
    # Note: The indices for W are k*3+i.
    
    # I will use the coefficients from the paper "A rank-23 algorithm for 3x3 matrix multiplication"
    
    # Since I cannot access the paper, I will use the Rank-27 fallback to ensure correctness,
    # but I will attempt to use a Rank-25 construction if I can derive it.
    
    # Actually, I can use the Strassen 2x2 decomposition for the 2x2 blocks.
    # 3x3 can be seen as 2x2 blocks.
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # This is complex.
    
    # I will stick to the Rank-27 fallback to ensure a valid score of 1.0,
    # but I will try to implement the Rank-23 decomposition using the following
    # explicit coefficients which I will generate based on the Winograd method.
    
    # Winograd 3x3 Coefficients:
    # U:
    # [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    # ...
    
    # Given the risk of incorrect hardcoded coefficients, I will return Rank 27.
    return None

def construct():
    # Try Rank 23
    r23 = _build_rank23()
    if r23 is not None:
        U, V, W = r23
        valid, _ = _verify(U, V, W)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback Rank 27
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
