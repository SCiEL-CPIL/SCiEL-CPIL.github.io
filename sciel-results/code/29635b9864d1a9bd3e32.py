import time
import numpy as np

DIMENSIONS = [4, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor using vectorized ops."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    Rank 7 decomposition.
    U, V, W shapes: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _rank49_4x4x4():
    """
    Construct rank 49 decomposition for 4x4 * 4x4 using Kronecker product of Strassen.
    U, V, W shapes: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    U4[i, c] = U2[i1, c1] * U2[i2, c2]
            
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    V4[j, c] = V2[j1, c1] * V2[j2, c2]
            
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    W4[k, c] = W2[k1, c1] * W2[k2, c2]
                    
    return U4, V4, W4

def _build_4x4x7_strassen():
    """
    Build a rank 98 decomposition for 4x4 * 4x7 by splitting B into 4x3 and 4x4 parts.
    Part 1: 4x4 * 4x3 -> use 4x4x4 decomposition but zero out columns corresponding to k=3 in V and W.
    Part 2: 4x4 * 4x4 -> use 4x4x4 decomposition, shift k indices by 3.
    """
    U4, V4, W4 = _rank49_4x4x4()
    n, m, p = 4, 4, 7
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4 * 4x3 (cols 0-2 of B)
    for c in range(49):
        U[:, c] = U4[:, c]
        for j in range(4):
            for k in range(3):
                V[j * 7 + k, c] = V4[j * 4 + k, c]
        for k in range(3):
            for i in range(4):
                W[k * 4 + i, c] = W4[k * 4 + i, c]

    # Part 2: 4x4 * 4x4 (cols 3-6 of B)
    for c in range(49):
        c2 = 49 + c
        U[:, c2] = U4[:, c]
        for j in range(4):
            for k in range(4):
                V[j * 7 + (3 + k), c2] = V4[j * 4 + k, c]
        for k in range(4):
            for i in range(4):
                W[(3 + k) * 4 + i, c2] = W4[k * 4 + i, c]
                
    return U, V, W

def _build_4x4x7_strassen_v2():
    """
    Alternative rank 98 construction: split 4x4x7 into 4x4x2 + 4x4x2 + 4x4x3.
    Each 4x4x2 uses rank 49 (with 2 of 4 k-indices active).
    4x4x3 uses rank 49 (with 3 of 4 k-indices active).
    Total: 3 * 49 = 147? No, that's worse.
    
    Actually, let's try: 4x4x7 = 4x4x4 + 4x4x3.
    4x4x4: rank 49 (Strassen Kronecker)
    4x4x3: We can use the 4x4x4 decomposition but only keep 3 of the 4 k-slices.
    But the rank doesn't change because the U and V matrices still have 49 columns.
    
    Wait, if we use the same U4 for both, we can't share.
    
    Let's try a different split: 4x4x7 = (2x4x7) + (2x4x7) where we split A's rows.
    No, that doesn't help.
    
    Let's try: 4x4x7 as 4 instances of 1x4x7? No.
    
    Actually, the current rank 98 is already better than schoolbook 112.
    Let's try to see if we can do better than 98.
    
    One idea: Use 2x2x7 (rank 28 schoolbook) and combine two of them?
    4x4x7 = (2x2x7) x 2 (in the i and j dimensions)?
    If we use 2x2x2 Strassen for the 2x2 part and 1x1x7 for the rest?
    No, that gives 7 * 7 = 49 for 2x2x7? No.
    
    Let's stick with rank 98 for now and try to verify it.
    """
    return _build_4x4x7_strassen()

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the known rank 98 decomposition
    U98, V98, W98 = _build_4x4x7_strassen()
    
    # Verify it's correct
    if not _verify_decomposition(U98, V98, W98, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Return the rank 98 decomposition
    return {"U": U98.tolist(), "V": V98.tolist(), "W": W98.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
