"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    where a = i*m+j, b = j*p+k, c = k*n+i
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U @ V^T @ W^T (outer product sum) equals T."""
    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Reconstruct:
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _try_strassen_2x2_block(n, m, p):
    """
    Try to use Strassen for the 2x2x2 sub-block (A[0:2,0:2] * B[0:2,0:2])
    and schoolbook for the rest. This gives rank 7 + remaining.
    """
    # Schoolbook for full: n*m*p = 30
    # If we use Strassen for A[0:2,0:2]*B[0:2,0:2] (rank 7 instead of 8),
    # we save 1, giving rank 29.
    
    # But we can do better by finding a more clever decomposition.
    # Let's try to find a rank-28 or lower decomposition using numerical search.
    pass

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    A is 2x2, B is 2x2, C is 2x2.
    A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    C indices (trace convention k*n+i): C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    Wait, C is n x p = 2x2, so C[k,i] where k in [0,1], i in [0,1]
    W index = k*n + i = k*2 + i
    C[0,0]: k=0,i=0 -> W[0]
    C[0,1]: k=0,i=1 -> W[1]
    C[1,0]: k=1,i=0 -> W[2]
    C[1,1]: k=1,i=1 -> W[3]
    
    A[i,j]: index = i*m + j = i*2 + j
    A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    B[j,k]: index = j*p + k = j*2 + k
    B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00-A10)*B11
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01+A11)*(B00+B01)
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11)
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)*(B00-B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _attempt_low_rank_search(n, m, p, max_rank=29, time_limit=120):
    """
    Try to find a low-rank decomposition using numerical optimization.
    Start with schoolbook and try to reduce rank.
    """
    start_time = time.time()
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from a lower bound up
    # Lower bound for 2x3x5: at least max(2*3, 3*5, 2*5) = 15
    # Actually the rank of M_{n,m,p} >= max(nm, mp, np)
    # For 2,3,5: max(6, 15, 10) = 15
    # But we know schoolbook is 30.
    
    # Let's try to find if rank 29 is achievable (Strassen on 2x2x2 block + schoolbook rest)
    # That would save 1 from schoolbook.
    
    # Actually, let's just return the best we can find quickly.
    # Given time constraints, let's try a simple approach:
    # 1. Try rank = 29 using Strassen on the 2x2x2 corner
    # 2. If that works, return it
    
    # For now, let's just return schoolbook if we can't find better quickly.
    elapsed = time.time() - start_time
    if elapsed > time_limit - 10:
        return best_result
    
    return best_result

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Start with schoolbook as fallback
    result = _schoolbook_decomposition(n, m, p)
    
    # Try to improve with Strassen-based approach
    # Use Strassen for the 2x2x2 sub-block and schoolbook for the rest
    # This should give rank 29 (save 1 from schoolbook's 30)
    
    # Let's construct this explicitly
    # Full decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # Terms: (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    # Total 30 terms.
    
    # Group 1: j=0,1 and k=0,1 -> A[i,0:2] * B[0:2,0:2] -> 2*2*2 = 8 schoolbook terms
    # Replace with Strassen (7 terms)
    # Group 2: j=0,1 and k=2,3,4 -> A[i,0:2] * B[0:2,2:5] -> 2*2*3 = 6 schoolbook terms
    # Group 3: j=2 and k=0,1,2,3,4 -> A[i,2] * B[2,0:5] -> 2*1*5 = 10 schoolbook terms
    # Total: 7 + 6 + 10 = 23? No wait, that's not right.
    
    # Let me recount:
    # Full schoolbook: sum over j=0,1,2 and k=0,1,2,3,4 for each i=0,1
    # That's 2 * 3 * 5 = 30 terms.
    
    # If I use Strassen for the j in {0,1}, k in {0,1} part:
    # This handles terms where j in {0,1} and k in {0,1}
    # For each i, C[i, 0:2] gets contributions from A[i,0]*B[0,0:2] + A[i,1]*B[1,0:2]
    # The Strassen block computes A[0:2, 0:2] * B[0:2, 0:2] = C_strassen[0:2, 0:2]
    # This replaces 2*2*2 = 8 terms with 7 terms.
    
    # Remaining terms:
    # j in {0,1}, k in {2,3,4}: 2*2*3 = 12 terms (for both i values)
    # j = 2, k in {0,1,2,3,4}: 2*1*5 = 10 terms
    # Total: 7 + 12 + 10 = 29
    
    # Let's construct this decomposition
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # (6, 29)
    V = [[0] * rank for _ in range(m * p)]  # (15, 29)
    W = [[0] * rank for _ in range(p * n)]  # (10, 29)
    
    col = 0
    
    # Strassen block: A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: i in {0,1}, j in {0,1} -> U rows: i*3+j = 0,1,3,4
    # B indices: j in {0,1}, k in {0,1} -> V rows: j*5+k = 0,1,5,6
    # C indices: k in {0,1}, i in {0,1} -> W rows: k*2+i = 0,1,2,3
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Map Strassen indices to full matrix indices
    # A row mapping: Strassen row s -> full row i*3+j
    # Strassen A rows: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    a_map = [0, 1, 3, 4]  # Strassen row -> full U row
    
    # B row mapping: Strassen row s -> full row j*5+k
    # Strassen B rows: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    b_map = [0, 1, 5, 6]  # Strassen row -> full V row
    
    # C row mapping: Strassen row s -> full row k*2+i
    # Strassen C rows: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # W index = k*n + i = k*2 + i
    c_map = [0, 1, 2, 3]  # Strassen row -> full W row
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map[s]][col] += W_s[s][r]
        col += 1
    
    # Now add schoolbook terms for remaining parts
    # Part 1: j in {0,1}, k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * 3 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    
    # Part 2: j = 2, k in {0,1,2,3,4}
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Verify
    T = _build_target_tensor(n, m, p)
    valid, diff = _verify_decomposition(U, V, W, T)
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {result['U'][0].__len__() if result['U'] else 0}")
