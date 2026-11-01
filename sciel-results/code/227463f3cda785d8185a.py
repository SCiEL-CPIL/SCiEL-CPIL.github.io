import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Using dot product: U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :])
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    nm, mp, pn = n * m, m * p, p * n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_rank_55(n, m, p):
    """
    Constructs a rank-55 decomposition for 2x3 * 3x10.
    Based on blockwise Strassen for 2x3 * 3x2 blocks (rank 7 per block for 5 blocks? No, 11 per block).
    Actually, the inspiration uses a specific 55-rank construction.
    Structure:
    - 5 blocks of 2 columns each.
    - For each block (cols k0, k1), we use 7 terms derived from Strassen-like identities for the 2x3 * 3x2 part, 
      plus handling for the third row of A (index 2) which is not part of the 2x2 Strassen core but interacts.
    
    Let's trace the inspiration's build_rank_55 logic:
    It loops b in range(5).
    Inside, it uses 7 terms (M1..M7).
    Then it loops k in range(p) (10 times) and adds 1 term each? 
    Wait, 5 * 7 = 35. 10 * 1 = 10. Total 45? 
    Let's re-read the inspiration code carefully.
    
    Inspiration code:
    for b in range(5):
       ... 7 terms ...
    for k in range(p): # p=10
       ... 2 terms? No, U[a02]... and U[a12]... 
       It adds 2 terms inside the loop? 
       "U[a02, col] += 1.0; ... col += 1"
       "U[a12, col] += 1.0; ... col += 1"
       So 10 * 2 = 20 terms.
    Total = 5 * 7 + 20 = 35 + 20 = 55.
    Yes, rank 55.
    """
    rank = 55
    nm, mp, pn = n * m, m * p, p * n
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    # Indices for A (2x3 flattened to 6)
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices (3x10 flattened to 30)
        # B[0, k0]=k0, B[0, k1]=k1
        # B[1, k0]=10+k0, B[1, k1]=10+k1
        # B[2, k0]=20+k0, B[2, k1]=20+k1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        b20, b21 = 20 + k0, 20 + k1
        
        # W indices (10x2 flattened to 20) -> W[k*n + i]
        # C[0, k0]=2*k0, C[0, k1]=2*k1
        # C[1, k0]=2*k0+1, C[1, k1]=2*k1+1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # The 7 terms for the 2x3 * 3x2 subproblem involving rows 0,1 of A and cols k0,k1 of B
        # These terms handle A[0,0], A[0,1], A[1,0], A[1,1] interacting with B[0,..], B[1,..]
        # Note: A[0,2] and A[1,2] are handled in the second loop.
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1

    # Handle A[0,2] and A[1,2] (the third column of A)
    # These interact with B[2, k] (the third row of B)
    # C[i, k] += A[i, 2] * B[2, k]
    for k in range(p):
        b2k = 2 * p + k  # Index for B[2, k]
        
        # Term for A[0,2] * B[2, k] -> C[0, k]
        U[a02, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k, col] += 1.0  # C[0, k] is at index 2*k
        col += 1
        
        # Term for A[1,2] * B[2, k] -> C[1, k]
        U[a12, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k + 1, col] += 1.0  # C[1, k] is at index 2*k+1
        col += 1
        
    return U, V, W

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Try to build and verify the rank-55 decomposition
    try:
        U55, V55, W55 = _build_rank_55(n, m, p)
        # Verify
        if _verify_decomposition(U55, V55, W55, n, m, p):
            return {"U": U55.tolist(), "V": V55.tolist(), "W": W55.tolist()}
        else:
            # If verification fails, we have a bug in the logic or implementation
            # Fall back to schoolbook
            pass
    except Exception:
        pass
    
    # 2. Fallback: Schoolbook decomposition (rank 60)
    U, V, W = _schoolbook_decomposition(n, m, p)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 3. Last resort (should not happen)
    return {"U": [], "V": [], "W": []}

if __name__ == "__main__":
    # Self-test
    res = construct()
    if res["U"]:
        U = np.array(res["U"])
        V = np.array(res["V"])
        W = np.array(res["W"])
        print(f"Rank: {U.shape[1]}")
        print(f"Verified: {_verify_decomposition(U, V, W, 2, 3, 10)}")
    else:
        print("Failed to construct")
