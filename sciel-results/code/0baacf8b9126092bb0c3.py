import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for t in range(rank):
                    lhs += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(lhs - 1.0) > tol:
                    return False
                # Check that wrong terms are zero
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i2, j2, k2) == (i, j, k):
                                continue
                            rhs = 0.0
                            for t in range(rank):
                                rhs += U[i2 * m + j2, t] * V[j2 * p + k2, t] * W[k2 * n + i2, t]
                            if abs(rhs) > tol:
                                return False
    return True

def _strassen_2x2_factors():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, each 4x7."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, -1]
    ], dtype=float)
    W = np.array([
        [1, 0, -1, 1, -1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0]
    ], dtype=float)
    return U, V, W

def _build_rank44():
    """Build rank-44 decomposition for 3x4x4 matrix multiplication."""
    n, m, p = 3, 4, 4
    U = np.zeros((n * m, 44), dtype=float)
    V = np.zeros((m * p, 44), dtype=float)
    W = np.zeros((p * n, 44), dtype=float)
    
    # Split A into A_top (2x4) and A_bot (1x4)
    # U indices: A_top uses rows 0-7, A_bot uses rows 8-11
    # V indices: all 16 rows (4x4)
    # W indices: C_top uses rows 0-7 (2x4), C_bot uses rows 8-11 (1x4)
    
    # Strassen 2x2 factors
    U2, V2, W2 = _strassen_2x2_factors()
    
    # C_top = A_top * B where A_top is 2x4, B is 4x4
    # Split: A_top = [A_L, A_R] where A_L, A_R are 2x2
    # B = [B_TL, B_TR; B_BL, B_BR] each 2x2
    # C_top = [A_L B_TL + A_R B_BL,  A_L B_TR + A_R B_BR]
    # C_L = A_L B_TL + A_R B_BL (2x2)
    # C_R = A_L B_TR + A_R B_BR (2x2)
    
    # Use Strassen for C_L = [A_L; A_R] * [B_TL; B_BL] ... 
    # Actually, C_L[i,k] = sum_{j=0}^{3} A_top[i,j] * B[j,k] for k=0,1
    # = A_L[i,:] @ B_TL[:,k] + A_R[i,:] @ B_BL[:,k]
    
    # This is a 2x4x2 multiplication. We can use two Strassen 2x2 decompositions.
    # For C_L: treat as (A_L + A_R) * (B_TL + B_BL) type structure
    # Actually, let's use the fact that 2x4x2 can be done in rank 14 (7+7)
    # by splitting into two 2x2x2 problems that share some structure.
    
    # C_L[i,k] = A_L[i,0]*B_TL[0,k] + A_L[i,1]*B_TL[1,k] + A_R[i,0]*B_BL[0,k] + A_R[i,1]*B_BL[1,k]
    # 
    # Let's define for the left half (k=0,1):
    # P1 = (A_L + A_R) @ (B_TL + B_BL)  -- gives extra terms
    # This doesn't directly work. Let me use a different approach.
    
    # Actually, for 2x4x2, we can view it as:
    # [A_L  A_R] * [B_TL; B_BL]
    # This is equivalent to two matrix products sharing the 2x4 input.
    # 
    # One approach: use Winograd/Strassen for the 4x4 case and restrict.
    # The Strassen 4x4 decomposition has rank 28. For 2x4x4, we can get rank 28.
    # But we want to split into C_L and C_R, each 2x4x2.
    
    # Let me try a direct construction for 2x4x2 with rank 14.
    # 2x4x2 schoolbook is rank 16. Can we do 14?
    # 
    # C[i,k] = sum_j A[i,j] B[j,k], i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # 
    # Group j: j=0,1 (top) and j=2,3 (bottom)
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # 
    # Let X = [A[:,0:2]; A[:,2:4]] conceptually, but A is already 2x4.
    # 
    # Use Strassen-like for 2x2x2 twice:
    # First 2x2x2: A_top_half (2x2) * B_top_half (2x2) -> contributes to C
    # Second 2x2x2: A_bot_half (2x2) * B_bot_half (2x2) -> contributes to C
    # 
    # But these are independent, so rank would be 7+7=14.
    # 
    # For the left half (output cols 0,1):
    # A_L = A_top[:, 0:2] (2x2), B_TL = B[0:2, 0:2] (2x2)
    # A_R = A_top[:, 2:4] (2x2), B_BL = B[2:4, 0:2] (2x2)
    # C_L = A_L @ B_TL + A_R @ B_BL
    # 
    # This is sum of two 2x2 products. Each can be done in rank 7 (Strassen).
    # So C_L can be done in rank 14.
    # Similarly C_R in rank 14.
    # Total for C_top: rank 28.
    # C_bot (1x4x4): schoolbook rank 16.
    # Total: 28 + 16 = 44. 
    
    # Now let's construct the factors.
    # For C_L = A_L @ B_TL + A_R @ B_BL:
    # We have two independent Strassen decompositions.
    
    # Strassen for A_L @ B_TL (first 2x2x2):
    # U indices for A_L: global U rows 0,1,4,5 (A[0,0],A[0,1],A[1,0],A[1,1])
    # V indices for B_TL: global V rows 0,1,4,5 (B[0,0],B[0,1],B[1,0],B[1,1])
    # W indices for C_L: global W rows 0,3,1,4 (C[0,0],C[0,1],C[1,0],C[1,1])
    
    # Strassen for A_R @ B_BL (second 2x2x2):
    # U indices for A_R: global U rows 2,3,6,7 (A[0,2],A[0,3],A[1,2],A[1,3])
    # V indices for B_BL: global V rows 8,9,12,13 (B[2,0],B[2,1],B[3,0],B[3,1])
    # W indices for C_L: same as above (C[0,0],C[0,1],C[1,0],C[1,1])
    
    # For C_R = A_L @ B_TR + A_R @ B_BR:
    # Strassen for A_L @ B_TR:
    # U indices: same as A_L: 0,1,4,5
    # V indices for B_TR: global V rows 2,3,6,7 (B[0,2],B[0,3],B[1,2],B[1,3])
    # W indices for C_R: global W rows 2,5,3,6 (C[0,2],C[0,3],C[1,2],C[1,3])
    
    # Strassen for A_R @ B_BR:
    # U indices: same as A_R: 2,3,6,7
    # V indices for B_BR: global V rows 10,11,14,15 (B[2,2],B[2,3],B[3,2],B[3,3])
    # W indices for C_R: same as above
    
    # Rank 1-7: A_L @ B_TL (Strassen)
    for t in range(7):
        # U: A_L elements
        U[0, t] += U2[0, t]  # A[0,0]
        U[1, t] += U2[1, t]  # A[0,1]
        U[4, t] += U2[2, t]  # A[1,0]
        U[5, t] += U2[3, t]  # A[1,1]
        # V: B_TL elements
        V[0, t] += V2[0, t]  # B[0,0]
        V[1, t] += V2[1, t]  # B[0,1]
        V[4, t] += V2[2, t]  # B[1,0]
        V[5, t] += V2[3, t]  # B[1,1]
        # W: C_L elements
        W[0, t] += W2[0, t]  # C[0,0]
        W[3, t] += W2[1, t]  # C[0,1]
        W[1, t] += W2[2, t]  # C[1,0]
        W[4, t] += W2[3, t]  # C[1,1]
    
    # Rank 8-14: A_R @ B_BL (Strassen)
    for t in range(7):
        col = 7 + t
        # U: A_R elements
        U[2, col] += U2[0, t]  # A[0,2]
        U[3, col] += U2[1, t]  # A[0,3]
        U[6, col] += U2[2, t]  # A[1,2]
        U[7, col] += U2[3, t]  # A[1,3]
        # V: B_BL elements
        V[8, col] += V2[0, t]  # B[2,0]
        V[9, col] += V2[1, t]  # B[2,1]
        V[12, col] += V2[2, t]  # B[3,0]
        V[13, col] += V2[3, t]  # B[3,1]
        # W: C_L elements (same output)
        W[0, col] += W2[0, t]  # C[0,0]
        W[3, col] += W2[1, t]  # C[0,1]
        W[1, col] += W2[2, t]  # C[1,0]
        W[4, col] += W2[3, t]  # C[1,1]
    
    # Rank 15-21: A_L @ B_TR (Strassen)
    for t in range(7):
        col = 14 + t
        # U: A_L elements
        U[0, col] += U2[0, t]
        U[1, col] += U2[1, t]
        U[4, col] += U2[2, t]
        U[5, col] += U2[3, t]
        # V: B_TR elements
        V[2, col] += V2[0, t]  # B[0,2]
        V[3, col] += V2[1, t]  # B[0,3]
        V[6, col] += V2[2, t]  # B[1,2]
        V[7, col] += V2[3, t]  # B[1,3]
        # W: C_R elements
        W[2, col] += W2[0, t]  # C[0,2]
        W[5, col] += W2[1, t]  # C[0,3]
        W[3, col] += W2[2, t]  # C[1,2]
        W[6, col] += W2[3, t]  # C[1,3]
    
    # Rank 22-28: A_R @ B_BR (Strassen)
    for t in range(7):
        col = 21 + t
        # U: A_R elements
        U[2, col] += U2[0, t]
        U[3, col] += U2[1, t]
        U[6, col] += U2[2, t]
        U[7, col] += U2[3, t]
        # V: B_BR elements
        V[10, col] += V2[0, t]  # B[2,2]
        V[11, col] += V2[1, t]  # B[2,3]
        V[14, col] += V2[2, t]  # B[3,2]
        V[15, col] += V2[3, t]  # B[3,3]
        # W: C_R elements
        W[2, col] += W2[0, t]  # C[0,2]
        W[5, col] += W2[1, t]  # C[0,3]
        W[3, col] += W2[2, t]  # C[1,2]
        W[6, col] += W2[3, t]  # C[1,3]
    
    # Rank 29-44: A_bot * B (1x4x4 schoolbook, rank 16)
    # A_bot is row 2 of A (2x4 matrix's second row), i.e., A[2,0],A[2,1],A[2,2],A[2,3]
    # B is 4x4, C_bot is 1x4 (row 2 of C)
    for j in range(4):
        for k in range(4):
            col = 28 + j * 4 + k
            U[8 + j, col] = 1.0  # A_bot[j]
            V[j * 4 + k, col] = 1.0  # B[j,k]
            W[k * 3 + 2, col] = 1.0  # C_bot[k] = C[2,k], W index = k*n + i = k*3 + 2
    
    return U, V, W

def _build_rank48():
    """Fallback: schoolbook rank-48 decomposition for 3x4x4."""
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 3, 4, 4
    deadline = 120.0
    start = time.time()
    
    # Try rank 44 first
    try:
        U, V, W = _build_rank44()
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, 44):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback to rank 48
    if time.time() - start > deadline - 10:
        raise TimeoutError("Time limit exceeded")
    
    U, V, W = _build_rank48()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
