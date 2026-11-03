"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = 0
                for r in range(rank):
                    val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                if val != 1:
                    return False
    # Check invalid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                for jp in range(m):
                    for kp in range(p):
                        for ip in range(n):
                            if jp == j and ip == i and kp == k:
                                continue
                            b_idx = jp * p + kp
                            c_idx = kp * n + ip
                            val = 0
                            for r in range(rank):
                                val += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                            if val != 0:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank-17 construction (Strassen on 2x2 block + schoolbook for column 2)
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen part for 2x2 block (indices 0-6)
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B00) -> contributes to C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> contributes to C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> contributes to C02, C00
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for remaining terms (indices 7-16)
    # A02*B20 -> C00
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    # A02*B21 -> C01
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    # A12*B20 -> C10
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    # A12*B21 -> C11
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    # A00*B02 -> C02
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    # A01*B12 -> C02
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    # A10*B02 -> C12
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    # A11*B12 -> C12
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    # A02*B22 -> C02
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    # A12*B22 -> C12
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Rank-17 decomposition failed verification")
    
    # Attempt to find rank-16 by searching for alternative constructions
    # within the same family (Strassen-like + schoolbook hybrid)
    if time.time() < deadline - 20:
        # Try different 2x2 block choices for Strassen
        # Current: uses A[0:2, 0:2] and B[0:2, 0:2] for the 2x2 Strassen part
        # Alternative: use A[0:2, 0:2] and B[0:2, 1:3] or other combinations
        
        # Try: Strassen on A cols 0-1, B cols 1-2
        rank2 = 17
        U2 = np.zeros((n * m, rank2), dtype=int)
        V2 = np.zeros((m * p, rank2), dtype=int)
        W2 = np.zeros((p * n, rank2), dtype=int)
        
        # Strassen on A[0:2, 0:2] and B[0:2, 1:3]
        # M1 = (A00+A11)(B01+B12)
        U2[0, 0] = 1; U2[4, 0] = 1
        V2[1, 0] = 1; V2[5, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # This doesn't work directly
        
        # This approach is getting complicated. Let me try a different strategy:
        # Search for rank-16 by trying to merge two schoolbook terms with Strassen terms
        
        # For now, stick with verified rank-17
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
