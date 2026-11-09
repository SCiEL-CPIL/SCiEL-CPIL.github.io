"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W, rank
    
    def verify(U, V, W):
        """Check if T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank 53: Strassen for (2x2)*(2x6) part + schoolbook for (2x3)*(3x6) part
    # Actually, let's try a few candidate ranks and return the smallest that works.
    
    # Candidate 1: rank 53 (21 from Strassen + 32 from rank-16 2x3x3)
    # But I don't have the rank-16 2x3x3 decomposition readily available.
    # Let's use rank 57: 21 + 36 (schoolbook for second block)
    
    rank_57 = 57
    U = np.zeros((n*m, rank_57))
    V = np.zeros((m*p, rank_57))
    W = np.zeros((p*n, rank_57))
    
    def add_term(u_entries, v_entries, w_entries, c):
        for idx, val in u_entries:
            U[idx, c] += val
        for idx, val in v_entries:
            V[idx, c] += val
        for idx, val in w_entries:
            W[idx, c] += val
    
    def add_strassen(i1, i2, j1, j2, k1, k2, c_start):
        u11 = i1*m + j1; u12 = i1*m + j2
        u21 = i2*m + j1; u22 = i2*m + j2
        v11 = j1*p + k1; v12 = j1*p + k2
        v21 = j2*p + k1; v22 = j2*p + k2
        w11 = k1*n + i1; w12 = k1*n + i2
        w21 = k2*n + i1; w22 = k2*n + i2
        
        add_term([(u11,1),(u22,1)], [(v11,1),(v22,1)], [(w11,1),(w22,1)], c_start)
        add_term([(u21,1),(u22,1)], [(v11,1)], [(w21,1)], c_start+1)
        add_term([(u11,1)], [(v12,1),(v22,-1)], [(w12,1),(w22,1)], c_start+2)
        add_term([(u22,1)], [(v21,1),(v11,-1)], [(w11,1),(w21,1)], c_start+3)
        add_term([(u11,1),(u12,1)], [(v22,1)], [(w11,-1),(w12,1)], c_start+4)
        add_term([(u21,1),(u11,-1)], [(v11,1),(v12,1)], [(w22,1)], c_start+5)
        add_term([(u12,1),(u22,-1)], [(v21,1),(v22,1)], [(w11,1),(w22,-1)], c_start+6)
        return c_start + 7
    
    col = 0
    # (2x2)*(2x6) with Strassen, split k into pairs
    col = add_strassen(0, 1, 0, 1, 0, 1, col)
    col = add_strassen(0, 1, 0, 1, 2, 3, col)
    col = add_strassen(0, 1, 0, 1, 4, 5, col)
    
    # (2x3)*(3x6) schoolbook: i in {0,1}, j in {2,3,4}, k in {0..5}
    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    assert col == rank_57, f"Got {col}, expected {rank_57}"
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback
        U, V, W, r = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
