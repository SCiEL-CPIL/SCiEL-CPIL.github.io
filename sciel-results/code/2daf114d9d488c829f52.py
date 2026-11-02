import numpy as np

def _build_rank_40():
    n, m, p = 2, 3, 10
    rank = 40
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for k in range(5):
        # Strassen part for A[0:2, 0:2] and B[0:2, 2k:2k+2]
        # A indices: 00=0, 01=1, 10=3, 11=4
        # B indices: 0,2k = 2k; 0,2k+1 = 2k+1; 1,2k = 4+2k; 1,2k+1 = 5+2k
        # C indices: 0,2k = 4k; 0,2k+1 = 4k+2; 1,2k = 4k+1; 1,2k+1 = 4k+3
        
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00, b01 = 2*k, 2*k+1
        b10, b11 = 4+2*k, 5+2*k
        c00, c01 = 4*k, 4*k+2
        c10, c11 = 4*k+1, 4*k+3
        
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
        
        # Tail part: A[0:2, 2] and B[2, 2k:2k+2]
        # A indices: 02=2, 12=5
        # B indices: 2,2k = 8+2k; 2,2k+1 = 9+2k
        # C indices: same as above
        
        a02, a12 = 2, 5
        b20, b21 = 8+2*k, 9+2*k
        
        U[a02, col] += 1.0; U[a12, col] += 1.0
        V[b20, col] += 1.0; V[b21, col] += 1.0
        W[c00, col] += 1.0; W[c01, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] += 1.0
        col += 1

    return U, V, W, rank

_U40, _V40, _W40, _RANK40 = _build_rank_40()

def construct():
    return {"U": _U40.tolist(), "V": _V40.tolist(), "W": _W40.tolist()}
