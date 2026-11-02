"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Build the true tensor
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1.0

    def verify(U, V, W, atol=1e-9):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(T_recon, T_true, atol=atol)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_decomposition():
        """
        Constructs the rank-49 decomposition using 7 Strassen terms per 2x2 block.
        p=14 allows 7 blocks of size 2.
        """
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        
        # Strassen 7 terms definitions for 2x2 matrices
        # Indices in U: A00=0, A01=1, A10=2, A11=3
        # Indices in V: B0k=0..6, B1k=7..13 (relative to block, offset p=14 for B1)
        # Indices in W: C0k=0..6, C1k=7..13 (relative to block)
        
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # Actually, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M7
        # C11 = M1 - M2 - M4 + M6
        
        # We map this to our tensor.
        # For block l (columns 2l, 2l+1 of B):
        # B00 is V[2l, :], B01 is V[2l+1, :], B10 is V[14+2l, :], B11 is V[14+2l+1, :]
        # C00 is W[2l, :], C01 is W[2l+1, :], C10 is W[14+2l, :], C11 is W[14+2l+1, :]
        
        def add_term(u_indices, v_indices, w_indices, sign):
            """Adds a term to U, V, W matrices."""
            # u_indices: list of (index, coeff)
            # v_indices: list of (index, coeff)
            # w_indices: list of (index, coeff)
            pass

        for l in range(7):
            base = l * 7
            # B indices
            b00 = 2 * l
            b01 = 2 * l + 1
            b10 = 14 + 2 * l
            b11 = 14 + 2 * l + 1
            
            # C indices
            c00 = 2 * l
            c01 = 2 * l + 1
            c10 = 14 + 2 * l
            c11 = 14 + 2 * l + 1
            
            # M1: (A00+A11)(B00+B11)
            # U: A00(+1), A11(+1)
            # V: B00(+1), B11(+1)
            # W: C00(+1), C11(+1)
            U[0, base] += 1; U[3, base] += 1
            V[b00, base] += 1; V[b11, base] += 1
            W[c00, base] += 1; W[c11, base] += 1
            
            # M2: (A10+A11)B00
            # U: A10(+1), A11(+1)
            # V: B00(+1)
            # W: C10(+1), C11(-1)
            U[2, base+1] += 1; U[3, base+1] += 1
            V[b00, base+1] += 1
            W[c10, base+1] += 1; W[c11, base+1] -= 1
            
            # M3: A00(B01-B11)
            # U: A00(+1)
            # V: B01(+1), B11(-1)
            # W: C01(+1)
            U[0, base+2] += 1
            V[b01, base+2] += 1; V[b11, base+2] -= 1
            W[c01, base+2] += 1
            
            # M4: A11(B10-B00)
            # U: A11(+1)
            # V: B10(+1), B00(-1)
            # W: C00(+1), C11(-1)
            U[3, base+3] += 1
            V[b10, base+3] += 1; V[b00, base+3] -= 1
            W[c00, base+3] += 1; W[c11, base+3] -= 1
            
            # M5: (A00+A01)B11
            # U: A00(+1), A01(+1)
            # V: B11(+1)
            # W: C00(-1), C01(+1)
            U[0, base+4] += 1; U[1, base+4] += 1
            V[b11, base+4] += 1
            W[c00, base+4] -= 1; W[c01, base+4] += 1
            
            # M6: (A10-A00)(B00+B01)
            # U: A10(+1), A00(-1)
            # V: B00(+1), B01(+1)
            # W: C11(+1)
            U[2, base+5] += 1; U[0, base+5] -= 1
            V[b00, base+5] += 1; V[b01, base+5] += 1
            W[c11, base+5] += 1
            
            # M7: (A01-A11)(B10+B11)
            # U: A01(+1), A11(-1)
            # V: B10(+1), B11(+1)
            # W: C00(-1), C10(+1)
            U[1, base+6] += 1; U[3, base+6] -= 1
            V[b10, base+6] += 1; V[b11, base+6] += 1
            W[c00, base+6] -= 1; W[c10, base+6] += 1

        return U, V, W

    # Primary strategy: Rank 49 Strassen
    U, V, W = build_strassen_decomposition()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook (Rank 56)
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
