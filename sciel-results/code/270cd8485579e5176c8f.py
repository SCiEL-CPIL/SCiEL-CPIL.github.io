import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _get_rank_32_construction():
    """
    Rank-32 decomposition: 16 terms per half (2 columns of B).
    
    Per half (columns c0, c1 of B):
    - 7 Strassen terms for A[0:2,0:2] * B[0:2,c0:c1] (2x2 * 2x2 block)
    - 4 shared cross terms: A[i,2]*(B[j,c0]+B[j,c1]) for i in {0,1}, j in {0,1}
    - 5 row-2 terms for A[2,0:3] * B[0:3,c0:c1]
    Total: 16 per half, 32 total.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    def add_term(u_indices, v_indices, w_indices, col):
        """Add a single rank-1 term at column col."""
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    def build_half(col_offset, c0, c1):
        """Build 16 terms for columns c0, c1 of B."""
        col = col_offset
        # B column indices: B[j, c] = j*p + c
        # For c0: B[j,c0] = j*4+c0, for c1: B[j,c1] = j*4+c1
        b00 = 0 * p + c0  # B[0,c0]
        b01 = 0 * p + c1  # B[0,c1]
        b10 = 1 * p + c0  # B[1,c0]
        b11 = 1 * p + c1  # B[1,c1]
        b20 = 2 * p + c0  # B[2,c0]
        b21 = 2 * p + c1  # B[2,c1]
        
        # W indices: C[i,c] = c*n + i
        # For c0: C[i,c0] = c0*3+i, for c1: C[i,c1] = c1*3+i
        w00 = c0 * n + 0  # C[0,c0]
        w01 = c0 * n + 1  # C[1,c0]
        w02 = c0 * n + 2  # C[2,c0]
        w10 = c1 * n + 0  # C[0,c1]
        w11 = c1 * n + 1  # C[1,c1]
        w12 = c1 * n + 2  # C[2,c1]
        
        # A indices: A[i,j] = i*m+j
        a00 = 0  # A[0,0]
        a01 = 1  # A[0,1]
        a02 = 2  # A[0,2]
        a10 = 3  # A[1,0]
        a11 = 4  # A[1,1]
        a12 = 5  # A[1,2]
        a20 = 6  # A[2,0]
        a21 = 7  # A[2,1]
        a22 = 8  # A[2,2]
        
        # === 7 Strassen terms for A[0:2,0:2] * B[0:2,c0:c1] ===
        # These compute: C[i,c] += sum_{j=0,1} A[i,j]*B[j,c] for i in {0,1}
        
        # S1 = (A00+A11)*(B00+B10) -> C[0,c0] += S1, C[1,c0] -= S1
        add_term([(a00,1),(a11,1)], [(b00,1),(b10,1)], [(w00,1),(w01,-1)], col); col += 1
        # S2 = (A10-A00)*(B00+B01) -> C[0,c0] -= S2, C[1,c0] += S2
        add_term([(a10,1),(a00,-1)], [(b00,1),(b01,1)], [(w00,-1),(w01,1)], col); col += 1
        # S3 = (A01-A11)*(B10+B11) -> C[0,c1] -= S3, C[1,c1] += S3
        add_term([(a01,1),(a11,-1)], [(b10,1),(b11,1)], [(w10,-1),(w11,1)], col); col += 1
        # S4 = A00*(B01-B10) -> C[0,c0] += S4, C[0,c1] -= S4
        add_term([(a00,1)], [(b01,1),(b10,-1)], [(w00,1),(w10,-1)], col); col += 1
        # S5 = A11*(B10-B01) -> C[1,c0] += S5, C[1,c1] -= S5
        add_term([(a11,1)], [(b10,1),(b01,-1)], [(w01,1),(w11,-1)], col); col += 1
        # S6 = (A00+A01)*(B11) -> C[0,c1] += S6, C[0,c0] -= S6
        add_term([(a00,1),(a01,1)], [(b11,1)], [(w10,1),(w00,-1)], col); col += 1
        # S7 = (A10-A11)*(B00) -> C[1,c0] += S7, C[1,c1] -= S7
        add_term([(a10,1),(a11,-1)], [(b00,1)], [(w01,1),(w11,-1)], col); col += 1
        
        # === 4 shared cross terms: A[i,2]*(B[j,c0]+B[j,c1]) ===
        # These contribute to both C[i,c0] and C[i,c1]
        # X00 = A[0,2]*(B[0,c0]+B[0,c1]) -> C[0,c0] += X00, C[0,c1] += X00
        add_term([(a02,1)], [(b00,1),(b01,1)], [(w00,1),(w10,1)], col); col += 1
        # X01 = A[0,2]*(B[1,c0]+B[1,c1]) -> C[0,c0] += X01, C[0,c1] += X01
        add_term([(a02,1)], [(b10,1),(b11,1)], [(w00,1),(w10,1)], col); col += 1
        # X10 = A[1,2]*(B[0,c0]+B[0,c1]) -> C[1,c0] += X10, C[1,c1] += X10
        add_term([(a12,1)], [(b00,1),(b01,1)], [(w01,1),(w11,1)], col); col += 1
        # X11 = A[1,2]*(B[1,c0]+B[1,c1]) -> C[1,c0] += X11, C[1,c1] += X11
        add_term([(a12,1)], [(b10,1),(b11,1)], [(w01,1),(w11,1)], col); col += 1
        
        # === 5 row-2 terms: A[2,0:3] * B[0:3,c0:c1] ===
        # C[2,c0] = A20*B00 + A21*B10 + A22*B20
        # C[2,c1] = A20*B01 + A21*B11 + A22*B21
        # Use shared sums:
        # R1 = A[2,0]*(B[0,c0]+B[0,c1]) -> C[2,c0] += R1, C[2,c1] += R1
        add_term([(a20,1)], [(b00,1),(b01,1)], [(w02,1),(w12,1)], col); col += 1
        # R2 = A[2,1]*(B[1,c0]+B[1,c1]) -> C[2,c0] += R2, C[2,c1] += R2
        add_term([(a21,1)], [(b10,1),(b11,1)], [(w02,1),(w12,1)], col); col += 1
        # R3 = A[2,2]*(B[2,c0]+B[2,c1]) -> C[2,c0] += R3, C[2,c1] += R3
        add_term([(a22,1)], [(b20,1),(b21,1)], [(w02,1),(w12,1)], col); col += 1
        # R4 = A[2,0]*(B[2,c0]-B[0,c0]) -> correction for C[2,c0]
        # Actually we need: C[2,c0] should have A20*B00, but R1 already gives A20*B00+A20*B01
        # So we need to subtract A20*B01 from C[2,c0]... but that's in C[2,c1]'s territory.
        # Let me reconsider.
        # 
        # C[2,c0] = A20*B[0,c0] + A21*B[1,c0] + A22*B[2,c0]
        # C[2,c1] = A20*B[0,c1] + A21*B[1,c1] + A22*B[2,c1]
        #
        # R1 = A20*(B[0,c0]+B[0,c1]) contributes A20*B[0,c0] to C[2,c0] AND A20*B[0,c1] to C[2,c1] ✓
        # R2 = A21*(B[1,c0]+B[1,c1]) contributes A21*B[1,c0] to C[2,c0] AND A21*B[1,c1] to C[2,c1] ✓
        # R3 = A22*(B[2,c0]+B[2,c1]) contributes A22*B[2,c0] to C[2,c0] AND A22*B[2,c1] to C[2,c1] ✓
        #
        # Wait! That's only 3 terms and it's complete! No corrections needed!
        # But I allocated 5 slots. Let me use only 3 and leave 2 spare, or restructure.
        # Actually 3 is enough. So per half: 7 + 4 + 3 = 14 terms. Total = 28!
        # Let me verify this is correct.
        
        # Actually wait - I need to double-check the Strassen terms are correct.
        # Let me verify S1-S7 produce the correct 2x2*2x2 product.
        # 
        # Standard Strassen for 2x2*2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # My terms above don't match standard Strassen exactly. Let me redo.
        # 
        # For A[0:2,0:2] * B[0:2,c0:c1]:
        # Let A_sub = [[A00,A01],[A10,A11]], B_sub = [[B0c0,B0c1],[B1c0,B1c1]]
        # C_sub[0,c0] = A00*B0c0 + A01*B1c0
        # C_sub[0,c1] = A00*B0c1 + A01*B1c1
        # C_sub[1,c0] = A10*B0c0 + A11*B1c0
        # C_sub[1,c1] = A10*B0c1 + A11*B1c1
        #
        # Strassen:
        # M1 = (A00+A11)(B0c0+B1c0)
        # M2 = (A10+A11)(B0c0)
        # M3 = A00(B1c0-B1c1)
        # M4 = A11(B0c1-B0c0)
        # M5 = (A00+A01)(B1c1)
        # M6 = (A10-A00)(B0c0+B0c1)
        # M7 = (A01-A11)(B1c0+B1c1)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        return col
    
    # The above build_half has issues. Let me write it cleanly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_dict, v_dict, w_dict, col):
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
    
    for half in range(2):
        c0 = half * 2
        c1 = half * 2 + 1
        col_base = half * 16
        col = col_base
        
        # B indices
        b = lambda j, c: j * p + c
        # W indices  
        w = lambda i, c: c * n + i
        # A indices
        a = lambda i, j: i * m + j
        
        # === 7 Strassen terms ===
        # M1 = (A00+A11)*(B0c0+B1c0)
        add_term({a(0,0):1, a(1,1):1}, {b(0,c0):1, b(1,c0):1}, {w(0,c0):1, w(1,c0):-1}, col); col += 1
        # M2 = (A10+A11)*(B0c0)
        add_term({a(1,0):1, a(1,1):1}, {b(0,c0):1}, {w(1,c0):1, w(1,c1):-1}, col); col += 1
        # M3 = A00*(B1c0-B1c1)
        add_term({a(0,0):1}, {b(1,c0):1, b(1,c1):-1}, {w(0,c0):1, w(0,c1):-1}, col); col += 1
        # M4 = A11*(B0c1-B0c0)
        add_term({a(1,1):1}, {b(0,c1):1, b(0,c0):-1}, {w(0,c0):1, w(1,c0):1}, col); col += 1
        # M5 = (A00+A01)*(B1c1)
        add_term({a(0,0):1, a(0,1):1}, {b(1,c1):1}, {w(0,c0):-1, w(0,c1):1}, col); col += 1
        # M6 = (A10-A00)*(B0c0+B0c1)
        add_term({a(1,0):1, a(0,0):-1}, {b(0,c0):1, b(0,c1):1}, {w(1,c1):1}, col); col += 1
        # M7 = (A01-A11)*(B1c0+B1c1)
        add_term({a(0,1):1, a(1,1):-1}, {b(1,c0):1, b(1,c1):1}, {w(0,c0):1, w(1,c1):1}, col); col += 1
        
        # === 4 cross terms: A[i,2]*(B[j,c0]+B[j,c1]) ===
        # X0 = A[0,2]*(B[0,c0]+B[0,c1])
        add_term({a(0,2):1}, {b(0,c0):1, b(0,c1):1}, {w(0,c0):1, w(0,c1):1}, col); col += 1
        # X1 = A[0,2]*(B[1,c0]+B[1,c1])
        add_term({a(0,2):1}, {b(1,c0):1, b(1,c1):1}, {w(0,c0):1, w(0,c1):1}, col); col += 1
        # X2 = A[1,2]*(B[0,c0]+B[0,c1])
        add_term({a(1,2):1}, {b(0,c0):1, b(0,c1):1}, {w(1,c0):1, w(1,c1):1}, col); col += 1
        # X3 = A[1,2]*(B[1,c0]+B[1,c1])
        add_term({a(1,2):1}, {b(1,c0):1, b(1,c1):1}, {w(1,c0):1, w(1,c1):1}, col); col += 1
        
        # === 3 row-2 terms: A[2,j]*(B[j,c0]+B[j,c1]) ===
        # R0 = A[2,0]*(B[0,c0]+B[0,c1])
        add_term({a(2,0):1}, {b(0,c0):1, b(0,c1):1}, {w(2,c0):1, w(2,c1):1}, col); col += 1
        # R1 = A[2,1]*(B[1,c0]+B[1,c1])
        add_term({a(2,1):1}, {b(1,c0):1, b(1,c1):1}, {w(2,c0):1, w(2,c1):1}, col); col += 1
        # R2 = A[2,2]*(B[2,c0]+B[2,c1])
        add_term({a(2,2):1}, {b(2,c0):1, b(2,c1):1}, {w(2,c0):1, w(2,c1):1}, col); col += 1
        
        # Total per half: 7 + 4 + 3 = 14 terms
        # We have 16 slots per half, so 2 are unused (zeros)
    
    return {"U": U, "V": V, "W": W}

def _get_rank_34_baseline():
    """Return a verified rank-34 decomposition as fallback."""
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Left half (columns 0,1 of B)
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
    # Right half (columns 2,3 of B)
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}

def _get_naive_decomposition():
    """Return the naive rank-36 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p // 4  # 3*3*4/4 = 9? No, it's n*p*m = 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 130  # Safety margin
    
    # Try rank-32 construction first
    result_32 = _get_rank_32_construction()
    U_arr = np.array(result_32["U"], dtype=float)
    V_arr = np.array(result_32["V"], dtype=float)
    W_arr = np.array(result_32["W"], dtype=float)
    
    if _verify_decomposition(U_arr, V_arr, W_arr):
        return result_32
    
    # Fallback to rank-34 baseline
    result_34 = _get_rank_34_baseline()
    U_arr = np.array(result_34["U"], dtype=float)
    V_arr = np.array(result_34["V"], dtype=float)
    W_arr = np.array(result_34["W"], dtype=float)
    
    if _verify_decomposition(U_arr, V_arr, W_arr):
        return result_34
    
    # Final fallback: naive rank-36
    return _get_naive_decomposition()
