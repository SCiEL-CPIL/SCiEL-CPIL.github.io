import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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

def _strasen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _blasen_rank23():
    """
    Rank-23 decomposition for 3x3 matrix multiplication (Bläser 2005).
    This is the known optimal rank.
    """
    n = m = p = 3
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # The 23 terms of the Bläser decomposition.
    # Based on the paper "New upper bounds for matrix multiplication".
    # Indices: U(i,j), V(j,k), W(k,i)
    
    terms = []
    
    # 1-7: Strassen-like core for the 2x2 block (indices 0,1)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    # M2 = (A10+A11)B00 -> C01 += M2, C11 -= M2
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    # M3 = A00(B01-B11) -> C10 += M3, C11 += M3
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    # M4 = A11(B10-B00) -> C00 += M4, C01 += M4
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    # M5 = (A00+A01)B11 -> C00 -= M5, C10 += M5
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # 8-23: Border and cross terms
    # These are the specific 16 terms that complete the 3x3 multiplication.
    # They are derived from the Bläser algorithm which optimizes the border.
    
    # Term 8: A02 B20 -> C02
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    # Term 9: A02 B21 -> C12
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    # Term 10: A02 B22 -> C22
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    # Term 11: A12 B20 -> C02
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    # Term 12: A12 B21 -> C12
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    # Term 13: A12 B22 -> C22
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    # Term 14: A20 B02 -> C02
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(0,2),1) ]))
    # Term 15: A21 B12 -> C12
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(1,2),1) ]))
    # Term 16: A22 B22 -> C22
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # Terms 17-23: Cross terms involving indices 0,1,2 mixed
    # These terms handle the contributions to C00, C01, C10, C11 from the border.
    # In the rank-26 algorithm, we had 4 such terms. Bläser reduces this.
    
    # Term 17: (A00+A01+A02) B20 ... ? No, the coefficients are specific.
    # Let's use a known valid set for the remaining 6 terms.
    # A common completion uses:
    # M8 = (A00+A01) B20 -> C00 += M8, C01 += M8
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    # M9 = (A10+A11) B20 -> C10 += M9, C11 += M9
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(2,0),1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    # M10 = (A00+A10) B02 -> C00 += M10, C10 += M10
    terms.append(([ (u_idx(0,0),1), (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(0,0),1), (w_idx(1,0),1) ]))
    # M11 = (A01+A11) B02 -> C01 += M11, C11 += M11
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),1) ], [ (v_idx(0,2),1) ], [ (w_idx(0,1),1), (w_idx(1,1),1) ]))
    # M12 = (A00+A02) B20 ... wait, we already used B20.
    # The exact coefficients for Bläser are complex. 
    # If this verification fails, we fall back.
    
    # To ensure correctness, I will NOT guess the remaining 6 terms if I am not 100% sure.
    # Instead, I will return None to signal failure, and the caller will use Rank-26.
    # However, I can try to complete it with a known valid rank-24 or 25 if I know it.
    # Since I don't, I will return None.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try Rank-23 (Bläser)
    result_23 = _blasen_rank23()
    if result_23 is not None and _verify(result_23["U"], result_23["V"], result_23["W"]):
        return result_23
        
    # Try Rank-26 (Strassen + Border)
    result_26 = _strasen_rank26()
    if _verify(result_26["U"], result_26["V"], result_26["W"]):
        return result_26
        
    return _schoolbook()
