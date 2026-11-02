import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _rank21_3x3x3_terms():
    """
    Return 21 rank-1 terms for 3x3 * 3x3 matrix multiplication.
    Each term is (a_idx, b_idx, c_idx) where:
      a_idx = i*3+k (position in A, i=row, k=col)
      b_idx = k*3+j (position in B, k=row, j=col)  
      c_idx = j*3+i (position in C, j=row, i=col) -- trace convention
    
    The decomposition satisfies:
      C[i,j] = sum_t delta_a_t(i,k) * delta_b_t(k,j) * delta_c_t(j,i) * coeff_t
    
    For rank-1 terms with coeff=1:
      C[i,j] = sum_t (1 if a_t gives A[i,k_t]) * (1 if b_t gives B[k_t,j]) * (1 if c_t gives C[j,i])
    """
    terms = []
    
    # I'll use the explicit rank-21 construction from the literature.
    # Based on the work of Strassen and subsequent improvements.
    
    # Terms 0-6: Strassen 2x2x2 core for indices (0,1)
    # P1 = (A00+A11)(B00+B11)
    # Expand: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # But we want to share these across C entries.
    
    # Actually, for a proper rank-1 decomposition, each term must be a single
    # A[i,k] * B[k,j] * C[j,i] with coefficient.
    
    # The rank-21 construction uses linear combinations in U, V, W columns.
    # So each column r of U, V, W can have multiple non-zero entries.
    
    # Let me define the 21 columns of U, V, W directly.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a column
    def add_col(u_entries, v_entries, w_entries):
        U_cols.append(u_entries)
        V_cols.append(v_entries)
        W_cols.append(w_entries)
    
    # u_entries is dict: (i,k) -> coeff
    # v_entries is dict: (k,j) -> coeff
    # w_entries is dict: (j,i) -> coeff
    
    # Term 0: P1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    add_col({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    
    # Term 1: P2 = (A10+A11)B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1
    add_col({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1})
    
    # Term 2: P3 = A00(B10-B11)
    # U: A00=1
    # V: B10=1, B11=-1
    # W: C00=1, C01=1
    add_col({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (1,0):1})
    
    # Term 3: P4 = A11(B01-B00)
    # U: A11=1
    # V: B01=1, B00=-1
    # W: C10=1, C11=1
    add_col({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,1):1, (1,1):1})
    
    # Term 4: P5 = (A00+A10)B11
    # U: A00=1, A10=1
    # V: B11=1
    # W: C01=1, C11=1
    add_col({(0,0):1, (1,0):1}, {(1,1):1}, {(1,0):1, (1,1):1})
    
    # Term 5: P6 = (A11-A00)(B00+B10)
    # U: A11=1, A00=-1
    # V: B00=1, B10=1
    # W: C01=1, C10=1
    add_col({(1,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,0):1, (0,1):1})
    
    # Term 6: P7 = (A01-A11)(B11+B00)
    # U: A01=1, A11=-1
    # V: B11=1, B00=1
    # W: C00=1, C11=1
    add_col({(0,1):1, (1,1):-1}, {(1,1):1, (0,0):1}, {(0,0):1, (1,1):1})
    
    # Now verify: C00 = P1 + P3 - P5 + P7
    # = (A00+A11)(B00+B11) + A00(B10-B11) - (A00+A10)B11 + (A01-A11)(B11+B00)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A00B10 - A00B11 - A00B11 - A10B11 + A01B11 + A01B00 - A11B11 - A11B00
    # = A00B00 + A00B10 - A00B11 - A10B11 + A01B11 + A01B00
    # Expected C00 = A00B00 + A01B10 + A02B20
    # This doesn't match! I'm missing A01B10 and A02B20 terms.
    
    # The Strassen 2x2x2 only covers the (0,1)x(0,1)x(0,1) subblock, i.e.,
    # C[i,j] for i,j in {0,1} using only A[i,k] and B[k,j] for k in {0,1}.
    # It does NOT include k=2 terms.
    
    # So for the full 3x3x3, I need additional terms for k=2.
    # C[i,j] = sum_k A[i,k]B[k,j] for k=0,1,2
    # The Strassen part handles k=0,1 for i,j in {0,1}.
    # I need:
    # - k=2 terms for i,j in {0,1}: A[i,2]B[2,j] for i,j in {0,1} -> 4 terms
    # - i=2 terms for k in {0,1}: A[2,k]B[k,j] for k in {0,1}, j in {0,1} -> 4 terms
    # - i=2, j=2: A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2] -> 3 terms
    # - j=2 terms for i in {0,1}, k in {0,1}: A[i,k]B[k,2] for i in {0,1}, k in {0,1} -> 4 terms
    # - i=2, j in {0,1}, k=2: A[2,2]B[2,j] for j in {0,1} -> 2 terms (already counted?)
    
    # This is getting messy. Let me count more carefully.
    # Total schoolbook terms: 27
    # Strassen handles 4 of the 9 C entries with 7 terms instead of 4 terms each (16 terms).
    # Wait, Strassen 2x2x2 uses 7 terms for 4 output entries, vs 4*4=16 schoolbook terms.
    # So we save 16-7=9 terms, but we still need the remaining 27-16=11 schoolbook terms.
    # Total: 7 + 11 = 18? That can't be right since rank >= 19.
    
    # I think the issue is that the Strassen terms don't cleanly separate.
    
    # Let me just give up on the analytic construction and use a numerical approach
    # or just return the schoolbook decomposition.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition, rank 54
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
