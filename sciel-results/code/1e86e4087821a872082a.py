import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank expected 66
    # We will build U, V, W as lists of lists or numpy arrays
    # Using numpy for construction is faster, then convert to list if needed?
    # The return type is {'U', 'V', 'W'}. The seed used lists.
    # I will use numpy arrays for internal construction and return them.
    # The evaluator likely accepts numpy arrays or lists. The seed returned lists.
    # I'll stick to lists to be safe, or check if numpy is better.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    # Let's use numpy for efficiency in building.
    # Rank is 66.
    R = 66
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col_idx = 0
    
    # Helper to add a 2x2x2 Strassen term
    # A_rows: [r0, r1], A_cols: [c0, c1]
    # B_rows: [c0, c1], B_cols: [k0, k1]
    # Output C_rows: [r0, r1], C_cols: [k0, k1]
    def add_strassen(r0, r1, c0, c1, k0, k1):
        nonlocal col_idx
        # Local indices in U (n*m), V (m*p), W (p*n)
        # U index: row * m + col
        # V index: row * p + col
        # W index: row * n + col
        
        # Strassen formulas
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # We need to express each m_i as sum of rank-1 tensors.
        # Actually, we just add the coefficients to U, V, W.
        
        # m1
        # (a+d)(e+h) = a e + a h + d e + d h
        # This is not a single rank 1 term in the global tensor, 
        # but in the Strassen decomposition, m1 is ONE multiplication.
        # So we create ONE column in U, V, W for m1.
        
        # U entries for m1:
        # a is at (r0, c0), d is at (r1, c1)
        # So U[r0*m+c0] += 1, U[r1*m+c1] += 1
        # V entries for m1:
        # e is at (c0, k0), h is at (c1, k1)
        # So V[c0*p+k0] += 1, V[c1*p+k1] += 1
        # W entries for m1:
        # m1 contributes to C11 and C22.
        # C11 = m1 + ...
        # C22 = m1 - ...
        # So W[k0*n+r0] += 1 (for C11)
        # W[k1*n+r1] += 1 (for C22)
        
        U[r0*m+c0, col_idx] += 1
        U[r1*m+c1, col_idx] += 1
        V[c0*p+k0, col_idx] += 1
        V[c1*p+k1, col_idx] += 1
        W[k0*n+r0, col_idx] += 1
        W[k1*n+r1, col_idx] += 1
        col_idx += 1
        
        # m2 = (c+d)e
        # c is (r1, c0), d is (r1, c1)
        # e is (c0, k0)
        # C21 = m2 + ...
        U[r1*m+c0, col_idx] += 1
        U[r1*m+c1, col_idx] += 1
        V[c0*p+k0, col_idx] += 1
        W[k0*n+r1, col_idx] += 1 # C21 is at row r1, col k0
        col_idx += 1
        
        # m3 = a(f-h)
        # a is (r0, c0)
        # f is (c0, k1), h is (c1, k1)
        # C12 = m3 + ...
        U[r0*m+c0, col_idx] += 1
        V[c0*p+k1, col_idx] += 1
        V[c1*p+k1, col_idx] -= 1
        W[k1*n+r0, col_idx] += 1 # C12 is at row r0, col k1
        col_idx += 1
        
        # m4 = d(g-e)
        # d is (r1, c1)
        # g is (c1, k0), e is (c0, k0)
        # C11 = ... + m4, C21 = ... + m4
        U[r1*m+c1, col_idx] += 1
        V[c1*p+k0, col_idx] += 1
        V[c0*p+k0, col_idx] -= 1
        W[k0*n+r0, col_idx] += 1 # C11
        W[k0*n+r1, col_idx] += 1 # C21
        col_idx += 1
        
        # m5 = (a+b)h
        # a is (r0, c0), b is (r0, c1)
        # h is (c1, k1)
        # C11 = ... - m5, C12 = ... + m5
        U[r0*m+c0, col_idx] += 1
        U[r0*m+c1, col_idx] += 1
        V[c1*p+k1, col_idx] += 1
        W[k0*n+r0, col_idx] -= 1 # C11
        W[k1*n+r0, col_idx] += 1 # C12
        col_idx += 1
        
        # m6 = (c-a)(e+f)
        # c is (r1, c0), a is (r0, c0)
        # e is (c0, k0), f is (c0, k1)
        # C22 = ... + m6
        U[r1*m+c0, col_idx] += 1
        U[r0*m+c0, col_idx] -= 1
        V[c0*p+k0, col_idx] += 1
        V[c0*p+k1, col_idx] += 1
        W[k1*n+r1, col_idx] += 1 # C22
        col_idx += 1
        
        # m7 = (b-d)(g+h)
        # b is (r0, c1), d is (r1, c1)
        # g is (c1, k0), h is (c1, k1)
        # C11 = ... + m7
        U[r0*m+c1, col_idx] += 1
        U[r1*m+c1, col_idx] -= 1
        V[c1*p+k0, col_idx] += 1
        V[c1*p+k1, col_idx] += 1
        W[k0*n+r0, col_idx] += 1 # C11
        col_idx += 1

    # Helper to add a 1x2x2 direct term
    # A is 1x2 at row r, cols c0, c1
    # B is 2x2 at rows c0, c1, cols k0, k1
    # C is 1x2 at row r, cols k0, k1
    def add_direct_1x2x2(r, c0, c1, k0, k1):
        nonlocal col_idx
        
        # C_k0 = A_c0 * B_c0_k0 + A_c1 * B_c1_k0
        # C_k1 = A_c0 * B_c0_k1 + A_c1 * B_c1_k1
        
        # Term 1: A_c0 * B_c0_k0 -> C_k0
        U[r*m+c0, col_idx] += 1
        V[c0*p+k0, col_idx] += 1
        W[k0*n+r, col_idx] += 1
        col_idx += 1
        
        # Term 2: A_c1 * B_c1_k0 -> C_k0
        U[r*m+c1, col_idx] += 1
        V[c1*p+k0, col_idx] += 1
        W[k0*n+r, col_idx] += 1
        col_idx += 1
        
        # Term 3: A_c0 * B_c0_k1 -> C_k1
        U[r*m+c0, col_idx] += 1
        V[c0*p+k1, col_idx] += 1
        W[k1*n+r, col_idx] += 1
        col_idx += 1
        
        # Term 4: A_c1 * B_c1_k1 -> C_k1
        U[r*m+c1, col_idx] += 1
        V[c1*p+k1, col_idx] += 1
        W[k1*n+r, col_idx] += 1
        col_idx += 1

    # Partitioning:
    # A rows: 0,1 (top), 2 (bot)
    # A cols: 0,1 (left), 2,3 (right)
    # B rows: 0,1 (top), 2,3 (bot)
    # B cols: 0,1 (L), 2,3 (M), 4,5 (R)
    
    # 2x2x2 Terms (Strassen)
    # 1. A_TL (0,1 | 0,1) * B_TL (0,1 | 0,1)
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # 2. A_TR (0,1 | 2,3) * B_BL (2,3 | 0,1)
    # Note: B rows are 2,3. B cols are 0,1.
    # A cols are 2,3.
    # This matches.
    add_strassen(0, 1, 2, 3, 0, 1)
    
    # 3. A_TL (0,1 | 0,1) * B_TM (0,1 | 2,3)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # 4. A_TR (0,1 | 2,3) * B_BM (2,3 | 2,3)
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # 5. A_TL (0,1 | 0,1) * B_TR (0,1 | 4,5)
    add_strassen(0, 1, 0, 1, 4, 5)
    
    # 6. A_TR (0,1 | 2,3) * B_BR (2,3 | 4,5)
    add_strassen(0, 1, 2, 3, 4, 5)
    
    # 1x2x2 Terms (Direct)
    # 7. A_BL (2 | 0,1) * B_TL (0,1 | 0,1)
    add_direct_1x2x2(2, 0, 1, 0, 1)
    
    # 8. A_BR (2 | 2,3) * B_BL (2,3 | 0,1)
    add_direct_1x2x2(2, 2, 3, 0, 1)
    
    # 9. A_BL (2 | 0,1) * B_TM (0,1 | 2,3)
    add_direct_1x2x2(2, 0, 1, 2, 3)
    
    # 10. A_BR (2 | 2,3) * B_BM (2,3 | 2,3)
    add_direct_1x2x2(2, 2, 3, 2, 3)
    
    # 11. A_BL (2 | 0,1) * B_TR (0,1 | 4,5)
    add_direct_1x2x2(2, 0, 1, 4, 5)
    
    # 12. A_BR (2 | 2,3) * B_BR (2,3 | 4,5)
    add_direct_1x2x2(2, 2, 3, 4, 5)
    
    # Check col_idx
    if col_idx != R:
        # If it doesn't match, we might have miscalculated.
        # But 6*7 + 6*4 = 42 + 24 = 66.
        pass
        
    # Convert to lists if required, or return numpy arrays.
    # The seed returned lists. I will return lists to be safe.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
