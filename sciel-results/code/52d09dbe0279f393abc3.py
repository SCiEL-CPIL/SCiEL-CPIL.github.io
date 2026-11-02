import time
import math

def _generate_strassen_3x3_terms():
    """
    Generate the 22 rank-1 terms for 3x3x3 multiplication using Strassen for the top-left 2x2 block
    and schoolbook for the remaining 5 outputs.
    Returns a list of (u, v, w) where u, v, w are lists of length 9.
    """
    terms = []
    
    # Helper to create a term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals, v_vals, w_vals are lists of (index, value)
        u = [0.0] * 9
        v = [0.0] * 9
        w = [0.0] * 9
        for idx, val in u_vals:
            u[idx] = val
        for idx, val in v_vals:
            v[idx] = val
        for idx, val in w_vals:
            w[idx] = val
        terms.append((u, v, w))

    # Indices for 3x3 matrix: i*3+j
    # A: 0..8, B: 0..8, C: 0..8
    
    # --- Strassen for Top-Left 2x2 (A[0:2, 0:2] * B[0:2, 0:2]) ---
    # Outputs: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # M1 = (A00 + A11) * (B00 + B11)
    # C00 += M1, C11 += M1
    add_term(
        [(0, 1), (4, 1)],  # A00, A11
        [(0, 1), (4, 1)],  # B00, B11
        [(0, 1), (4, 1)]   # C00, C11
    )
    
    # M2 = (A10 + A11) * B00
    # C10 += M2, C11 -= M2
    add_term(
        [(3, 1), (4, 1)],  # A10, A11
        [(0, 1)],          # B00
        [(3, 1), (4, -1)]  # C10, C11
    )
    
    # M3 = A00 * (B01 - B11)
    # C01 += M3, C11 += M3
    add_term(
        [(0, 1)],          # A00
        [(1, 1), (4, -1)], # B01, B11
        [(1, 1), (4, 1)]   # C01, C11
    )
    
    # M4 = A11 * (B10 - B00)
    # C00 += M4, C10 += M4
    add_term(
        [(4, 1)],          # A11
        [(3, 1), (0, -1)], # B10, B00
        [(0, 1), (3, 1)]   # C00, C10
    )
    
    # M5 = (A00 + A01) * B11
    # C00 -= M5, C01 += M5
    add_term(
        [(0, 1), (1, 1)],  # A00, A01
        [(4, 1)],          # B11
        [(0, -1), (1, 1)]  # C00, C01
    )
    
    # M6 = (A10 - A00) * (B00 + B01)
    # C11 += M6
    add_term(
        [(3, 1), (0, -1)], # A10, A00
        [(0, 1), (1, 1)],  # B00, B01
        [(4, 1)]           # C11
    )
    
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 += M7
    add_term(
        [(1, 1), (4, -1)], # A01, A11
        [(3, 1), (4, 1)],  # B10, B11
        [(0, 1)]           # C00
    )
    
    # --- Schoolbook for Remaining Outputs ---
    # C02 = A00*B02 + A01*B12 + A02*B22
    add_term([(0, 1)], [(2, 1)], [(2, 1)])
    add_term([(1, 1)], [(5, 1)], [(2, 1)])
    add_term([(2, 1)], [(8, 1)], [(2, 1)])
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    add_term([(3, 1)], [(2, 1)], [(5, 1)])
    add_term([(4, 1)], [(5, 1)], [(5, 1)])
    add_term([(5, 1)], [(8, 1)], [(5, 1)])
    
    # C20 = A20*B00 + A21*B10 + A22*B20
    add_term([(6, 1)], [(0, 1)], [(6, 1)])
    add_term([(7, 1)], [(3, 1)], [(6, 1)])
    add_term([(8, 1)], [(6, 1)], [(6, 1)])
    
    # C21 = A20*B01 + A21*B11 + A22*B21
    add_term([(6, 1)], [(1, 1)], [(7, 1)])
    add_term([(7, 1)], [(4, 1)], [(7, 1)])
    add_term([(8, 1)], [(7, 1)], [(7, 1)])
    
    # C22 = A20*B02 + A21*B12 + A22*B22
    add_term([(6, 1)], [(2, 1)], [(8, 1)])
    add_term([(7, 1)], [(5, 1)], [(8, 1)])
    add_term([(8, 1)], [(8, 1)], [(8, 1)])
    
    return terms

def _verify_3x3_terms(terms):
    """Verify the 22-term decomposition for 3x3x3."""
    for i in range(3):
        for j in range(3):
            for k in range(3):
                total = 0.0
                for u, v, w in terms:
                    total += u[i*3+j] * v[j*3+k] * w[k*3+i]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 145
    
    # Get the 22-term decomposition for 3x3x3
    terms_3x3 = _generate_strassen_3x3_terms()
    
    if _verify_3x3_terms(terms_3x3):
        # Construct 3x3x6 decomposition
        # Rank = 22 * 2 = 44
        rank = 44
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # First block: k=0,1,2
        for t_idx, (u, v, w) in enumerate(terms_3x3):
            for idx in range(9):
                U[idx][t_idx] = u[idx]
            for j in range(3):
                for k in range(3):
                    V[j*6+k][t_idx] = v[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*3+i][t_idx] = w[k*3+i]
                    
        # Second block: k=3,4,5
        for t_idx, (u, v, w) in enumerate(terms_3x3):
            col_offset = 22
            for idx in range(9):
                U[idx][col_offset + t_idx] = u[idx]
            for j in range(3):
                for k in range(3):
                    V[j*6+(k+3)][col_offset + t_idx] = v[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i][col_offset + t_idx] = w[k*3+i]
        
        # Final verification for 3x3x6
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(rank):
                        total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
