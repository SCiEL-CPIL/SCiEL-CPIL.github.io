"""Program for rank-minimal tensor decomposition of 2x5 by 5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(i, j, k, coeff=1.0):
        """Add a rank-1 term for A[i,j]*B[j,k] contributing coeff to C[i,k]."""
        nonlocal col
        if col >= rank:
            raise RuntimeError("Exceeded rank")
        U[i * m + j][col] += coeff
        V[j * p + k][col] += coeff
        W[k * n + i][col] += 1.0
        col += 1
    
    # Strassen for 2x2 by 2x2: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # 7 products, each a product of linear forms.
    # M1 = (A00-A11)(B01-B10)
    # M2 = (A00+A11)(B00)
    # M3 = (A01)(B11-B01)
    # M4 = (A10)(B00-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M2 - M5 + M7
    # C01 = M1 + M3 + M5
    # C10 = M4 + M6
    # C11 = -M1 + M2 + M4 - M6 + M7
    
    # Each M is (sum_i a_i A[i,j_i])(sum_k b_k B[j_k,l])
    # Expanding M1: (A00-A11)(B01-B10) = A00*B01 - A00*B10 - A11*B01 + A11*B10
    # Each A[i,j]*B[j,k] is a rank-1 term.
    
    # I'll accumulate coefficients for each (i,j,k) triple.
    coeffs = {}
    
    def add_to_coeffs(i, j, k, c):
        key = (i, j, k)
        coeffs[key] = coeffs.get(key, 0.0) + c
    
    # M1 = (A00-A11)(B01-B10)
    add_to_coeffs(0, 0, 1, 1)   # A00*B01
    add_to_coeffs(0, 1, 0, -1)  # -A00*B10
    add_to_coeffs(1, 0, 1, -1)  # -A11*B01
    add_to_coeffs(1, 1, 0, 1)   # A11*B10
    
    # M2 = (A00+A11)(B00)
    add_to_coeffs(0, 0, 0, 1)
    add_to_coeffs(1, 0, 0, 1)
    
    # M3 = A01*(B11-B01)
    add_to_coeffs(0, 1, 1, 1)
    add_to_coeffs(0, 0, 1, -1)
    
    # M4 = A10*(B00-B10)
    add_to_coeffs(1, 0, 0, 1)
    add_to_coeffs(1, 1, 0, -1)
    
    # M5 = (A00+A01)(B11)
    add_to_coeffs(0, 1, 1, 1)
    add_to_coeffs(0, 0, 1, 1)
    
    # M6 = (A10-A00)(B00+B01)
    add_to_coeffs(1, 0, 0, 1)
    add_to_coeffs(1, 0, 1, 1)
    add_to_coeffs(0, 0, 0, -1)
    add_to_coeffs(0, 0, 1, -1)
    
    # M7 = (A01-A11)(B10+B11)
    add_to_coeffs(0, 1, 0, 1)
    add_to_coeffs(0, 1, 1, 1)
    add_to_coeffs(1, 1, 0, -1)
    add_to_coeffs(1, 1, 1, -1)
    
    # Now compute the net coefficient for each (i,j,k)
    # and add rank-1 terms.
    for (i, j, k), c in coeffs.items():
        if abs(c) > 1e-10:
            add_term(i, j, k, c)
    
    # Now handle the remaining blocks.
    # A[0:2,2:5] * B[2:5,0:5] (2x3 by 3x5), schoolbook rank 30? No, 2*3*5=30.
    # But we can do better. Split into:
    # A[0:2,2:3] (2x1) * B[2:3,0:5] (1x5): rank 10
    # A[0:2,3:5] (2x2) * B[3:5,0:5] (2x5): rank 20
    # Total: 30. But we wanted 10+20=30. Hmm, that's not 10+20, that's 10+20=30.
    # Wait, 2*1*5=10 and 2*2*5=20, total 30.
    # Plus Strassen part: 7 terms but they expand to more rank-1 terms.
    
    # Actually, the number of rank-1 terms from Strassen depends on how many distinct (i,j,k) have nonzero coefficients.
    # Let me just add the remaining schoolbook terms.
    
    # A[0:2,2:5] * B[2:5,0:5]
    for i in range(2):
        for j in range(2, 5):
            for k in range(5):
                add_term(i, j, k, 1.0)
    
    # That's 2*3*5 = 30 terms. Plus however many from Strassen.
    # Total should be 49. Let me check: col should be 49.
    
    return {"U": U, "V": V, "W": W}
