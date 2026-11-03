"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]
import time
_START = time.time()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Block-Strassen decomposition: rank 84.
    # Split A (4x4) into 2x2 blocks of 2x2.
    # Split B (4x6) into 2x2 blocks of 2x3.
    # Use 7 Strassen products, each (2x2)*(2x3) with rank 12.
    # Total rank = 7 * 12 = 84.
    
    # Each Strassen product contributes to one or two C blocks.
    # For each local term (ia, jb, ka) in the (2x2)*(2x3) space,
    # the A-part is a linear form (sum of A entries), B-part is a linear form (sum of B entries).
    # The product of two linear forms is a sum of rank-1 terms.
    # Each rank-1 term has U=coeff_a, V=coeff_b, W=1 (or similar).
    
    # Let me define the 7 products with their A and B linear forms.
    # For each product, I list the (ia, jb, ka) terms and their A/B coefficients.
    
    # P1 = (A11+A22)(B11+B22)
    # A[ia,jb] + A[ia+2,jb+2] times B[jb,ka] + B[jb+2,ka+3]
    # Expands to 4 terms per (ia,jb,ka):
    #   (A[ia,jb]*B[jb,ka] + A[ia,jb]*B[jb+2,ka+3] + A[ia+2,jb+2]*B[jb,ka] + A[ia+2,jb+2]*B[jb+2,ka+3])
    # Each contributes to C11[ka,ia] and C22[ka,ia].
    
    # I'll build the full list of rank-1 terms.
    terms = []  # list of (a_idx, b_idx, c_idx, coeff)
    
    def add_term(a_row, a_col, b_row, b_col, c_row, c_col, coeff):
        a_idx = a_row * m + a_col
        b_idx = b_row * p + b_col
        c_idx = c_row * n + c_col
        terms.append((a_idx, b_idx, c_idx, coeff))
    
    # P1: (A11+A22)(B11+B22) -> C11 and C22
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia, jb) and (ia+2, jb+2)
                # B entries: (jb, ka) and (jb+2, ka+3)
                for a_row, a_col in [(ia, jb), (ia+2, jb+2)]:
                    for b_row, b_col in [(jb, ka), (jb+2, ka+3)]:
                        add_term(a_row, a_col, b_row, b_col, ka, ia, 1)          # C11
                        add_term(a_row, a_col, b_row, b_col, ka+2, ia+2, 1)        # C22
    
    # P2: (A21+A22)B11 -> C21 and C22
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia+2, jb) and (ia+2, jb+2)
                # B entries: (jb, ka)
                for a_row, a_col in [(ia+2, jb), (ia+2, jb+2)]:
                    b_row, b_col = jb, ka
                    add_term(a_row, a_col, b_row, b_col, ka+2, ia, 1)               # C21
                    add_term(a_row, a_col, b_row, b_col, ka+2, ia+2, -1)             # C22
    
    # P3: A11(B12-B22) -> C12 and C22
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia, jb)
                # B entries: (jb, ka+3) and (jb+2, ka+3)
                a_row, a_col = ia, jb
                for b_row, b_col, coeff in [(jb, ka+3, 1), (jb+2, ka+3, -1)]:
                    add_term(a_row, a_col, b_row, b_col, ka, ia+2, coeff)           # C12
                    add_term(a_row, a_col, b_row, b_col, ka+2, ia+2, coeff)          # C22
    
    # P4: A22(B21-B11) -> C11 and C21
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia+2, jb+2)
                # B entries: (jb+2, ka) and (jb, ka)
                a_row, a_col = ia+2, jb+2
                for b_row, b_col, coeff in [(jb+2, ka, 1), (jb, ka, -1)]:
                    add_term(a_row, a_col, b_row, b_col, ka, ia, coeff)              # C11
                    add_term(a_row, a_col, b_row, b_col, ka+2, ia, coeff)             # C21
    
    # P5: (A11+A12)B22 -> C11 and C12
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia, jb) and (ia, jb+2)
                # B entries: (jb+2, ka+3)
                b_row, b_col = jb+2, ka+3
                for a_row, a_col in [(ia, jb), (ia, jb+2)]:
                    add_term(a_row, a_col, b_row, b_col, ka, ia, -1)                 # C11
                    add_term(a_row, a_col, b_row, b_col, ka, ia+2, 1)                # C12
    
    # P6: (A21-A11)(B11+B12) -> C22
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia+2, jb) and (ia, jb)
                # B entries: (jb, ka) and (jb, ka+3)
                for a_row, a_col, a_coeff in [(ia+2, jb, 1), (ia, jb, -1)]:
                    for b_row, b_col, b_coeff in [(jb, ka, 1), (jb, ka+3, 1)]:
                        add_term(a_row, a_col, b_row, b_col, ka+2, ia+2, a_coeff * b_coeff)  # C22
    
    # P7: (A12-A22)(B21+B22) -> C11 and C22
    for ia in range(2):
        for jb in range(2):
            for ka in range(3):
                # A entries: (ia, jb+2) and (ia+2, jb+2)
                # B entries: (jb+2, ka) and (jb+2, ka+3)
                for a_row, a_col, a_coeff in [(ia, jb+2, 1), (ia+2, jb+2, -1)]:
                    for b_row, b_col, b_coeff in [(jb+2, ka, 1), (jb+2, ka+3, 1)]:
                        add_term(a_row, a_col, b_row, b_col, ka, ia, a_coeff * b_coeff)    # C11
                        add_term(a_row, a_col, b_row, b_col, ka+2, ia+2, a_coeff * b_coeff)  # C22
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (a_idx, b_idx, c_idx, coeff) in enumerate(terms):
        U[a_idx][col] = coeff
        V[b_idx][col] = 1
        W[c_idx][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

