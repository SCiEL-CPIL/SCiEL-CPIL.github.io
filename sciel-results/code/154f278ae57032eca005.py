"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    # C = A1*B1 + A2*B2
    # Each Ai is 2x2, Bi is 2x7.
    # Split Bi into B_i1 (cols 0,1), B_i2 (cols 2,3), B_i3 (cols 4,5), B_i4 (col 6)
    # C_i = Ai*B_i1 + Ai*B_i2 + Ai*B_i3 + Ai*B_i4
    # Ai*B_i1, Ai*B_i2, Ai*B_i3 are 2x2 * 2x2 (Rank 7 each)
    # Ai*B_i4 is 2x2 * 2x1 (Rank 4 each)
    # Total per i: 3*7 + 4 = 25.
    # Total rank: 25 + 25 = 50.
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1
        
    # We need to map local indices to global indices
    # A global: i in [0,1], j in [0,3]
    # B global: j in [0,3], k in [0,6]
    # C global: k in [0,6], i in [0,1] -> index k*n + i
    
    # Strassen 2x2x2 decomposition
    # Inputs: a,b,c,d (A), e,f,g,h (B)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1] (Local 2x2)
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (a+b)g
    # P11 = M1 - M4 + M5 - M7
    # P12 = M2 + M4
    # P21 = M3 + M5
    # P22 = M1 - M2 + M6 - M7
    
    def add_strassen_block(a_i, a_j_offset, b_j_offset, b_k_offset):
        nonlocal col
        # a_i is the row offset for A (0 or 1)
        # a_j_offset is col offset for A (0 or 2)
        # b_j_offset is row offset for B (0 or 2)
        # b_k_offset is col offset for B (0, 2, 4)
        
        # Local A indices: (0,0), (0,1), (1,0), (1,1)
        # Global A: (a_i, a_j_offset + 0), (a_i, a_j_offset + 1), (a_i+1, a_j_offset + 0), (a_i+1, a_j_offset + 1)
        # Note: A is 2x4. a_i is 0 or 1.
        
        a00_idx = a_i * m + (a_j_offset + 0)
        a01_idx = a_i * m + (a_j_offset + 1)
        a10_idx = (a_i + 1) * m + (a_j_offset + 0)
        a11_idx = (a_i + 1) * m + (a_j_offset + 1)
        
        # Local B indices: (0,0), (0,1), (1,0), (1,1)
        # Global B: (b_j_offset + 0, b_k_offset + 0), etc.
        b00_idx = (b_j_offset + 0) * p + (b_k_offset + 0)
        b01_idx = (b_j_offset + 0) * p + (b_k_offset + 1)
        b10_idx = (b_j_offset + 1) * p + (b_k_offset + 0)
        b11_idx = (b_j_offset + 1) * p + (b_k_offset + 1)
        
        # Local C indices: (0,0), (0,1), (1,0), (1,1)
        # Global C: k in [b_k_offset, b_k_offset+1], i in [a_i, a_i+1]
        # Index: k*n + i
        c00_idx = (b_k_offset + 0) * n + a_i
        c01_idx = (b_k_offset + 1) * n + a_i
        c10_idx = (b_k_offset + 0) * n + (a_i + 1)
        c11_idx = (b_k_offset + 1) * n + (a_i + 1)
        
        # M1: (a00+a11)(b00+b11)
        # We need to expand (a00+a11) and (b00+b11) into rank-1 terms?
        # No, Strassen mults are products of sums.
        # (a00+a11) is not a single variable.
        # The tensor decomposition requires U, V, W to be linear in A, B.
        # The "mults" in Strassen are intermediate values, not direct multiplications of input variables.
        # Wait, the tensor decomposition is $\sum u_r \otimes v_r \otimes w_r$.
        # This corresponds to $\sum_r U_{,r} V_{,r} W_{,r}$.
        # Each term is a product of one A-linear form, one B-linear form, and one C-linear form.
        # Strassen's algorithm computes 7 intermediate products.
        # Each product is of the form (linear combo of A) * (linear combo of B).
        # So each Strassen mult corresponds to 1 rank-1 term in the decomposition?
        # Yes.
        # M1 = (A_{00}+A_{11}) (B_{00}+B_{11}).
        # This is one term in the decomposition.
        # U vector for this term: 1 at A00, 1 at A11.
        # V vector for this term: 1 at B00, 1 at B11.
        # W vector for this term: coefficients for C.
        
        # Let's define the 7 terms.
        # Term 1: (a00+a11) * (b00+b11)
        # U: a00:1, a11:1
        # V: b00:1, b11:1
        # W: c00:1, c01:-1, c10:1, c11:-1 (from P11=M1..., P12=..., P21=..., P22=M1...)
        # P11 coeff for M1: 1
        # P12 coeff for M1: 0
        # P21 coeff for M1: 0
        # P22 coeff for M1: 1
        
        def add_strassen_term(u_coefs, v_coefs, w_coefs):
            nonlocal col
            if col >= rank: return
            for idx, val in u_coefs.items():
                U[idx][col] += val
            for idx, val in v_coefs.items():
                V[idx][col] += val
            for idx, val in w_coefs.items():
                W[idx][col] += val
            col += 1
            
        # M1
        add_strassen_term(
            {a00_idx: 1, a11_idx: 1},
            {b00_idx: 1, b11_idx: 1},
            {c00_idx: 1, c11_idx: 1} # P11, P22
        )
        
        # M2: (c+d)e -> (a10+a11) * b00
        add_strassen_term(
            {a10_idx: 1, a11_idx: 1},
            {b00_idx: 1},
            {c01_idx: 1, c11_idx: -1} # P12=M2..., P22=-M2...
        )
        
        # M3: a(f-h) -> a00 * (b01-b10)
        add_strassen_term(
            {a00_idx: 1},
            {b01_idx: 1, b10_idx: -1},
            {c10_idx: 1} # P21=M3...
        )
        
        # M4: d(g-f) -> a11 * (b10-b01)
        add_strassen_term(
            {a11_idx: 1},
            {b10_idx: 1, b01_idx: -1},
            {c00_idx: -1, c01_idx: 1} # P11=-M4..., P12=M4...
        )
        
        # M5: (a+b)(e+f) -> (a00+a01) * (b00+b01)
        add_strassen_term(
            {a00_idx: 1, a01_idx: 1},
            {b00_idx: 1, b01_idx: 1},
            {c00_idx: 1, c10_idx: 1} # P11=M5..., P21=M5...
        )
        
        # M6: (c-d)(g+h) -> (a10-a11) * (b10+b11)
        add_strassen_term(
            {a10_idx: 1, a11_idx: -1},
            {b10_idx: 1, b11_idx: 1},
            {c11_idx: 1} # P22=M6...
        )
        
        # M7: (a+b)g -> (a00+a01) * b10
        add_strassen_term(
            {a00_idx: 1, a01_idx: 1},
            {b10_idx: 1},
            {c00_idx: -1, c11_idx: -1} # P11=-M7..., P22=-M7...
        )
        
    # Schoolbook 2x2x1 decomposition
    # A 2x2, B 2x1.
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    def add_schoolbook_2x2x1_block(a_i, a_j_offset, b_j_offset, b_k):
        nonlocal col
        a00_idx = a_i * m + (a_j_offset + 0)
        a01_idx = a_i * m + (a_j_offset + 1)
        a10_idx = (a_i + 1) * m + (a_j_offset + 0)
        a11_idx = (a_i + 1) * m + (a_j_offset + 1)
        
        b0_idx = (b_j_offset + 0) * p + b_k
        b1_idx = (b_j_offset + 1) * p + b_k
        
        c0_idx = b_k * n + a_i
        c1_idx = b_k * n + (a_i + 1)
        
        def add_term_simple(u_idx, v_idx, w_idx, coeff):
            nonlocal col
            if col >= rank: return
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
            col += 1
            
        # a00*b0 -> c0
        add_term_simple(a00_idx, b0_idx, c0_idx, 1)
        # a01*b1 -> c0
        add_term_simple(a01_idx, b1_idx, c0_idx, 1)
        # a10*b0 -> c1
        add_term_simple(a10_idx, b0_idx, c1_idx, 1)
        # a11*b1 -> c1
        add_term_simple(a11_idx, b1_idx, c1_idx, 1)

    # Block 1: A1 (a_i=0, a_j_offset=0), B1 (b_j_offset=0)
    # B1 cols 0,1 (Strassen)
    add_strassen_block(0, 0, 0, 0)
    # B1 cols 2,3 (Strassen)
    add_strassen_block(0, 0, 0, 2)
    # B1 cols 4,5 (Strassen)
    add_strassen_block(0, 0, 0, 4)
    # B1 col 6 (Schoolbook)
    add_schoolbook_2x2x1_block(0, 0, 0, 6)
    
    # Block 2: A2 (a_i=0? No, A2 is rows 0,1 but cols 2,3. Wait.
    # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
    # My a_i parameter in add_strassen_block is the row index of A.
    # A1 uses rows 0,1. A2 uses rows 0,1.
    # So a_i is always 0?
    # Yes, A has only 2 rows.
    # a_j_offset distinguishes A1 (0) and A2 (2).
    
    # Block 2: A2 (a_i=0, a_j_offset=2), B2 (b_j_offset=2)
    # B2 cols 0,1 (Strassen)
    add_strassen_block(0, 2, 2, 0)
    # B2 cols 2,3 (Strassen)
    add_strassen_block(0, 2, 2, 2)
    # B2 cols 4,5 (Strassen)
    add_strassen_block(0, 2, 2, 4)
    # B2 col 6 (Schoolbook)
    add_schoolbook_2x2x1_block(0, 2, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
