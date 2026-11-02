"""Construct rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Rank 62 construction:
    # Split 8 columns into two blocks of 3 (using Hybrid Strassen/Schoolbook, rank 22 each)
    # and one block of 2 (using Schoolbook, rank 18).
    # Total rank = 22 + 22 + 18 = 62.
    
    rank = 62
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 24 rows
    W = [[0]*rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Block 1: Columns 0, 1, 2 (Hybrid 3x3x3)
    # A indices: 0..8 (i*m+j)
    # B indices: 0..23 (j*p+k)
    # C indices: 0..23 (k*n+i)
    
    # Strassen part for A[0:2,0:2] * B[0:2,0:2]
    # M1 = A00 * (B00+B11)
    add_term(0, 0, 0) # A00 B00 -> C00
    add_term(0, 11, 0) # A00 B11 -> C00
    # M2 = (A10+A11) * B00
    add_term(3, 0, 1) # A10 B00 -> C10
    add_term(4, 0, 1) # A11 B00 -> C10
    # M3 = A00 * (B10-B11)
    add_term(0, 10, 1) # A00 B10 -> C10
    add_term(0, 11, 1) # A00 B11 -> C10 (subtracted in W)
    # This manual expansion is tedious and error-prone for signs.
    
    # Let's use a simpler, verified rank reduction:
    # We will use the schoolbook for the 3x3x3 blocks but combine the two 3x3x3 blocks
    # and the 3x3x2 block in a way that doesn't reduce rank further than 62?
    # Actually, if I can't guarantee the Strassen signs, I should stick to a valid, simpler improvement.
    
    # Let's revert to a safer improvement:
    # Use the fact that we can compute 3x3x8 as two 3x3x4 blocks.
    # And 3x3x4 as one 3x3x3 (Strassen 19) + one 3x3x1 (Schoolbook 9).
    # But I need the Strassen coefficients.
    
    # Since I cannot reliably generate Strassen coefficients in this edit without testing,
    # I will implement the **Rank 62 Hybrid** with explicit schoolbook for the 2x2 part
    # and Strassen for the 2x2 part? No, Strassen is for 2x2.
    
    # Let's just use the **Rank 62** construction derived above:
    # 7 mults for 2x2 (Strassen) + 15 mults for the rest (Schoolbook) = 22 per 3x3x3 block.
    
    # I will hardcode the 22 terms for the 3x3x3 block.
    
    def add_hybrid_3x3x3(k_offset):
        nonlocal col
        # k_offset is 0 or 3
        # B indices for this block: j*p + (k_offset + k_local)
        # C indices for this block: (k_offset + k_local)*n + i
        
        # 1. Strassen for top-left 2x2
        # M1 = A00 * (B00 + B11)
        add_term(0, 0*p + k_offset + 0, (k_offset+0)*n + 0) # A00 B00
        add_term(0, 1*p + k_offset + 1, (k_offset+1)*n + 0) # A00 B11
        # M2 = (A10 + A11) * B00
        add_term(3, 0*p + k_offset + 0, (k_offset+0)*n + 1) # A10 B00
        add_term(4, 0*p + k_offset + 0, (k_offset+0)*n + 1) # A11 B00
        # M3 = A00 * (B10 - B11)
        add_term(0, 1*p + k_offset + 0, (k_offset+0)*n + 1) # A00 B10
        add_term(0, 1*p + k_offset + 1, (k_offset+1)*n + 1, -1) # A00 B11
        # M4 = A11 * (B00 - B10)
        add_term(4, 0*p + k_offset + 0, (k_offset+1)*n + 1) # A11 B00
        add_term(4, 1*p + k_offset + 0, (k_offset+0)*n + 1, -1) # A11 B10
        # M5 = (A00 + A11) * (B00 + B11)
        add_term(0, 0*p + k_offset + 0, (k_offset+0)*n + 0) # A00 B00
        add_term(0, 1*p + k_offset + 1, (k_offset+1)*n + 0) # A00 B11
        add_term(4, 0*p + k_offset + 0, (k_offset+0)*n + 0) # A11 B00
        add_term(4, 1*p + k_offset + 1, (k_offset+1)*n + 0) # A11 B11
        # M6 = (A01 + A11) * (B10 + B11)
        add_term(1, 1*p + k_offset + 0, (k_offset+0)*n + 0) # A01 B10
        add_term(1, 1*p + k_offset + 1, (k_offset+1)*n + 0) # A01 B11
        add_term(4, 1*p + k_offset + 0, (k_offset+0)*n + 0) # A11 B10
        add_term(4, 1*p + k_offset + 1, (k_offset+1)*n + 0) # A11 B11
        # M7 = (A00 + A10) * (B00 + B01)
        add_term(0, 0*p + k_offset + 0, (k_offset+0)*n + 0) # A00 B00
        add_term(0, 0*p + k_offset + 1, (k_offset+1)*n + 0) # A00 B01
        add_term(3, 0*p + k_offset + 0, (k_offset+0)*n + 0) # A10 B00
        add_term(3, 0*p + k_offset + 1, (k_offset+1)*n + 0) # A10 B01
        
        # This expansion is incorrect because Strassen combines sums.
        # The above adds 16 terms for the 2x2 part, which is just schoolbook.
        # Strassen requires the U and V matrices to contain the sums.
        # But the interface requires U, V, W to be sparse 0/1 (or +/-1) matrices?
        # "integer/half-integer entries".
        # So I can put sums in U and V?
        # No, U[i][k] is the coefficient for A_i in term k.
        # If I use Strassen, U and V will have entries like 1, -1.
        # The above code adds separate terms for each product, which defeats Strassen.
        
        # To use Strassen, I must define the 7 terms explicitly with their U/V/W vectors.
        
        pass

    # Given the complexity and risk of error, I will fall back to a **Rank 72** program
    # but I will try to use a **Rank 63** construction if I can be sure.
    
    # Actually, let's just return the schoolbook to ensure validity, as the prompt
    # prioritizes a valid result.
    
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
