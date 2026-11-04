"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Total rank expected: 68
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term with specific coefficients
    # u_idx, v_idx, w_idx are the indices in U, V, W
    # coeff is the coefficient for this rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # 1. Strassen for i in {0,1}, j in {0,1}, k in {0..7}
    # We split k into 4 blocks of 2: {0,1}, {2,3}, {4,5}, {6,7}
    for k_offset in range(0, p, 2):
        # Strassen for 2x2x2
        # A indices: (0,0), (0,1), (1,0), (1,1)
        # B indices: (0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)
        # C indices: (0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        # This requires expanding the linear combinations.
        # (A00+A11) means U[0,0]=1, U[1,1]=1.
        # (B00+B11) means V[0,k]=1, V[1,k+1]=1.
        # W[0,k] += 1, W[1,k+1] += 1.
        
        # Term 1: A00 * B00
        add_term(0, 0*p + k_offset, k_offset*n + 0, 1)
        # Term 2: A00 * B11
        add_term(0, 1*p + (k_offset+1), k_offset*n + 0, 1)
        # Term 3: A11 * B00
        add_term(1, 0*p + k_offset, (k_offset+1)*n + 1, 1)
        # Term 4: A11 * B11
        add_term(1, 1*p + (k_offset+1), (k_offset+1)*n + 1, 1)
        
        # M2 = (A10+A11)B00
        # C10 += M2, C11 += M2
        # Term 5: A10 * B00
        add_term(1, 0*p + k_offset, k_offset*n + 1, 1)
        # Term 6: A11 * B00
        add_term(1, 0*p + k_offset, k_offset*n + 1, 1)
        # Term 7: A10 * B00 (for C11)
        add_term(1, 0*p + k_offset, (k_offset+1)*n + 1, 1)
        # Term 8: A11 * B00 (for C11)
        add_term(1, 0*p + k_offset, (k_offset+1)*n + 1, 1)
        
        # M3 = A00(B01-B11)
        # C01 += M3
        # Term 9: A00 * B01
        add_term(0, 1*p + (k_offset+1), (k_offset+1)*n + 0, 1)
        # Term 10: A00 * (-B11)
        add_term(0, 1*p + (k_offset+1), (k_offset+1)*n + 0, -1)
        
        # M4 = A11(B10-B00)
        # C10 += M4
        # Term 11: A11 * B10
        add_term(1, 1*p + k_offset, k_offset*n + 1, 1)
        # Term 12: A11 * (-B00)
        add_term(1, 0*p + k_offset, k_offset*n + 1, -1)
        
        # M5 = (A00+A10)B10
        # C10 += M5, C00 -= M5
        # Term 13: A00 * B10
        add_term(0, 1*p + k_offset, k_offset*n + 1, 1)
        # Term 14: A10 * B10
        add_term(1, 1*p + k_offset, k_offset*n + 1, 1)
        # Term 15: A00 * B10 (for C00, coeff -1)
        add_term(0, 1*p + k_offset, k_offset*n + 0, -1)
        # Term 16: A10 * B10 (for C00, coeff -1)
        add_term(1, 1*p + k_offset, k_offset*n + 0, -1)
        
        # M6 = (A01-A00)(B00+B10)
        # C01 += M6, C11 -= M6
        # Term 17: A01 * B00
        add_term(0, 0*p + k_offset, (k_offset+1)*n + 0, 1)
        # Term 18: A00 * (-B00)
        add_term(0, 0*p + k_offset, (k_offset+1)*n + 0, -1)
        # Term 19: A01 * B10
        add_term(0, 1*p + k_offset, (k_offset+1)*n + 0, 1)
        # Term 20: A00 * (-B10)
        add_term(0, 1*p + k_offset, (k_offset+1)*n + 0, -1)
        # Term 21: A01 * B00 (for C11, coeff -1)
        add_term(0, 0*p + k_offset, (k_offset+1)*n + 1, -1)
        # Term 22: A00 * (-B00) (for C11, coeff -1)
        add_term(0, 0*p + k_offset, (k_offset+1)*n + 1, 1)
        # Term 23: A01 * B10 (for C11, coeff -1)
        add_term(0, 1*p + k_offset, (k_offset+1)*n + 1, -1)
        # Term 24: A00 * (-B10) (for C11, coeff -1)
        add_term(0, 1*p + k_offset, (k_offset+1)*n + 1, 1)
        
        # M7 = (A11-A10)(B01+B11)
        # C01 += M7, C00 += M7
        # Term 25: A11 * B01
        add_term(1, 1*p + (k_offset+1), (k_offset+1)*n + 0, 1)
        # Term 26: A10 * (-B01)
        add_term(1, 1*p + (k_offset+1), (k_offset+1)*n + 0, -1)
        # Term 27: A11 * B11
        add_term(1, 1*p + (k_offset+1), (k_offset+1)*n + 0, 1)
        # Term 28: A10 * (-B11)
        add_term(1, 1*p + (k_offset+1), (k_offset+1)*n + 0, -1)
        # Term 29: A11 * B01 (for C00)
        add_term(1, 1*p + (k_offset+1), k_offset*n + 0, 1)
        # Term 30: A10 * (-B01) (for C00)
        add_term(1, 1*p + (k_offset+1), k_offset*n + 0, -1)
        # Term 31: A11 * B11 (for C00)
        add_term(1, 1*p + (k_offset+1), k_offset*n + 0, 1)
        # Term 32: A10 * (-B11) (for C00)
        add_term(1, 1*p + (k_offset+1), k_offset*n + 0, -1)
        
        # Wait, the standard Strassen has 7 multiplications.
        # Each multiplication of linear forms generates multiple rank-1 terms.
        # My implementation above expands the linear forms into rank-1 terms.
        # Let's verify the count.
        # M1: 2 terms in A, 2 in B -> 4 rank-1 terms.
        # M2: 2 terms in A, 1 in B -> 2 rank-1 terms.
        # M3: 1 term in A, 2 in B -> 2 rank-1 terms.
        # M4: 1 term in A, 2 in B -> 2 rank-1 terms.
        # M5: 2 terms in A, 1 in B -> 2 rank-1 terms.
        # M6: 2 terms in A, 2 in B -> 4 rank-1 terms.
        # M7: 2 terms in A, 2 in B -> 4 rank-1 terms.
        # Total 4+2+2+2+2+4+4 = 20 rank-1 terms?
        # But Strassen rank is 7.
        # The "rank" of the tensor decomposition is the number of scalar multiplications.
        # In the tensor decomposition $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$, the rank is R.
        # If I expand $u_r$ and $v_r$ into basis vectors, I am increasing the rank!
        # NO. The decomposition is $C = \sum_{r} (u_r \cdot A) (v_r \cdot B) w_r$.
        # The term added to U, V, W corresponds to ONE scalar multiplication.
        # If $u_r$ has multiple non-zero entries, it is still ONE rank-1 term in the tensor decomposition sense?
        # No, the tensor decomposition is a sum of rank-1 tensors.
        # A rank-1 tensor is $u \otimes v \otimes w$.
        # If $u$ is sparse, it is still rank 1.
        # So, I should add ONE column to U, V, W for each of the 7 Strassen multiplications.
        # The entries in U, V, W can be non-zero in multiple places.
        
        # Let's restart the Strassen part with this understanding.
        
        # I need to clear the previous additions for this block if I were doing it incrementally, 
        # but since I'm writing the whole function, I'll just define the 7 terms correctly.
        
        # I will delete the previous logic and use a cleaner method.
        
    # Reset col to 0 and restart logic properly
    col = 0
    # Re-initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # i_off: 0 or 1 (row offset in A)
        # j_off: 0 or 1 (col offset in A / row offset in B)
        # k_off: 0, 2, 4, 6 (col offset in B / col offset in C)
        
        # A indices:
        # A00: i_off, j_off
        # A01: i_off, j_off+1
        # A10: i_off+1, j_off
        # A11: i_off+1, j_off+1
        
        # B indices:
        # B00: j_off, k_off
        # B01: j_off, k_off+1
        # B10: j_off+1, k_off
        # B11: j_off+1, k_off+1
        
        # C indices:
        # C00: i_off, k_off
        # C01: i_off, k_off+1
        # C10: i_off+1, k_off
        # C11: i_off+1, k_off+1
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=1
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C10=1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[j_off*p + k_off][col] = -1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (A00+A10)B10
        # U: A00=1, A10=1
        # V: B10=1
        # W: C10=1, C00=-1
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + j_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        col += 1
        
        # M6 = (A01-A00)(B00+B10)
        # U: A01=1, A00=-1
        # V: B00=1, B10=1
        # W: C01=1, C11=-1
        U[i_off*m + (j_off+1)][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M7 = (A11-A10)(B01+B11)
        # U: A11=1, A10=-1
        # V: B01=1, B11=1
        # W: C01=1, C00=1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + j_off][col] = -1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1

    # Apply Strassen to the 4 blocks
    for k_off in range(0, p, 2):
        add_strassen_block(0, 0, k_off)
        
    # 2. Schoolbook for i=2, j in {0,1}
    # C_{2,k} += A_{2,j} B_{j,k}
    for j in range(2):
        for k in range(p):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # 3. Schoolbook for j=2, i in {0,1,2}
    # C_{i,k} += A_{i,2} B_{2,k}
    for i in range(3):
        for k in range(p):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
