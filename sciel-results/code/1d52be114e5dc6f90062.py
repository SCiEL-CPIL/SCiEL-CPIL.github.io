"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 5,5,5
    
    # We'll use a hybrid: Strassen for the top-left 2x2 block multiplication,
    # schoolbook for all other terms. This saves 1 multiplication: rank 124.
    
    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Define helper to add a rank-1 term
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal col
        U[u_row][col] = u_val
        V[v_row][col] = v_val
        W[w_row][col] = w_val
        col += 1
    
    # Strassen for top-left 2x2:
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    
    # Then:
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Each M is a product of two linear combinations, which in tensor decomposition
    # means each M contributes ONE rank-1 term where U, V, W encode the linear combos.
    
    # For M1: U vector has 1s at A00 and A11 positions, V vector has 1s at B00 and B11
    # A00: U[0][col]=1, U[1][col]=1 (row 0, cols 0 and 1)
    # A11: U[6][col]=1, U[7][col]=1 (row 1, cols 0 and 1)
    # B00: V[0][col]=1, V[1][col]=1 (row 0, cols 0 and 1)
    # B11: V[6][col]=1, V[7][col]=1 (row 1, cols 0 and 1)
    # W: C00 gets +1, C11 gets +1 -> W[0][col]=1 (C00: k=0,i=0), W[10][col]=1? 
    # Wait, W[k*n+i] where k is col of B / row of C output, i is row of A / col of C output
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # W[k*n + i] corresponds to output C[i,k]
    
    # C00: i=0,k=0 -> W[0*5+0] = W[0]
    # C01: i=0,k=1 -> W[1*5+0] = W[5]
    # C10: i=1,k=0 -> W[0*5+1] = W[1]
    # C11: i=1,k=1 -> W[1*5+1] = W[6]
    
    # M1 contributes to C00 (+1), C11 (+1)
    # But W is a single vector per rank term. This won't work directly because
    # Strassen's linear combinations mean W has multiple non-zero entries.
    
    # Actually, in the tensor decomposition, each rank-1 term is U⊗V⊗W where
    # U, V, W are vectors. The product U⊗V⊗W gives a 3D array where
    # T[i,j,k] = U[i,j]*V[j,k]*W[k,i]... no.
    
    # The matrix multiplication tensor is T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # So T[i,j,k] = delta[i,j] for A * delta[j,k] for B * delta[k,i] for C... 
    # Actually T is defined such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # and T[i,j,k] = 1 for all valid i,j,k (schoolbook).
    
    # A rank-1 decomposition: T = sum_r U[r] ⊗ V[r] ⊗ W[r]
    # where (U[r] ⊗ V[r] ⊗ W[r])[i,j,k] = U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
    
    # For Strassen, each "product" M is actually a bilinear form, not a rank-1 tensor.
    # Strassen doesn't directly give a tensor decomposition in this form.
    
    # I think I'm overcomplicating. Let me just use schoolbook correctly.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for idx in range(rank):
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
