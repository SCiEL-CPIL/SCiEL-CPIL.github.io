import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 7 Strassen blocks (rank 7 each) + 1 Schoolbook 2x1 block (rank 4)
    # 7 * 7 + 4 = 53
    rank = 53
    
    # U: (n*m, rank) = (4, 53)
    # V: (m*p, rank) = (30, 53)
    # W: (p*n, rank) = (30, 53)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for 2x2 multiplication
    # k_start: starting column index in B (and C)
    def add_strassen(k_start):
        nonlocal col
        # A indices: 0: A00, 1: A01, 2: A10, 3: A11
        # B indices for this block:
        # B00: 0*p + k_start = k_start
        # B01: 0*p + k_start + 1 = k_start + 1
        # B10: 1*p + k_start = p + k_start
        # B11: 1*p + k_start + 1 = p + k_start + 1
        b00 = k_start
        b01 = k_start + 1
        b10 = p + k_start
        b11 = p + k_start + 1
        
        # C indices for this block:
        # C00: k_start * n + 0 = 2*k_start
        # C01: k_start * n + 1 = 2*k_start + 1
        # C10: (k_start+1) * n + 0 = 2*k_start + 2
        # C11: (k_start+1) * n + 1 = 2*k_start + 3
        c00 = 2 * k_start
        c01 = 2 * k_start + 1
        c10 = 2 * k_start + 2
        c11 = 2 * k_start + 3
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0][col] += 1
        U[3][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[2][col] += 1
        U[3][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[0][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[3][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[0][col] += 1
        U[1][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2][col] += 1
        U[0][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1][col] += 1
        U[3][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

    # Helper to add Schoolbook 2x1 multiplication
    # k_start: column index in B (and C)
    def add_schoolbook_2x1(k_start):
        nonlocal col
        # B indices:
        # v0: 0*p + k_start = k_start
        # v1: 1*p + k_start = p + k_start
        v0 = k_start
        v1 = p + k_start
        
        # C indices:
        # u0: k_start * n + 0 = 2*k_start
        # u1: k_start * n + 1 = 2*k_start + 1
        u0 = 2 * k_start
        u1 = 2 * k_start + 1
        
        # P1 = A00 * v0 -> C00
        U[0][col] += 1
        V[v0][col] += 1
        W[u0][col] += 1
        col += 1
        
        # P2 = A01 * v1 -> C00
        U[1][col] += 1
        V[v1][col] += 1
        W[u0][col] += 1
        col += 1
        
        # P3 = A10 * v0 -> C10
        U[2][col] += 1
        V[v0][col] += 1
        W[u1][col] += 1
        col += 1
        
        # P4 = A11 * v1 -> C10
        U[3][col] += 1
        V[v1][col] += 1
        W[u1][col] += 1
        col += 1

    # Apply blocks
    # Columns 0, 1: Strassen
    add_strassen(0)
    # Columns 2, 3: Strassen
    add_strassen(2)
    # Columns 4, 5: Strassen
    add_strassen(4)
    # Columns 6, 7: Strassen
    add_strassen(6)
    # Columns 8, 9: Strassen
    add_strassen(8)
    # Columns 10, 11: Strassen
    add_strassen(10)
    # Columns 12, 13: Strassen
    add_strassen(12)
    # Column 14: Schoolbook 2x1
    add_schoolbook_2x1(14)
    
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
