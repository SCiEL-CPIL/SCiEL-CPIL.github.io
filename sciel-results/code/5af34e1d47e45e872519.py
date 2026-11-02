"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank 7 for 2x2x2, 5 blocks for p=10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(5):
        c0 = 7 * b
        k0 = 2 * b
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # U: A00 (0), A11 (3)
        U[0][c0] += 1
        U[3][c0] += 1
        # V: B00 (2b), B11 (10+2b+1)
        V[k0][c0] += 1
        V[10 + k0 + 1][c0] += 1
        # W: C00 (4b), C01 (4b+1), C10 (4b+2), C11 (4b+3)
        W[4*b][c0] += 1
        W[4*b+1][c0] += 1
        W[4*b+2][c0] += 1
        W[4*b+3][c0] += 1
        
        # Term 2: M2 = (A10 + A11)B00
        # U: A10 (2), A11 (3)
        U[2][c0+1] += 1
        U[3][c0+1] += 1
        # V: B00 (2b)
        V[k0][c0+1] += 1
        # W: C10 (4b+2) +1, C11 (4b+3) -1
        W[4*b+2][c0+1] += 1
        W[4*b+3][c0+1] -= 1
        
        # Term 3: M3 = A00(B01 - B11)
        # U: A00 (0)
        U[0][c0+2] += 1
        # V: B01 (10+2b+1) +1, B11 (10+2b) -1 
        # Note: B indices in Strassen block are local.
        # Local B00 is global (0, 2b). Local B01 is global (0, 2b+1).
        # Local B10 is global (1, 2b). Local B11 is global (1, 2b+1).
        # My previous V mapping for M3: V[10+2b+1] (Local B11? No, Local B01 is row 0, col 1. Global row 0, col 2b+1. Index 2b+1. Local B11 is row 1, col 1. Global row 1, col 2b+1. Index 10+2b+1.)
        # Formula M3 = A00 (B01 - B11).
        # B01 is global (0, 2b+1) -> index 2b+1.
        # B11 is global (1, 2b+1) -> index 10+2b+1.
        # So V[2b+1] += 1, V[10+2b+1] -= 1.
        V[k0+1][c0+2] += 1
        V[10 + k0 + 1][c0+2] -= 1
        # W: C01 (4b+1) +1, C11 (4b+3) +1
        W[4*b+1][c0+2] += 1
        W[4*b+3][c0+2] += 1
        
        # Term 4: M4 = A11(B10 - B00)
        # U: A11 (3)
        U[3][c0+3] += 1
        # V: B10 (10+2b) +1, B00 (2b) -1
        V[10 + k0][c0+3] += 1
        V[k0][c0+3] -= 1
        # W: C00 (4b) +1, C10 (4b+2) +1
        W[4*b][c0+3] += 1
        W[4*b+2][c0+3] += 1
        
        # Term 5: M5 = (A00 + A01)B11
        # U: A00 (0), A01 (1)
        U[0][c0+4] += 1
        U[1][c0+4] += 1
        # V: B11 (10+2b+1) +1
        V[10 + k0 + 1][c0+4] += 1
        # W: C00 (4b) -1, C01 (4b+1) +1, C10 (4b+2) -1
        W[4*b][c0+4] -= 1
        W[4*b+1][c0+4] += 1
        W[4*b+2][c0+4] -= 1
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        # U: A10 (2) +1, A00 (0) -1
        U[2][c0+5] += 1
        U[0][c0+5] -= 1
        # V: B00 (2b) +1, B01 (2b+1) +1
        V[k0][c0+5] += 1
        V[k0+1][c0+5] += 1
        # W: C11 (4b+3) +1
        W[4*b+3][c0+5] += 1
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        # U: A01 (1) +1, A11 (3) -1
        U[1][c0+6] += 1
        U[3][c0+6] -= 1
        # V: B10 (10+2b) +1, B11 (10+2b+1) +1
        V[10 + k0][c0+6] += 1
        V[10 + k0 + 1][c0+6] += 1
        # W: C00 (4b) +1
        W[4*b][c0+6] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
