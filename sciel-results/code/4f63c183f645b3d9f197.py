"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-14 tensor decomposition for 2x2x4 matrix multiplication
    using two independent Strassen decompositions for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(matrix, row, col, val):
        matrix[row][col] += val

    # Block 1: Multiplying A (2x2) by B1 (2x2, cols 0,1 of B)
    # Strassen terms M1..M7 mapped to columns 0..6
    
    # M1: (a+d)(e+h)
    # A: a(0,0)=1, d(1,1)=1 -> U[0], U[3]
    # B: e(0,0)=1, h(1,1)=1 -> V[0], V[5]
    # C: c11(0,0)=1, c22(1,1)=1 -> W[0], W[3]
    set_val(U, 0, 0, 1); set_val(U, 3, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 5, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)
    
    # M2: (c+d)e
    # A: c(1,0)=1, d(1,1)=1 -> U[2], U[3]
    # B: e(0,0)=1 -> V[0]
    # C: c21(1,0)=1, c22(1,1)=-1 -> W[1], W[3]
    set_val(U, 2, 1, 1); set_val(U, 3, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1); set_val(W, 3, 1, -1)
    
    # M3: a(f-h)
    # A: a(0,0)=1 -> U[0]
    # B: f(0,1)=1, h(1,1)=-1 -> V[1], V[5]
    # C: c12(0,1)=1, c22(1,1)=1 -> W[2], W[3]
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 5, 2, -1)
    set_val(W, 2, 2, 1); set_val(W, 3, 2, 1)
    
    # M4: d(g-e)
    # A: d(1,1)=1 -> U[3]
    # B: g(1,0)=1, e(0,0)=-1 -> V[4], V[0]
    # C: c11(0,0)=1, c21(1,0)=1 -> W[0], W[1]
    set_val(U, 3, 3, 1)
    set_val(V, 4, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1); set_val(W, 1, 3, 1)
    
    # M5: (a+b)h
    # A: a(0,0)=1, b(0,1)=1 -> U[0], U[1]
    # B: h(1,1)=1 -> V[5]
    # C: c11(0,0)=-1, c12(0,1)=1 -> W[0], W[2]
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 5, 4, 1)
    set_val(W, 0, 4, -1); set_val(W, 2, 4, 1)
    
    # M6: (c-d)(f+h)
    # A: c(1,0)=1, d(1,1)=-1 -> U[2], U[3]
    # B: f(0,1)=1, h(1,1)=1 -> V[1], V[5]
    # C: c22(1,1)=1 -> W[3]
    set_val(U, 2, 5, 1); set_val(U, 3, 5, -1)
    set_val(V, 1, 5, 1); set_val(V, 5, 5, 1)
    set_val(W, 3, 5, 1)
    
    # M7: (b-d)(e+g)
    # A: b(0,1)=1, d(1,1)=-1 -> U[1], U[3]
    # B: e(0,0)=1, g(1,0)=1 -> V[0], V[4]
    # C: c11(0,0)=1 -> W[0]
    set_val(U, 1, 6, 1); set_val(U, 3, 6, -1)
    set_val(V, 0, 6, 1); set_val(V, 4, 6, 1)
    set_val(W, 0, 6, 1)
    
    # Block 2: Multiplying A (2x2) by B2 (2x2, cols 2,3 of B)
    # Strassen terms M1..M7 mapped to columns 7..13
    
    # M1: (a+d)(e+h)
    # A: a(0,0)=1, d(1,1)=1 -> U[0], U[3]
    # B: e(0,2)=1, h(1,3)=1 -> V[2], V[7]
    # C: c13(0,2)=1, c24(1,3)=1 -> W[4], W[7]
    set_val(U, 0, 7, 1); set_val(U, 3, 7, 1)
    set_val(V, 2, 7, 1); set_val(V, 7, 7, 1)
    set_val(W, 4, 7, 1); set_val(W, 7, 7, 1)
    
    # M2: (c+d)e
    # A: c(1,0)=1, d(1,1)=1 -> U[2], U[3]
    # B: e(0,2)=1 -> V[2]
    # C: c23(1,2)=1, c24(1,3)=-1 -> W[5], W[7]
    set_val(U, 2, 8, 1); set_val(U, 3, 8, 1)
    set_val(V, 2, 8, 1)
    set_val(W, 5, 8, 1); set_val(W, 7, 8, -1)
    
    # M3: a(f-h)
    # A: a(0,0)=1 -> U[0]
    # B: f(0,3)=1, h(1,3)=-1 -> V[3], V[7]
    # C: c14(0,3)=1, c24(1,3)=1 -> W[6], W[7]
    set_val(U, 0, 9, 1)
    set_val(V, 3, 9, 1); set_val(V, 7, 9, -1)
    set_val(W, 6, 9, 1); set_val(W, 7, 9, 1)
    
    # M4: d(g-e)
    # A: d(1,1)=1 -> U[3]
    # B: g(1,2)=1, e(0,2)=-1 -> V[6], V[2]
    # C: c13(0,2)=1, c23(1,2)=1 -> W[4], W[5]
    set_val(U, 3, 10, 1)
    set_val(V, 6, 10, 1); set_val(V, 2, 10, -1)
    set_val(W, 4, 10, 1); set_val(W, 5, 10, 1)
    
    # M5: (a+b)h
    # A: a(0,0)=1, b(0,1)=1 -> U[0], U[1]
    # B: h(1,3)=1 -> V[7]
    # C: c13(0,2)=-1, c14(0,3)=1 -> W[4], W[6]
    set_val(U, 0, 11, 1); set_val(U, 1, 11, 1)
    set_val(V, 7, 11, 1)
    set_val(W, 4, 11, -1); set_val(W, 6, 11, 1)
    
    # M6: (c-d)(f+h)
    # A: c(1,0)=1, d(1,1)=-1 -> U[2], U[3]
    # B: f(0,3)=1, h(1,3)=1 -> V[3], V[7]
    # C: c24(1,3)=1 -> W[7]
    set_val(U, 2, 12, 1); set_val(U, 3, 12, -1)
    set_val(V, 3, 12, 1); set_val(V, 7, 12, 1)
    set_val(W, 7, 12, 1)
    
    # M7: (b-d)(e+g)
    # A: b(0,1)=1, d(1,1)=-1 -> U[1], U[3]
    # B: e(0,2)=1, g(1,2)=1 -> V[2], V[6]
    # C: c13(0,2)=1 -> W[4]
    set_val(U, 1, 13, 1); set_val(U, 3, 13, -1)
    set_val(V, 2, 13, 1); set_val(V, 6, 13, 1)
    set_val(W, 4, 13, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
