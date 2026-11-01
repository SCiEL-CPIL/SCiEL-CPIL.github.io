"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 7 * (p // 2) = 7 * 6 = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ]
    
    # Precompute V and W contributions for each Strassen form t
    # V_t is a list of (row, val) pairs for the 2x2 block
    # W_t is a list of (row, val) pairs for the 2x2 block
    # Block u maps to B rows 2u, 2u+1 and W rows 2u, 2u+1
    
    V_defs = [
        [(0, 1), (1, 1)],       # t=0: b00 + b11
        [(0, 1)],               # t=1: b00
        [(0, 1), (1, -1)],      # t=2: b01 - b11 (Note: b01 is row 0 col 1, b11 is row 1 col 1)
        [(1, 1), (0, -1)],      # t=3: b10 - b00 (Note: b10 is row 1 col 0, b00 is row 0 col 0)
        [(1, 1)],               # t=4: b11
        [(0, 1), (0, 1)],       # t=5: b00 + b01
        [(1, 1), (1, 1)]        # t=6: b10 + b11
    ]
    
    # Correcting V_defs based on Strassen formulas for B
    # Standard Strassen:
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)(b00)
    # M2 = (a00)(a01-a11) -> Wait, standard is M2 = a00(b01-b11)?
    # Let's re-verify standard Strassen V forms.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)(b00)
    # M2 = (a00)(b01-b11)
    # M3 = (a11)(b10-b00)
    # M4 = (a00+a01)(b11)
    # M5 = (a10-a00)(b10+b11)
    # M6 = (a01-a11)(b00+b10)
    
    # My previous code used:
    # t=0: b00+b11
    # t=1: b00
    # t=2: b01-b11
    # t=3: b10-b00
    # t=4: b11
    # t=5: b00+b01  <-- This differs from standard M5 (b10+b11)
    # t=6: b10+b11  <-- This differs from standard M6 (b00+b10)
    
    # Let's check the W logic in the current code to see which Strassen variant it uses.
    # W for t=5 (M5): W[(2u+1)*n + 1] += 1.
    # C_11 gets M5.
    # Standard C_11 = M0 - M1 + M2 + M5.
    # If M5 corresponds to (a10-a00)(b10+b11), then V5 should be b10+b11.
    # Current code V5 is b00+b01.
    # If V5 is b00+b01, then M5 = (a10-a00)(b00+b01).
    # Then C_11 = ... + (a10-a00)(b00+b01).
    # This is a valid Strassen-like decomposition if W is adjusted.
    
    # I will preserve the EXACT logic of the current code to avoid breaking the valid rank-42 decomposition.
    # The current code is verified to work (score 1.14).
    
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            # Fill V
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
                
            # Fill W
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6:
                W[(2*u) * n + 0][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
