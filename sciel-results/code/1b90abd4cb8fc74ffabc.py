import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_dict, V_dict, W_dict):
        U_cols.append(U_dict)
        V_cols.append(V_dict)
        W_cols.append(W_dict)

    # Helper to add standard basis term
    def add_term(i, j, k, u_val, v_val, w_val):
        U = { (i, j): u_val }
        V = { (j, k): v_val }
        W = { (k, i): w_val }
        add_block(U, V, W)

    # Helper to add Strassen 2x2x2 block
    def add_strassen(I0, J0, K0):
        # M1
        U = { (I0, J0): 1, (I0+1, J0+1): 1 }
        V = { (J0, K0): 1, (J0+1, K0+1): 1 }
        W = { (K0, I0): 1, (K0+1, I0+1): 1 }
        add_block(U, V, W)
        
        # M2
        U = { (I0+1, J0): 1, (I0+1, J0+1): 1 }
        V = { (J0, K0): 1 }
        W = { (K0, I0+1): 1, (K0+1, I0+1): -1 }
        add_block(U, V, W)
        
        # M3
        U = { (I0, J0): 1 }
        V = { (J0, K0+1): 1, (J0+1, K0+1): -1 }
        W = { (K0+1, I0): 1, (K0+1, I0+1): 1 }
        add_block(U, V, W)
        
        # M4
        U = { (I0+1, J0+1): 1 }
        V = { (J0+1, K0): 1, (J0, K0): -1 }
        W = { (K0, I0): 1, (K0, I0+1): 1 }
        add_block(U, V, W)
        
        # M5
        U = { (I0, J0): 1, (I0, J0+1): 1 }
        V = { (J0+1, K0+1): 1 }
        W = { (K0, I0): -1, (K0+1, I0): 1 }
        add_block(U, V, W)
        
        # M6
        U = { (I0+1, J0): 1, (I0, J0): -1 }
        V = { (J0, K0): 1, (J0, K0+1): 1 }
        W = { (K0+1, I0+1): 1 }
        add_block(U, V, W)
        
        # M7
        U = { (I0, J0+1): 1, (I0+1, J0+1): -1 }
        V = { (J0+1, K0): 1, (J0+1, K0+1): 1 }
        W = { (K0, I0): 1 }
        add_block(U, V, W)

    # Helper to add 2x2x1 block (Rank 3)
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # M1 = A00 B00
    # M2 = A01 B10
    # M3 = A10 B00
    # C0 = M1 + M2
    # C1 = M3 + (A11 B10)? No.
    # Actually, rank 3 decomposition for 2x2x1:
    # M1 = A00 B00
    # M2 = A01 B10
    # M3 = (A10 + A11) B00 ? No.
    # Let's just use schoolbook for 2x2x1 if we can't find rank 3 easily in code without verification.
    # But we know it exists.
    # Actually, let's use rank 4 (schoolbook) for 2x2x1 to be safe and simple,
    # then the total rank is 83 + (4-3)*3 = 86?
    # 3 blocks of 2x2x1.
    # If rank 4, total rank increases by 3.
    # 83 + 3 = 86.
    # Score 105/86 = 1.22.
    # Still good.
    # Let's try to implement rank 3.
    # Decomposition of M_{2,2,1} rank 3:
    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    # M1 = A00 B00
    # M2 = A01 B10
    # M3 = A10 B00
    # C0 = M1 + M2
    # C1 = M3 + ... missing A11 B10.
    # It is not trivial.
    # I will use schoolbook for 2x2x1 (Rank 4) to ensure correctness.
    
    def add_2x2x1(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0, J0+1, K0, 1, 1, 1)
        add_term(I0+1, J0, K0, 1, 1, 1)
        add_term(I0+1, J0+1, K0, 1, 1, 1)

    def add_2x1x2(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0, J0, K0+1, 1, 1, 1)
        add_term(I0+1, J0, K0, 1, 1, 1)
        add_term(I0+1, J0, K0+1, 1, 1, 1)

    def add_2x1x1(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0+1, J0, K0, 1, 1, 1)

    def add_1x2x2(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0, J0+1, K0+1, 1, 1, 1)
        # Wait, 1x2x2: C0 = A0 B00 + A1 B10, C1 = A0 B01 + A1 B11
        # Rank 2?
        # M1 = A0 (B00+B11)? No.
        # M1 = A0 B00, M2 = A1 B10? No.
        # Standard rank 2:
        # M1 = A0 B00, M2 = A1 B10?
        # C0 = M1 + A1 B10?
        # Actually, rank of 1x2x2 is 2.
        # M1 = A0 (B00 + B01)? No.
        # Let's just use schoolbook (Rank 4) for 1x2x2?
        # No, rank is 2.
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # M1 = A0 B00, M2 = A1 B10, M3 = A0 B01, M4 = A1 B11.
        # Can we do 2?
        # M1 = A0 (B00 + B01)? No, B is fixed.
        # It is known rank 2.
        # M1 = A0 B00 + A1 B10? No, that's one term.
        # I will use schoolbook for small blocks to be safe, 
        # but I know 1x2x2 is rank 2.
        # Let's implement rank 2 for 1x2x2.
        # M1 = A0 B00
        # M2 = A1 B10
        # This doesn't cover all.
        # I will use schoolbook (4) for 1x2x2 as well to be safe.
        # Total rank will be higher.
        # 1x2x2 blocks: 3 in A11B1, 3 in A12B2.
        # If rank 4 instead of 2, cost +2 per block.
        # 6 blocks. +12.
        # Total rank 83 + 12 = 95.
        # Score 105/95 = 1.10.
        # Still better than 1.06.
        
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0, J0, K0+1, 1, 1, 1)
        add_term(I0, J0+1, K0, 1, 1, 1)
        add_term(I0, J0+1, K0+1, 1, 1, 1)

    def add_1x2x1(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0, J0+1, K0, 1, 1, 1)

    def add_1x1x2(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)
        add_term(I0, J0, K0+1, 1, 1, 1)

    def add_1x1x1(I0, J0, K0):
        add_term(I0, J0, K0, 1, 1, 1)

    # Decomposition Plan:
    # A: 3x5 -> Rows 0, 1-2
    # B: 5x7 -> Cols 0-1, 2-3, 4
    
    # Block A0 (1x5) x B (5x7)
    # A0 B0 (1x2 x 2x7)
    # Split B0 (2x7) into cols 0-1, 2-3, 4
    # Sub-blocks:
    # (0, 0, 0): 1x2x2
    # (0, 0, 2): 1x2x2
    # (0, 0, 4): 1x2x1
    
    add_1x2x2(0, 0, 0)
    add_1x2x2(0, 0, 2)
    add_1x2x1(0, 0, 4)
    
    # (0, 2, 0): 1x2x2
    # (0, 2, 2): 1x2x2
    # (0, 2, 4): 1x2x1
    add_1x2x2(0, 2, 0)
    add_1x2x2(0, 2, 2)
    add_1x2x1(0, 2, 4)
    
    # (0, 4, 0): 1x1x2
    # (0, 4, 2): 1x1x2
    # (0, 4, 4): 1x1x1
    add_1x1x2(0, 4, 0)
    add_1x1x2(0, 4, 2)
    add_1x1x1(0, 4, 4)
    
    # Block A1-2 (2x5) x B (5x7)
    # A1-2 B0 (2x2 x 2x7)
    # Split B0 into 2x2, 2x2, 2x1
    # (1, 0, 0): 2x2x2
    # (1, 0, 2): 2x2x2
    # (1, 0, 4): 2x2x1
    add_strassen(1, 0, 0)
    add_strassen(1, 0, 2)
    add_2x2x1(1, 0, 4)
    
    # (1, 2, 0): 2x2x2
    # (1, 2, 2): 2x2x2
    # (1, 2, 4): 2x2x1
    add_strassen(1, 2, 0)
    add_strassen(1, 2, 2)
    add_2x2x1(1, 2, 4)
    
    # (1, 4, 0): 2x1x2
    # (1, 4, 2): 2x1x2
    # (1, 4, 4): 2x1x1
    add_2x1x2(1, 4, 0)
    add_2x1x2(1, 4, 2)
    add_2x1x1(1, 4, 4)
    
    if time.time() > deadline:
        # Fallback to schoolbook if timeout? No, just return what we have.
        # But we must return valid matrices.
        pass

    # Convert sparse to dense
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Decomposition using mixed Strassen and schoolbook blocks reduces rank from 99 to ~95.", q: 0.8, delta: [0.05, 0.1]}
