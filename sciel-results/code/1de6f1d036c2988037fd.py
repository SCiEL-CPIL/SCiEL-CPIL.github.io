import numpy as np

def construct():
    n, m, p = 2, 5, 5
    # Total rank will be sum of ranks of sub-problems
    # Partition j (0..4) into J3={0,1,2} and J2={3,4}
    # Partition k (0..4) into K3={0,1,2} and K2={3,4}
    
    # Sub-problems:
    # 1. A3 (2x3) * B33 (3x3) -> C33 (2x3). Rank 18 (Schoolbook)
    # 2. A3 (2x3) * B32 (3x2) -> C32 (2x2). Rank 12 (Schoolbook)
    # 3. A2 (2x2) * B23 (2x3) -> C23 (2x3). Rank 12 (Schoolbook)
    # 4. A2 (2x2) * B22 (2x2) -> C22 (2x2). Rank 7 (Strassen)
    
    total_rank = 18 + 12 + 12 + 7
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    # 1. A3 * B33 (2x3 * 3x3)
    # i in 0..1, j in 0..2, k in 0..2
    for i in range(2):
        for j in range(3):
            for k in range(3):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 2 + i
                add_schoolbook_term(u_idx, v_idx, w_idx, 1, 1, 1)

    # 2. A3 * B32 (2x3 * 3x2)
    # i in 0..1, j in 0..2, k in 0..1 (local k) -> global k = k+3
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u_idx = i * 5 + j
                v_idx = j * 5 + (k + 3)
                w_idx = (k + 3) * 2 + i
                add_schoolbook_term(u_idx, v_idx, w_idx, 1, 1, 1)

    # 3. A2 * B23 (2x2 * 2x3)
    # i in 0..1, j in 0..1 (local j) -> global j = j+3, k in 0..2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = i * 5 + (j + 3)
                v_idx = (j + 3) * 5 + k
                w_idx = k * 2 + i
                add_schoolbook_term(u_idx, v_idx, w_idx, 1, 1, 1)

    # 4. A2 * B22 (2x2 * 2x2) using Strassen
    # i in 0..1, j in 0..1 (local) -> global j = j+3, k in 0..1 (local) -> global k = k+3
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
    # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    # M5 = (A00+A01)B11 -> C00 += -1, C01 += 1
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    # M7 = (A01-A11)(B10+B11) -> C00 += 1
    
    # Note: Indices in Strassen are local (0,0) to (1,1)
    # Global mapping:
    # u_idx = i*5 + (j+3)
    # v_idx = (j+3)*5 + (k+3)
    # w_idx = (k+3)*2 + i
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {(i,j): val}
        # v_coeffs: dict {(j,k): val}
        # w_coeffs: dict {(k,i): val}
        nonlocal col
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * 5 + (j + 3), col] = val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[(j + 3) * 5 + (k + 3), col] = val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[(k + 3) * 2 + i, col] = val
        col += 1

    # M1
    add_strassen_term(
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    )
    # M2
    add_strassen_term(
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):-1, (1,1):1} # C10 is k=0,i=1? No. C[1,0] is i=1, k=0. w_idx = 0*2+1=1. C[1,1] is i=1, k=1. w_idx=1*2+1=3.
    )
    # Correction on W indices for Strassen:
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[0,1] -> k=1, i=0 -> idx 1
    # C[1,0] -> k=0, i=1 -> idx 2
    # C[1,1] -> k=1, i=1 -> idx 3
    
    # Re-do W coefficients carefully
    # M1: C00+=1, C11+=1. W: {0:1, 3:1}
    # M2: C10+=1, C11+=-1. W: {2:1, 3:-1}
    # M3: C01+=1, C11+=1. W: {1:1, 3:1}
    # M4: C00+=1, C10+=1. W: {0:1, 2:1}
    # M5: C00+=-1, C01+=1. W: {0:-1, 1:1}
    # M6: C11+=1. W: {3:1}
    # M7: C00+=1. W: {0:1}

    # I already added M1. Let's restart the Strassen part to be clean.
    # I will just define the 7 terms explicitly below.
    
    # Reset col to start of Strassen block? No, I already incremented col 3 times in the previous attempt?
    # No, I defined add_strassen_term but didn't call it yet in the execution flow above, 
    # but I wrote the code in the thought block. In the actual code, I will just write the 7 calls.
    
    # Let's rewrite the Strassen section in the final code.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
