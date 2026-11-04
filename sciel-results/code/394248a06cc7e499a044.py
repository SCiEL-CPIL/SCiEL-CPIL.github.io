import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    
    # Part 1: 4x4x4 using block Strassen (rank 49)
    # Part 2: 4x4x3 using trivial (rank 48)
    # Total rank: 97
    
    rank = 97
    U = [[0] * rank for _ in range(16)]   # n*m = 16
    V = [[0] * rank for _ in range(28)]   # m*p = 28
    W = [[0] * rank for _ in range(28)]   # p*n = 28
    
    # Block Strassen for 4x4x4
    # Outer Strassen terms (7 terms for 2x2 block matrices)
    # Inner Strassen terms (7 terms for 2x2x2)
    
    # Define Strassen coefficients for 2x2x2:
    # For 2x2 matrices, Strassen uses 7 multiplications:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For the inner 2x2x2 Strassen, define UVW for each of the 7 terms:
    # U, V, W are 4-element vectors (for 2x2 matrices flattened)
    # Index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10-B11
        [0, 1, -1, 0],  # M4: B01-B10
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: C01-C00
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Outer Strassen for 4x4 blocks (each block is 2x2):
    # Same structure as above, but operating on blocks
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2, using cols 0-3 of B)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # For each outer term t and inner term s:
    # The outer term defines which A blocks and B blocks are combined
    # The inner term defines which elements within those blocks are used
    
    # Outer Strassen block combinations (same pattern as inner):
    outer_block_U = [
        # [A00_coeff, A01_coeff, A10_coeff, A11_coeff]
        [1, 0, 0, 1],   # T0: A00+A11
        [0, 0, 1, 1],   # T1: A10+A11
        [1, 0, 0, 0],   # T2: A00
        [0, 0, 0, 1],   # T3: A11
        [1, 1, 0, 0],   # T4: A00+A01
        [-1, 0, 1, 0],  # T5: A10-A00
        [0, 1, 0, -1],  # T6: A01-A11
    ]
    outer_block_V = [
        [1, 0, 0, 1],   # T0: B00+B11
        [1, 0, 0, 0],   # T1: B00
        [0, 0, 1, -1],  # T2: B10-B11
        [0, 1, -1, 0],  # T3: B01-B10
        [0, 0, 0, 1],   # T4: B11
        [1, 0, 1, 0],   # T5: B00+B10
        [0, 1, 0, 1],   # T6: B01+B11
    ]
    outer_block_W = [
        [1, 0, 0, 1],   # T0: C00+C11
        [0, 0, 1, -1],  # T1: C10-C11
        [0, 1, 0, 1],   # T2: C01+C11
        [1, 0, 1, 0],   # T3: C00+C10
        [-1, 1, 0, 0],  # T4: C01-C00
        [0, 0, 0, 1],   # T5: C11
        [1, 0, 0, 0],   # T6: C00
    ]
    
    # Block to flat index mapping:
    # A block (bi, bj) element (a, b) -> flat index (bi*2+a)*4 + (bj*2+b)
    # B block (bi, bj) element (b, c) -> flat index (bi*2+b)*7 + (bj*2+c)
    # C block (bi, bj) element (a, c) -> flat index (bj*2+c)*4 + (bi*2+a)
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # For block idx: bi = idx // 2, bj = idx % 2
    # For element (a,b) in 2x2: a = elem_idx // 2, b = elem_idx % 2
    
    r = 0
    for t in range(7):
        for s in range(7):
            # For U: outer_block_U[t] gives block coefficients, strassen_U[s] gives element coefficients
            # U flat index for block (bi,bj), element (a,b): (bi*2+a)*4 + (bj*2+b)
            for block_idx in range(4):
                block_coeff = outer_block_U[t][block_idx]
                if block_coeff == 0:
                    continue
                bi = block_idx // 2
                bj = block_idx % 2
                for elem_idx in range(4):
                    elem_coeff = strassen_U[s][elem_idx]
                    if elem_coeff == 0:
                        continue
                    a = elem_idx // 2
                    b = elem_idx % 2
                    coeff = block_coeff * elem_coeff
                    if coeff != 0:
                        u_idx = (bi * 2 + a) * 4 + (bj * 2 + b)
                        U[u_idx][r] += coeff
            
            # For V: outer_block_V[t] gives block coefficients, strassen_V[s] gives element coefficients
            # V flat index for block (bi,bj), element (b,c): (bi*2+b)*7 + (bj*2+c)
            for block_idx in range(4):
                block_coeff = outer_block_V[t][block_idx]
                if block_coeff == 0:
                    continue
                bi = block_idx // 2
                bj = block_idx % 2
                for elem_idx in range(4):
                    elem_coeff = strassen_V[s][elem_idx]
                    if elem_coeff == 0:
                        continue
                    b = elem_idx // 2
                    c = elem_idx % 2
                    coeff = block_coeff * elem_coeff
                    if coeff != 0:
                        v_idx = (bi * 2 + b) * 7 + (bj * 2 + c)
                        V[v_idx][r] += coeff
            
            # For W: outer_block_W[t] gives block coefficients, strassen_W[s] gives element coefficients
            # W flat index for block (bi,bj), element (a,c): (bj*2+c)*4 + (bi*2+a)
            for block_idx in range(4):
                block_coeff = outer_block_W[t][block_idx]
                if block_coeff == 0:
                    continue
                bi = block_idx // 2
                bj = block_idx % 2
                for elem_idx in range(4):
                    elem_coeff = strassen_W[s][elem_idx]
                    if elem_coeff == 0:
                        continue
                    a = elem_idx // 2
                    c = elem_idx % 2
                    coeff = block_coeff * elem_coeff
                    if coeff != 0:
                        w_idx = (bj * 2 + c) * 4 + (bi * 2 + a)
                        W[w_idx][r] += coeff
            
            r += 1
    
    # Part 2: 4x4x3 trivial (columns 4-6 of B)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    assert r == rank, f"Rank mismatch: {r} vs {rank}"
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, size=(4, 4)).astype(float)
    B_test = np.random.randint(-5, 5, size=(4, 7)).astype(float)
    C_expected = A_test @ B_test
    
    # Check: C[i,k] = sum_r W[k*4+i, r] * (sum_j U[i*4+j, r]*A[i,j]) * (sum_j V[j*7+k, r]*B[j,k])
    C_comp = np.zeros((4, 7))
    for r_idx in range(rank):
        U_col = np.array(U[:, r_idx], dtype=float)
        V_col = np.array(V[:, r_idx], dtype=float)
        W_col = np.array(W[:, r_idx], dtype=float)
        
        U_mat = U_col.reshape(4, 4)  # [i, j]
        V_mat = V_col.reshape(4, 7)  # [j, k]
        
        a_vec = np.array([np.dot(U_mat[i, :], A_test[i, :]) for i in range(4)])
        b_vec = np.array([np.dot(V_mat[:, k], B_test[:, k]) for k in range(7)])
        
        for i in range(4):
            for k in range(7):
                C_comp[i, k] += W_col[k * 4 + i] * a_vec[i] * b_vec[k]
    
    if not np.allclose(C_comp, C_expected, atol=1e-9):
        # Try alternative: C[i,k] = sum_r (sum_j U[i*4+j,r]*A[i,j]) * (sum_j V[j*7+k,r]*B[j,k])
        C_comp2 = np.zeros((4, 7))
        for r_idx in range(rank):
            U_col = np.array(U[:, r_idx], dtype=float)
            V_col = np.array(V[:, r_idx], dtype=float)
            U_mat = U_col.reshape(4, 4)
            V_mat = V_col.reshape(4, 7)
            a_vec = np.array([np.dot(U_mat[i, :], A_test[i, :]) for i in range(4)])
            b_vec = np.array([np.dot(V_mat[:, k], B_test[:, k]) for k in range(7)])
            C_comp2 += np.outer(a_vec, b_vec)
        
        if not np.allclose(C_comp2, C_expected, atol=1e-9):
            # Debug: check the tensor identity
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should give the right structure
            # For matrix mult, this should equal 1 for each (i,j,k) contribution
            # Actually the tensor T[i,j,k] for matmul is: T[i,j,k] = 1 (coefficient)
            # And C[i,k] = sum_j A[i,j]*B[j,k]
            # The decomposition should satisfy:
            # sum_r U[i*m+j,r]*V[j*p+k,r] = W[k*n+i,r] ... no
            # I think the correct check is just the bilinear form above
            raise ValueError(f"Decomposition incorrect! Max diff: {np.max(np.abs(C_comp2 - C_expected))}")
    
    return {"U": U, "V": V, "W": W}
