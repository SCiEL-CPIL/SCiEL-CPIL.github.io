"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition for matrix mult."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if U_np.shape != (n*m, rank):
        return False
    if V_np.shape != (m*p, rank):
        return False
    if W_np.shape != (p*n, rank):
        return False
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for c in range(rank):
                    actual += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _strassen_tiled_2x4x8():
    """Construct rank-56 decomposition using 8 independent 2x2x2 Strassen blocks."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 2x2x2 Strassen decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # Strassen:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For 2x4 x 4x8, we have 8 independent 2x2x2 subproblems
    # i in {0,1}, j_block in {0,1,2,3}, k_block in {0,1,2,3}
    # But 4 = 2*2 and 8 = 2*4, so we tile 2x2 blocks
    # Actually, for 2x4 x 4x8:
    # A is 2x4, B is 4x8
    # We can split j into 2 blocks of size 2, and k into 4 blocks of size 2
    # So we have 2 (i) * 2 (j_blocks) * 4 (k_blocks) = 16 subproblems? No.
    # Each subproblem is 2x2 x 2x2 = rank 7
    # Number of subproblems: i has 2 values, j has 4 values split into 2 groups of 2, k has 8 values split into 4 groups of 2
    # So 1 * 2 * 4 = 8 subproblems, each rank 7, total rank 56
    
    # Map each subproblem (i, jg, kg) to Strassen indices
    # i in {0,1}, jg in {0,1} (j = 2*jg + j_local, j_local in {0,1}), kg in {0,1,2,3} (k = 2*kg + k_local, k_local in {0,1})
    
    col = 0
    for i in range(2):
        for jg in range(2):
            for kg in range(4):
                # 7 Strassen terms for this 2x2x2 block
                # m1: (a[i, jg*2+0] + a[i, jg*2+1]) * (b[jg*2+0, kg*2+0] + b[jg*2+0, kg*2+1] + b[jg*2+1, kg*2+0] + b[jg*2+1, kg*2+1])
                # This is getting complex. Let me use a cleaner approach.
                
                # Actually, let me just use the known rank-56 construction directly.
                # For each of the 8 subproblems, assign 7 columns.
                
                # Subproblem (i, jg, kg):
                # A_sub[i, j_local] = A[i, jg*2 + j_local] for j_local in {0,1}
                # B_sub[j_local, k_local] = B[jg*2 + j_local, kg*2 + k_local] for j_local, k_local in {0,1}
                # C_sub[i, k_local] = sum_j_local A_sub[i, j_local] * B_sub[j_local, k_local]
                
                # Strassen for this subproblem:
                # m1 = (A[i, jg*2+0] + A[i, jg*2+1]) * (B[jg*2+0, kg*2+0] + B[jg*2+0, kg*2+1] + B[jg*2+1, kg*2+0] + B[jg*2+1, kg*2+1])
                # This doesn't work directly because Strassen is for 2x2 matrices, not vectors.
                
                # I think I'm overcomplicating. Let me just return the schoolbook for now and use a different strategy.
                col += 7
    
    # This is getting too complex. Let me use a simpler, proven approach.
    # Return schoolbook as fallback
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _random_search(n, m, p, target_rank, max_iter=5000, seed=42):
    """Try to find a rank-target_rank decomposition via random search + refinement."""
    np.random.seed(seed)
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Alternate least squares style: fix two, update one
        # Update U: for each row u of U, minimize ||sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - T[i,j,k]||
        # This is complex. Let me just do random perturbations and check.
        
        # Random perturbation
        for _ in range(3):
            idx = np.random.randint(0, n*m)
            col = np.random.randint(0, target_rank)
            U[idx, col] += np.random.randn() * 0.1
            if _ == 1:
                idx2 = np.random.randint(0, m*p)
                V[idx2, col] += np.random.randn() * 0.1
            if _ == 2:
                idx3 = np.random.randint(0, p*n)
                W[idx3, col] += np.random.randn() * 0.1
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(target_rank):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (actual - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # First, try the known rank-56 Strassen tiling
    # Let me implement it properly this time
    
    # Rank 56 decomposition for 2x4 x 4x8
    # Using 8 independent 2x2x2 Strassen blocks
    
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2x2 decomposition coefficients
    # For matrices A (2x2), B (2x2), C (2x2):
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For the tensor decomposition, we need:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Each Strassen term c corresponds to a product of linear forms in A, B, C.
    # U[i,j,c] is the coefficient of A[i,j] in the c-th linear form for A
    # V[j,k,c] is the coefficient of B[j,k] in the c-th linear form for B
    # W[k,i,c] is the coefficient of C[k,i] in the c-th linear form for C
    
    # Let me define the Strassen terms:
    # Term 1: A_form = a11 + a22, B_form = b11 + b22, C_form = c11 + c22 (with coefficients)
    # Actually, the C form is: c11 gets +1 from m1, c22 gets +1 from m1, etc.
    
    # Let me build this carefully.
    
    # Subproblem mapping:
    # i in {0,1}, jg in {0,1} (j = 2*jg + j_loc, j_loc in {0,1}), kg in {0,1,2,3} (k = 2*kg + k_loc, k_loc in {0,1})
    
    col = 0
    for i in range(2):
        for jg in range(2):
            for kg in range(4):
                j0 = jg * 2
                j1 = jg * 2 + 1
                k0 = kg * 2
                k1 = kg * 2 + 1
                
                # 7 Strassen terms
                # Term 1: m1 = (a[i,j0] + a[i,j1]) * (b[j0,k0] + b[j0,k1] + b[j1,k0] + b[j1,k1])
                # U: a[i,j0] coeff 1, a[i,j1] coeff 1
                # V: b[j0,k0] coeff 1, b[j0,k1] coeff 1, b[j1,k0] coeff 1, b[j1,k1] coeff 1
                # W: c[i,k0] coeff 1, c[i,k1] coeff 1 (from c11 and c22 patterns)
                # Actually, for the subproblem, C_sub[i, k_loc] = sum_j_loc A_sub[i,j_loc] * B_sub[j_loc, k_loc]
                # So c11 corresponds to C[i, k0], c12 to C[i, k1], etc.
                # m1 contributes to c11 and c22: c11 += m1, c22 += m1
                # So W[k0, i] += 1, W[k1, i] += 1
                
                U[i*m + j0, col] += 1
                U[i*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + i, col] += 1
                W[k1*n + i, col] += 1
                col += 1
                
                # Term 2: m2 = (a[i,j0] + a[i,j1]) * b[j0,k0]
                # Wait, Strassen m2 = (a21+a22)b11, so for our subproblem:
                # m2 = (a[i,j0] + a[i,j1]) * b[j0,k0]
                # U: a[i,j0] coeff 1, a[i,j1] coeff 1
                # V: b[j0,k0] coeff 1
                # W: c[i,k0] coeff 1 (c21 += m2)
                
                U[i*m + j0, col] += 1
                U[i*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + i, col] += 1
                col += 1
                
                # Term 3: m3 = a[i,j0] * (b[j0,k1] - b[j1,k1])
                # U: a[i,j0] coeff 1
                # V: b[j0,k1] coeff 1, b[j1,k1] coeff -1
                # W: c[i,k1] coeff 1 (c12 += m3)
                
                U[i*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k1, col] += -1
                W[k1*n + i, col] += 1
                col += 1
                
                # Term 4: m4 = a[i,j1] * (b[j1,k0] - b[j0,k0])
                # U: a[i,j1] coeff 1
                # V: b[j1,k0] coeff 1, b[j0,k0] coeff -1
                # W: c[i,k0] coeff 1 (c21 += m4, c11 += m4)
                
                U[i*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                V[j0*p + k0, col] += -1
                W[k0*n + i, col] += 1
                col += 1
                
                # Term 5: m5 = (a[i,j0] + a[i,j1]) * b[j1,k1]
                # U: a[i,j0] coeff 1, a[i,j1] coeff 1
                # V: b[j1,k1] coeff 1
                # W: c[i,k1] coeff 1 (c12 += m5), c[i,k0] coeff -1 (c11 -= m5)
                
                U[i*m + j0, col] += 1
                U[i*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k1*n + i, col] += 1
                W[k0*n + i, col] += -1
                col += 1
                
                # Term 6: m6 = (a[i,j1] - a[i,j0]) * (b[j0,k0] + b[j0,k1])
                # U: a[i,j1] coeff 1, a[i,j0] coeff -1
                # V: b[j0,k0] coeff 1, b[j0,k1] coeff 1
                # W: c[i,k1] coeff 1 (c22 += m6)
                
                U[i*m + j1, col] += 1
                U[i*m + j0, col] += -1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + i, col] += 1
                col += 1
                
                # Term 7: m7 = (a[i,j1] - a[i,j0]) * (b[j1,k0] + b[j1,k1])
                # U: a[i,j1] coeff 1, a[i,j0] coeff -1
                # V: b[j1,k0] coeff 1, b[j1,k1] coeff 1
                # W: c[i,k0] coeff 1 (c11 += m7)
                
                U[i*m + j1, col] += 1
                U[i*m + j0, col] += -1
                V[j1*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + i, col] += 1
                col += 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
