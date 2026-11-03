"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition of rank 64.
    2. Use the block structure (partition into 2x2 * 2x2 sub-problems) to apply 
       Strassen's algorithm where beneficial, reducing rank.
    3. The 2x4 * 4x8 multiplication can be viewed as 2 (i-groups) x 2 (j-groups) 
       independent 2x2 * 2x2 multiplications (since p=8, we have 4 k-groups of size 2,
       but the j dimension is 4, so 2 j-groups of size 2).
       
       Actually, let's think more carefully:
       - A is 2x4 (n=2, m=4)
       - B is 4x8 (m=4, p=8)
       - C = A*B is 2x8 (n=2, p=8)
       
       We can partition:
       - i in {0, 1} (n=2)
       - j in {0, 1, 2, 3} (m=4) -> 2 groups of 2: j in {0,1} and j in {2,3}
       - k in {0, ..., 7} (p=8) -> 4 groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
       
       For each (j-group, k-group) pair, we have a 2x2 * 2x2 multiplication.
       There are 2 j-groups * 4 k-groups = 8 such sub-problems.
       
       Each 2x2 * 2x2 can be done with rank 7 (Strassen) instead of rank 4.
       So total rank = 8 * 7 = 56.
       
       But wait - we need to be careful about how the indices map.
       
       Let's define the block structure more precisely:
       - A[i, j] where i in [0,1], j in [0,3]
       - B[j, k] where j in [0,3], k in [0,7]
       - C[i, k] = sum_j A[i,j] * B[j,k]
       
       Partition j into two groups: J0 = {0,1}, J1 = {2,3}
       Partition k into four groups: K0={0,1}, K1={2,3}, K2={4,5}, K3={6,7}
       
       For a fixed j-group g (g in {0,1}) and k-group h (h in {0,1,2,3}):
       - A_block is 2x2: A[i, 2g + j'] for i in [0,1], j' in [0,1]
       - B_block is 2x2: B[2g + j', 4h + k'] for j' in [0,1], k' in [0,1]
       - C_block is 2x2: C[i, 4h + k'] = sum_{j'=0}^{1} A[i, 2g+j'] * B[2g+j', 4h+k']
       
       So for each (g, h) pair, we have a 2x2 * 2x2 = 2x2 multiplication.
       There are 2 * 4 = 8 such independent sub-problems.
       
       Using Strassen for each 2x2*2x2: rank 7 each, total rank = 56.
       
       Now I need to construct the U, V, W matrices.
       
       Strassen's algorithm for 2x2 matrices:
       Given X (2x2) and Y (2x2), compute Z = X*Y.
       X = [[x11, x12], [x21, x22]]
       Y = [[y11, y12], [y21, y22]]
       
       M1 = (x11+x22)(y11+y22)
       M2 = (x21+x22)y11
       M3 = x11(y12-y22)
       M4 = x22(y21-y11)
       M5 = (x11+x12)y22
       M6 = (x21-x11)(y11+y12)
       M7 = (x12-x22)(y21+y22)
       
       Z11 = M1 + M4 - M5 + M7
       Z12 = M3 + M5
       Z21 = M2 + M4
       Z22 = M1 - M2 + M3 + M6
       
       For each sub-problem (g, h), the local indices are:
       - A_local[i, j'] = A[i, 2g + j']
       - B_local[j', k'] = B[2g + j', 4h + k']
       - C_local[i, k'] = sum_j' A_local[i, j'] * B_local[j', k']
       
       In the global tensor decomposition:
       - U has shape (n*m, rank) = (8, rank)
       - V has shape (m*p, rank) = (32, rank)
       - W has shape (p*n, rank) = (16, rank)
       
       The trace convention is W[k*n + i, r].
       
       For each sub-problem (g, h) and each Strassen term t (t in 0..6):
       - U contributes to rows corresponding to (i, j) = (i, 2g + j')
       - V contributes to rows corresponding to (j, k) = (2g + j', 4h + k')
       - W contributes to row (k, i) = (4h + k', i)
       
       Let me work out the Strassen coefficients carefully.
       
       For sub-problem (g, h):
       Local A indices: (i, j') where i in {0,1}, j' in {0,1}
       Global A index: i * m + j = i * 4 + (2g + j')
       
       Local B indices: (j', k') where j' in {0,1}, k' in {0,1}
       Global B index: j * p + k = (2g + j') * 8 + (4h + k')
       
       Local C indices: (i, k') where i in {0,1}, k' in {0,1}
       Global C index: k * n + i = (4h + k') * 2 + i
       
       Strassen M1 = (A[0,0]+A[1,1])*(B[0,0]+B[0,1]+B[1,0]+B[1,1])
       In global terms for sub-problem (g,h):
       A[0,0] = A[0, 2g+0], A[1,1] = A[1, 2g+1]
       B[0,0] = B[2g+0, 4h+0], B[0,1] = B[2g+0, 4h+1], B[1,0] = B[2g+1, 4h+0], B[1,1] = B[2g+1, 4h+1]
       
       So for M1:
       U[0*4 + 2g+0, col] += 1
       U[1*4 + 2g+1, col] += 1
       V[(2g+0)*8 + 4h+0, col] += 1
       V[(2g+0)*8 + 4h+1, col] += 1
       V[(2g+1)*8 + 4h+0, col] += 1
       V[(2g+1)*8 + 4h+1, col] += 1
       
       And W for M1 contributes to:
       Z11 += M1, Z22 += M1
       Z11: C[0, 4h+0] -> W[(4h+0)*2 + 0, col] += 1
       Z22: C[1, 4h+1] -> W[(4h+1)*2 + 1, col] += 1
       
       Let me code this up systematically.
    """
    start_time = time.time()
    DEADLINE = 120  # seconds
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Number of sub-problems: 2 j-groups x 4 k-groups = 8
    # Each uses Strassen rank 7
    # Total rank = 56
    num_j_groups = m // 2  # 2
    num_k_groups = p // 2  # 4
    num_subproblems = num_j_groups * num_k_groups  # 8
    strassen_rank = 7
    total_rank = num_subproblems * strassen_rank  # 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # For each M_t, define:
    # U_coeffs: dict from (i, j') to coefficient
    # V_coeffs: dict from (j', k') to coefficient
    # W_coeffs: dict from (i, k') to coefficient (for C[i, k'])
    
    strassen_terms = []
    
    # M1 = (x11+x22)(y11+y22)
    # U: (0,0):1, (1,1):1
    # V: (0,0):1, (0,1):1, (1,0):1, (1,1):1
    # W: (0,0):1, (1,1):1  [Z11 += M1, Z22 += M1]
    strassen_terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (x21+x22)y11
    # U: (1,0):1, (1,1):1
    # V: (0,0):1
    # W: (1,0):1, (1,1):-1  [Z21 += M2, Z22 -= M2]
    strassen_terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    
    # M3 = x11(y12-y22)
    # U: (0,0):1
    # V: (0,1):1, (1,1):-1
    # W: (0,1):1, (1,1):1  [Z12 += M3, Z22 += M3]
    strassen_terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = x22(y21-y11)
    # U: (1,1):1
    # V: (1,0):1, (0,0):-1
    # W: (0,0):1, (1,0):1  [Z11 += M4, Z21 += M4]
    strassen_terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    
    # M5 = (x11+x12)y22
    # U: (0,0):1, (0,1):1
    # V: (1,1):1
    # W: (0,0):-1, (0,1):1  [Z11 -= M5, Z12 += M5]
    strassen_terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (x21-x11)(y11+y12)
    # U: (1,0):1, (0,0):-1
    # V: (0,0):1, (0,1):1
    # W: (1,1):1  [Z22 += M6]
    strassen_terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (x12-x22)(y21+y22)
    # U: (0,1):1, (1,1):-1
    # V: (1,0):1, (1,1):1
    # W: (0,0):1  [Z11 += M7]
    strassen_terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    
    col = 0
    for g in range(num_j_groups):
        for h in range(num_k_groups):
            for t in range(strassen_rank):
                term = strassen_terms[t]
                
                # U contributions
                for (i, jp), coeff in term['U'].items():
                    global_row = i * m + (2 * g + jp)
                    U[global_row][col] += coeff
                
                # V contributions
                for (jp, kp), coeff in term['V'].items():
                    global_row = (2 * g + jp) * p + (4 * h + kp)
                    V[global_row][col] += coeff
                
                # W contributions
                for (i, kp), coeff in term['W'].items():
                    global_row = (4 * h + kp) * n + i
                    W[global_row][col] += coeff
                
                col += 1
    
    # Verify the decomposition
    if time.time() - start_time < DEADLINE - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Check: for random A, B, compute C = A*B and compare with tensor reconstruction
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_test = A_test @ B_test
            
            # Tensor reconstruction: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_recon = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    total = 0
                    for j in range(m):
                        for r in range(total_rank):
                            total += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    C_recon[i, k] = total
            
            max_err = np.max(np.abs(C_recon - C_test))
            if max_err > 1e-10:
                # Fallback to schoolbook
                rank = n * m * p
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                column = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][column] = 1
                            V[j * p + k][column] = 1
                            W[k * n + i][column] = 1
                            column += 1
        except Exception:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
