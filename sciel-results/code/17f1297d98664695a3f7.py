"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute C via tensor decomposition:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # C[i,k] = sum_{j,c} A[i,j] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] / ... 
        # Actually the decomposition is: C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        # Wait, that's not right. Let me think about the tensor structure.
        
        # The matrix multiplication tensor M has: M[i,j,k; a,b,c] where
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_{j,c} A[i,j] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This doesn't work directly. Let me reconsider.
        
        # Actually, the standard decomposition is:
        # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # No, that gives a product of two sums times W, which isn't bilinear in A and B.
        
        # The correct interpretation: The tensor T such that
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # can be written as C[i,k] = sum_c U[i*m+j, c]... 
        # 
        # Actually, the decomposition means:
        # For the bilinear map (A, B) -> C, we have:
        # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_l B[l,k] * V[j*p+l, c]) * W[k*n+i, c]
        # This still doesn't work because j appears in both U and V indices.
        
        # Let me reconsider. The matrix multiplication tensor has 3 input indices (i,j) for A, (j,k) for B,
        # and 2 output indices (i,k) for C. The shared index is j.
        # 
        # A rank-1 term in the tensor decomposition corresponds to:
        # C[i,k] += (sum_j A[i,j] * u_{i,j}) * (sum_j B[j,k] * v_{j,k}) * w_{i,k}
        # But this isn't quite right either because the j-sums are independent.
        
        # The correct form: C[i,k] = sum_c A[i,:] · U_c[i,:] * V_c[:,k] · B[:,k] * W_c[i,k]
        # where U_c is m-vector for each i, V_c is m-vector for each k, W_c is scalar for each (i,k).
        # But U has shape (n*m, rank), so U[i*m+j, c] is the (i,j) entry of the c-th factor's A-part.
        # V has shape (m*p, rank), so V[j*p+k, c] is the (j,k) entry of the c-th factor's B-part.
        # W has shape (p*n, rank), so W[k*n+i, c] is the (k,i) entry of the c-th factor's C-part.
        
        # So: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # This IS bilinear in A and B! The first sum is linear in A, second is linear in B, and W is a constant.
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # First factor: for each i, sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            # Second factor: for each k, sum_j V[j*p+k, c] * B[j,k]
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            # Multiply and scale by W
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _try_rank_reduction(U, V, W, n=2, m=4, p=8, max_attempts=500):
    """Try to reduce rank by removing columns that are linearly dependent or zero."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = len(U[0])
    
    for _ in range(max_attempts):
        # Try removing each column
        removed = False
        for c in range(rank):
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, trials=50, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Initialize schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Try random perturbations and rank reduction
    random.seed(42)
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Create a perturbed version of schoolbook
        U_pert = [row[:] for row in U]
        V_pert = [row[:] for row in V]
        W_pert = [row[:] for row in W]
        
        # Add small random noise to some entries
        for _ in range(5):
            ri = random.randint(0, len(U_pert) - 1)
            ci = random.randint(0, schoolbook_rank - 1)
            U_pert[ri][ci] += random.gauss(0, 0.1)
            V_pert[ri % len(V_pert)][ci] += random.gauss(0, 0.1)
            W_pert[ri % len(W_pert)][ci] += random.gauss(0, 0.1)
        
        # Try to verify and reduce
        if _verify(U_pert, V_pert, W_pert, n, m, p, trials=100, tol=1e-8):
            U_r, V_r, W_r, rank_r = _try_rank_reduction(U_pert, V_pert, W_pert, n, m, p)
            if rank_r < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank_r
                print(f"Improved to rank {best_rank}")
    
    # Final verification
    if best_rank >= schoolbook_rank:
        # Return schoolbook
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}


# EVOLVE-BLOCK-START
def _build_strassen_block_2x2(n=2, m=4, p=8):
    """
    Build a rank-56 tensor decomposition using 2x2 block Strassen.
    
    A (2x4) is split into 2x2 blocks of 1x2:
      A = [[A11, A12], [A21, A22]]
    B (4x8) is split into 2x2 blocks of 2x4:
      B = [[B11, B12], [B21, B22]]
    C (2x8) is split into 2x2 blocks of 1x4:
      C = [[C11, C12], [C21, C22]]
    
    Strassen intermediates (each is a (1x2)*(2x4) = 1x4 product, rank 8):
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Total rank: 7 * 8 = 56
    """
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Block indices for A (2x4):
    # A[i, j] where i in {0,1}, j in {0,1,2,3}
    # A11 = A[0,0:2], A12 = A[0,2:4], A21 = A[1,0:2], A22 = A[1,2:4]
    
    # Block indices for B (4x8):
    # B[j, k] where j in {0,1,2,3}, k in {0,...,7}
    # B11 = B[0:2, 0:4], B12 = B[0:2, 4:8], B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    
    # Strassen A-side combinations (each 1x2):
    # For M1: A11+A22 -> A[0,0]+A[1,2], A[0,1]+A[1,3]
    # For M2: A21+A22 -> A[1,0]+A[1,2], A[1,1]+A[1,3]
    # For M3: A11 -> A[0,0], A[0,1]
    # For M4: A22 -> A[1,2], A[1,3]
    # For M5: A11+A12 -> A[0,0]+A[0,2], A[0,1]+A[0,3]
    # For M6: A21-A11 -> A[1,0]-A[0,0], A[1,1]-A[0,1]
    # For M7: A12-A22 -> A[0,2]-A[1,2], A[0,3]-A[1,3]
    
    # Strassen B-side combinations (each 2x4):
    # For M1: B11+B22
    # For M2: B11
    # For M3: B12-B22
    # For M4: B21-B11
    # For M5: B22
    # For M6: B11+B12
    # For M7: B21+B22
    
    # Define the 7 Strassen terms
    # Each term has:
    #   A-side: coefficients for A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]
    #   B-side: coefficients for B[j,k] for j in 0..3, k in 0..7
    #   C-side: coefficients for C[i,k] for i in 0..1, k in 0..7
    
    # A-side vectors (length 8, indexed as i*m+j = i*4+j)
    A_sides = {
        'M1': [1, 1, 0, 0, 0, 0, 1, 1],  # A11+A22: A[0,0]+A[1,2], A[0,1]+A[1,3]
        'M2': [0, 0, 0, 0, 1, 1, 1, 1],  # A21+A22: A[1,0]+A[1,2], A[1,1]+A[1,3]
        'M3': [1, 1, 0, 0, 0, 0, 0, 0],  # A11
        'M4': [0, 0, 0, 0, 0, 0, 1, 1],  # A22
        'M5': [1, 1, 1, 1, 0, 0, 0, 0],  # A11+A12
        'M6': [1, 1, 0, 0, -1, -1, 0, 0],  # A21-A11
        'M7': [0, 0, 1, 1, 0, 0, -1, -1],  # A12-A22
    }
    
    # B-side: for each term, which B[j,k] entries get which coefficient
    # B has 4 rows, 8 cols = 32 entries, indexed as j*p+k = j*8+k
    B_sides = {
        'M1': {},  # B11+B22
        'M2': {},  # B11
        'M3': {},  # B12-B22
        'M4': {},  # B21-B11
        'M5': {},  # B22
        'M6': {},  # B11+B12
        'M7': {},  # B21+B22
    }
    
    # B11: rows 0-1, cols 0-3
    # B12: rows 0-1, cols 4-7
    # B21: rows 2-3, cols 0-3
    # B22: rows 2-3, cols 4-7
    
    def add_B_block(side_dict, row_range, col_range, coeff):
        for j in row_range:
            for k in col_range:
                idx = j * p + k
                side_dict[idx] = side_dict.get(idx, 0) + coeff
    
    # M1: B11 + B22
    add_B_block(B_sides['M1'], [0, 1], [0, 1, 2, 3], 1)
    add_B_block(B_sides['M1'], [2, 3], [4, 5, 6, 7], 1)
    
    # M2: B11
    add_B_block(B_sides['M2'], [0, 1], [0, 1, 2, 3], 1)
    
    # M3: B12 - B22
    add_B_block(B_sides['M3'], [0, 1], [4, 5, 6, 7], 1)
    add_B_block(B_sides['M3'], [2, 3], [4, 5, 6, 7], -1)
    
    # M4: B21 - B11
    add_B_block(B_sides['M4'], [2, 3], [0, 1, 2, 3], 1)
    add_B_block(B_sides['M4'], [0, 1], [0, 1, 2, 3], -1)
    
    # M5: B22
    add_B_block(B_sides['M5'], [2, 3], [4, 5, 6, 7], 1)
    
    # M6: B11 + B12
    add_B_block(B_sides['M6'], [0, 1], [0, 1, 2, 3], 1)
    add_B_block(B_sides['M6'], [0, 1], [4, 5, 6, 7], 1)
    
    # M7: B21 + B22
    add_B_block(B_sides['M7'], [2, 3], [0, 1, 2, 3], 1)
    add_B_block(B_sides['M7'], [2, 3], [4, 5, 6, 7], 1)
    
    # C-side: for each term, contribution to C[i,k]
    # C has 2 rows, 8 cols = 16 entries, indexed as k*n+i = k*2+i
    # C11 = M1 + M4 - M5 + M7  (C[0, 0:4])
    # C12 = M3 + M5             (C[0, 4:8])
    # C21 = M2 + M4             (C[1, 0:4])
    # C22 = M1 - M2 + M3 + M6  (C[1, 4:8])
    
    C_sides = {
        'M1': {},
        'M2': {},
        'M3': {},
        'M4': {},
        'M5': {},
        'M6': {},
        'M7': {},
    }
    
    def add_C_block(side_dict, row, col_range, coeff):
        for k in col_range:
            idx = k * n + row
            side_dict[idx] = side_dict.get(idx, 0) + coeff
    
    # M1 contributes to C11 (+1) and C22 (+1)
    add_C_block(C_sides['M1'], 0, [0, 1, 2, 3], 1)
    add_C_block(C_sides['M1'], 1, [4, 5, 6, 7], 1)
    
    # M2 contributes to C21 (+1) and C22 (-1)
    add_C_block(C_sides['M2'], 1, [0, 1, 2, 3], 1)
    add_C_block(C_sides['M2'], 1, [4, 5, 6, 7], -1)
    
    # M3 contributes to C12 (+1) and C22 (+1)
    add_C_block(C_sides['M3'], 0, [4, 5, 6, 7], 1)
    add_C_block(C_sides['M3'], 1, [4, 5, 6, 7], 1)
    
    # M4 contributes to C11 (+1) and C21 (+1)
    add_C_block(C_sides['M4'], 0, [0, 1, 2, 3], 1)
    add_C_block(C_sides['M4'], 1, [0, 1, 2, 3], 1)
    
    # M5 contributes to C11 (-1) and C12 (+1)
    add_C_block(C_sides['M5'], 0, [0, 1, 2, 3], -1)
    add_C_block(C_sides['M5'], 0, [4, 5, 6, 7], 1)
    
    # M6 contributes to C22 (+1)
    add_C_block(C_sides['M6'], 1, [4, 5, 6, 7], 1)
    
    # M7 contributes to C11 (+1)
    add_C_block(C_sides['M7'], 0, [0, 1, 2, 3], 1)
    
    # Now build U, V, W
    # For each Strassen term t (M1..M7), we have 8 scalar multiplications
    # (since each term is (1x2)*(2x4) = 8 scalar products)
    # For each scalar product, we create one rank column
    
    term_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
    col = 0
    for t in term_names:
        # For this term, the A-side is a 1x2 vector, B-side is 2x4
        # The 8 scalar products are: for a_idx in {0,1}, b_row in {0,1}, b_col in {0,1,2,3}
        # But actually, the A-side vector has 8 entries (for A[0,0..3], A[1,0..3])
        # and B-side has 32 entries (for B[0..3, 0..7])
        # The "multiplication" is: sum over all (a_idx, b_idx) of A_side[a_idx] * B_side[b_idx] * C_side contribution
        
        # Actually, each Strassen term computes a 1x4 vector:
        # result[k] = sum_{j=0}^{1} A_comb[j] * B_comb[j, k]
        # where A_comb is 1x2 and B_comb is 2x4
        
        # This can be decomposed into 8 rank-1 terms (one for each j, k pair)
        # For j in {0,1}, k in {0,1,2,3}:
        #   A contribution: A_comb[j] to the appropriate A entry
        #   B contribution: B_comb[j, k] to the appropriate B entry
        #   C contribution: 1 to C[i, k] (for the appropriate i)
        
        # Let me think about this differently.
        # The Strassen term M_t computes: (A_t) * (B_t) where A_t is 1x2 and B_t is 2x4
        # Result is 1x4: result[k] = A_t[0]*B_t[0,k] + A_t[1]*B_t[1,k]
        
        # Each such result is then added to C with appropriate signs.
        
        # For the tensor decomposition, each scalar product A_t[j] * B_t[j,k] contributes
        # to C[i, k'] with coefficient from C_sides.
        
        # But wait - the A-side and B-side combinations are linear, so I need to expand them.
        
        # Let me use a different approach: directly construct the rank-56 decomposition
        # by expanding all the linear combinations.
        
        # For term t, the A-side vector is A_sides[t] (length 8)
        # The B-side dict is B_sides[t] (sparse, up to 32 entries)
        # The C-side dict is C_sides[t] (sparse, up to 16 entries)
        
        # The term computes: for each output C[i, k']:
        #   contribution = (sum_{a} A_sides[t][a] * A[a]) * ... 
        #   No, that's not right. The term is a product of two linear forms.
        
        # Actually, each Strassen term M_t is:
        #   M_t = (linear form in A) * (linear form in B) * (linear form in C)
        # But that's not right either. M_t is a 1x4 matrix product of (1x2) and (2x4).
        
        # The correct way: M_t[k] = sum_j (A_t[j]) * (B_t[j,k])
        # where A_t[j] is a linear form in A, and B_t[j,k] is a linear form in B.
        
        # For the tensor decomposition, I need to express:
        # C[i, k'] = sum_t (coeff_t[i,k']) * sum_j A_t[j] * B_t[j, k'_t]
        
        # where k'_t maps the 4 columns of M_t to the 8 columns of C.
        
        # Let me just directly build the rank-56 by enumerating all scalar products.
        
        # For each term t, and for each (j, k) in the 2x4 B block:
        #   A coefficient: A_sides[t] mapped to the appropriate A entry
        #   B coefficient: B_sides[t] mapped to the appropriate B entry  
        #   C coefficient: C_sides[t] mapped to the appropriate C entry
        
        # But this gets complicated. Let me use a simpler direct construction.
        
        pass  # Will handle below
    
    # Simpler approach: directly enumerate
    # For each of the 7 Strassen terms, and for each of the 8 scalar products within it,
    # create one rank column.
    
    col = 0
    for t in term_names:
        # A-side: which A entries are involved and with what coefficients
        # A_sides[t] is a vector of length 8: [A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3]]
        a_vec = A_sides[t]
        
        # B-side: B_sides[t] is a dict {j*8+k: coeff}
        b_dict = B_sides[t]
        
        # C-side: C_sides[t] is a dict {k*2+i: coeff}
        c_dict = C_sides[t]
        
        # The Strassen term computes a 1x4 result that gets added to C.
        # But for the tensor decomposition, I need to decompose this into rank-1 terms.
        
        # Key insight: The term (A_comb) * (B_comb) where A_comb is 1x2 and B_comb is 2x4
        # can be written as sum over the 2x4 = 8 scalar products.
        
        # But A_comb is itself a linear combination of A entries, and B_comb is a linear
        # combination of B entries. So each "scalar product" A_comb[j] * B_comb[j,k] is
        # actually a sum over all A and B entries.
        
        # For the tensor decomposition, I need to express the entire computation as:
        # C[i,k'] = sum_rank U[i*m+j, rank] * V[j'*p+k'', rank] * W[k'''*n+i'', rank]
        
        # This is getting complex. Let me use a numerical approach instead.
        pass
    
    # Let me use a completely different, cleaner approach.
    # I'll construct the decomposition numerically by solving for U, V, W.
    
    # Actually, the cleanest way: for each of the 7 Strassen terms, the term is:
    #   M_t = A_linear_form * B_linear_form
    # where A_linear_form is a scalar (sum of A entries with coefficients) and
    # B_linear_form is a scalar (sum of B entries with coefficients).
    # Then M_t contributes to C entries with coefficients from C_sides[t].
    
    # But M_t is not a scalar - it's a 1x4 vector! So I need to decompose it further.
    
    # Let me reconsider. In the 2x2 block Strassen:
    # A is 2x2 blocks of 1x2, B is 2x2 blocks of 2x4.
    # M1 = (A11+A22)(B11+B22)
    # A11+A22 is 1x2, B11+B22 is 2x4, so M1 is 1x4.
    
    # M1[k] = (A11[0]+A22[0]) * (B11[0,k]+B22[0,k]) + (A11[1]+A22[1]) * (B11[1,k]+B22[1,k])
    
    # This is a sum of 4 terms (2 from A, 2 from B dimensions).
    # Each term is a product of one A entry and one B entry.
    # So M1 has rank 4 (as a bilinear map from A x B to the 1x4 output).
    
    # Wait, no. M1[k] = sum_{j=0}^{1} (A11[j]+A22[j]) * (B11[j,k]+B22[j,k])
    # = (A11[0]+A22[0])*(B11[0,k]+B22[0,k]) + (A11[1]+A22[1])*(B11[1,k]+B22[1,k])
    
    # For fixed k, this is a product of two linear forms in A and B.
    # As a bilinear map, (A11[0]+A22[0])*(B11[0,k]+B22[0,k]) has rank 1.
    # (A11[1]+A22[1])*(B11[1,k]+B22[1,k]) has rank 1.
    # So M1[k] has rank 2 for each k, and the total for all 4 k values is rank 8.
    
    # Yes! Each Strassen term has rank 8 (2 A-dimensions x 4 B-columns... wait)
    # Actually: M1[k] = sum_{j=0}^{1} f_j(A) * g_{j,k}(B)
    # This is a sum of 2 rank-1 terms for each k. For 4 k values, that's 8 rank-1 terms.
    
    # So total rank = 7 * 8 = 56. 
    
    # Now let me construct it properly.
    
    # For each Strassen term t, and for each j in {0,1} (the "inner" dimension of the 1x2 x 2x4 product):
    #   A_linear_form = the j-th component of the A-side combination
    #   B_linear_form_k = the (j,k)-th component of the B-side combination, for k in {0,1,2,3}
    
    # Wait, I need to be more careful. Let me re-derive.
    
    # M_t = A_t * B_t where A_t is 1x2 and B_t is 2x4.
    # M_t[k] = A_t[0]*B_t[0,k] + A_t[1]*B_t[1,k]
    
    # A_t[j] is a linear form in A entries.
    # B_t[j,k] is a linear form in B entries.
    
    # M_t[k] = sum_{j=0}^{1} A_t[j] * B_t[j,k]
    
    # Each A_t[j] * B_t[j,k] is a product of two linear forms, hence rank 1.
    # For fixed t, there are 2*4 = 8 such products.
    
    # Each product contributes to C[i, k'] with coefficient c_dict[k'*n+i].
    
    # So for each (t, j, k), I create one rank column:
    #   U[a, col] = coefficient of A[a] in A_t[j]
    #   V[b, col] = coefficient of B[b] in B_t[j,k]  
    #   W[c, col] = coefficient of C[c] in C_sides[t] (but only for the specific k)
    
    # Wait, the C-side coefficient depends on which C entry this contributes to.
    # M_t[k] contributes to C entries based on C_sides[t].
    
    # Let me re-examine. C_sides[t] maps C indices to coefficients.
    # M_t is a 1x4 vector. M_t[k] gets added to C[i, k_c] for appropriate i and k_c.
    
    # From the Strassen formulas:
    # M1 contributes to C[0, 0:4] with +1 and C[1, 4:8] with +1
    # So M1[k] adds to C[0, k] with +1 and C[1, k+4] with +1 for k in 0..3.
    
    # For the rank-1 term A_t[j] * B_t[j,k], it contributes to:
    #   C[0, k] with coefficient +1 (from M1's C-side)
    #   C[1, k+4] with coefficient +1 (from M1's C-side)
    
    # So W has two non-zero entries for this rank column.
    
    # Let me implement this.
    
    # First, I need to determine A_t[j] for each term and j.
    # A_t is a 1x2 vector. A_t[j] is a linear form in A.
    
    # For M1: A11+A22. A11 = [A[0,0], A[0,1]], A22 = [A[1,2], A[1,3]]
    #   A_t[0] = A[0,0] + A[1,2]
    #   A_t[1] = A[0,1] + A[1,3]
    
    # For M2: A21+A22. A21 = [A[1,0], A[1,1]], A22 = [A[1,2], A[1,3]]
    #   A_t[0] = A[1,0] + A[1,2]
    #   A_t[1] = A[1,1] + A[1,3]
    
    # etc.
    
    # Let me define A_t[j] as a dict {A_index: coeff}
    A_terms = {}
    # M1: A11+A22
    A_terms['M1'] = {
        0: {0: 1, 6: 1},   # A[0,0]+A[1,2] (indices 0 and 6 in flat A)
        1: {1: 1, 7: 1},   # A[0,1]+A[1,3] (indices 1 and 7)
    }
    # M2: A21+A22
    A_terms['M2'] = {
        0: {4: 1, 6: 1},   # A[1,0]+A[1,2]
        1: {5: 1, 7: 1},   # A[1,1]+A[1,3]
    }
    # M3: A11
    A_terms['M3'] = {
        0: {0: 1},
        1: {1: 1},
    }
    # M4: A22
    A_terms['M4'] = {
        0: {6: 1},
        1: {7: 1},
    }
    # M5: A11+A12
    A_terms['M5'] = {
        0: {0: 1, 2: 1},   # A[0,0]+A[0,2]
        1: {1: 1, 3: 1},   # A[0,1]+A[0,3]
    }
    # M6: A21-A11
    A_terms['M6'] = {
        0: {4: 1, 0: -1},  # A[1,0]-A[0,0]
        1: {5: 1, 1: -1},  # A[1,1]-A[0,1]
    }
    # M7: A12-A22
    A_terms['M7'] = {
        0: {2: 1, 6: -1},  # A[0,2]-A[1,2]
        1: {3: 1, 7: -1},  # A[0,3]-A[1,3]
    }
    
    # B_t[j,k] for each term. B_t is 2x4.
    # B_t[j,k] is a linear form in B entries.
    # B flat index: j_row * 8 + k_col
    
    B_terms = {}
    # M1: B11+B22. B11 = B[0:2, 0:4], B22 = B[2:4, 4:8]
    B_terms['M1'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M1'][(j, k)] = {j * 8 + k: 1, (j+2) * 8 + (k+4): 1}
    
    # M2: B11
    B_terms['M2'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M2'][(j, k)] = {j * 8 + k: 1}
    
    # M3: B12-B22
    B_terms['M3'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M3'][(j, k)] = {j * 8 + (k+4): 1, (j+2) * 8 + (k+4): -1}
    
    # M4: B21-B11
    B_terms['M4'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M4'][(j, k)] = {(j+2) * 8 + k: 1, j * 8 + k: -1}
    
    # M5: B22
    B_terms['M5'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M5'][(j, k)] = {(j+2) * 8 + (k+4): 1}
    
    # M6: B11+B12
    B_terms['M6'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M6'][(j, k)] = {j * 8 + k: 1, j * 8 + (k+4): 1}
    
    # M7: B21+B22
    B_terms['M7'] = {}
    for j in range(2):
        for k in range(4):
            B_terms['M7'][(j, k)] = {(j+2) * 8 + k: 1, (j+2) * 8 + (k+4): 1}
    
    # C-side: for each term t and each k in 0..3, which C entries get which coefficient.
    # C flat index: k_col * 2 + i_row
    
    C_terms = {}
    # M1: C[0, 0:4] +1, C[1, 4:8] +1
    C_terms['M1'] = {}
    for k in range(4):
        C_terms['M1'][k] = {k * 2 + 0: 1, (k+4) * 2 + 1: 1}
    
    # M2: C[1, 0:4] +1, C[1, 4:8] -1
    C_terms['M2'] = {}
    for k in range(4):
        C_terms['M2'][k] = {k * 2 + 1: 1, (k+4) * 2 + 1: -1}
    
    # M3: C[0, 4:8] +1, C[1, 4:8] +1
    C_terms['M3'] = {}
    for k in range(4):
        C_terms['M3'][k] = {(k+4) * 2 + 0: 1, (k+4) * 2 + 1: 1}
    
    # M4: C[0, 0:4] +1, C[1, 0:4] +1
    C_terms['M4'] = {}
    for k in range(4):
        C_terms['M4'][k] = {k * 2 + 0: 1, k * 2 + 1: 1}
    
    # M5: C[0, 0:4] -1, C[0, 4:8] +1
    C_terms['M5'] = {}
    for k in range(4):
        C_terms['M5'][k] = {k * 2 + 0: -1, (k+4) * 2 + 0: 1}
    
    # M6: C[1, 4:8] +1
    C_terms['M6'] = {}
    for k in range(4):
        C_terms['M6'][k] = {(k+4) * 2 + 1: 1}
    
    # M7: C[0, 0:4] +1
    C_terms['M7'] = {}
    for k in range(4):
        C_terms['M7'][k] = {k * 2 + 0: 1}
    
    # Now build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in term_names:
        for j in range(2):
            for k in range(4):
                # A-side: A_terms[t][j]
                for a_idx, coeff in A_terms[t][j].items():
                    U[a_idx][col] = coeff
                
                # B-side: B_terms[t][(j, k)]
                for b_idx, coeff in B_terms[t][(j, k)].items():
                    V[b_idx][col] = coeff
                
                # C-side: C_terms[t][k]
                for c_idx, coeff in C_terms[t][k].items():
                    W[c_idx][col] = coeff
                
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build Strassen block decomposition (rank 56)
    U, V, W = _build_strassen_block_2x2(n, m, p)
    rank = len(U[0])
    
    # Verify
    if _verify(U, V, W, n, m, p, trials=100, tol=1e-9):
        print(f"Strassen block decomposition verified at rank {rank}")
        return {"U": U, "V": V, "W": W}
    else:
        print("Strassen verification failed, falling back to schoolbook")
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
