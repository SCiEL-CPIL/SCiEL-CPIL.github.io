"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # First, build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        c = start_col
        # M1
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1
    
    # A is 2x4: indices i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
    # For each pair of rows (r1,r2) and pair of cols (c1,c2):
    # b00=B[r1,c1], b01=B[r1,c2], b10=B[r2,c1], b11=B[r2,c2]
    
    # Block structure: split A into left 2 cols and right 2 cols
    # Split B into 3 blocks of 2 cols each
    # For each A block and B col-block, we have a 2x2 * 2x2 multiplication
    
    # B col block 0: cols 0,1
    #   Rows 0,1: b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=6, b11=B[1,1]=7
    #   Rows 2,3: b00=B[2,0]=12, b01=B[2,1]=13, b10=B[3,0]=18, b11=B[3,1]=19
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    
    # W indices for C cols 0,1: C[i,k] -> W[k*n+i]
    # C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta for M_{n,m,p}"""
        # For each (i,j,k), the tensor value should be 1 if this is a valid MM tensor
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Actually T[i,j,k] = 1 for the MM tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies among the 42 rank-1 terms
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # If we can find coefficients c_r such that sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0,
    # then we can eliminate one term and reduce rank.
    
    # Flatten each rank-1 tensor into a vector and check linear independence
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Each rank-1 term u_r ⊗ v_r ⊗ w_r has shape (n*m, m*p, p*n)
    # Flatten to vector of length n*m * m*p * p*n = 8 * 24 * 12 = 2304
    # Build matrix where column r is the flattened u_r ⊗ v_r ⊗ w_r
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    
    # Stack into matrix: shape (2304, 42)
    M = np.array(flat_terms).T  # (2304, 42)
    
    # Check rank of M - if < 42, there are linear dependencies
    # Use SVD to find the null space
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        # Find singular values that are effectively zero
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < rank:
            # There are (rank - effective_rank) linear dependencies
            # The null space is spanned by the last (rank - effective_rank) rows of Vt_svd
            null_space = Vt_svd[effective_rank:, :]  # shape ((rank-eff_rank), rank)
            
            # For each null space vector, we can eliminate one column
            # We want to find the best set of columns to eliminate
            # Simple approach: for each null vector, find the largest component and eliminate that column
            
            new_U = np.array(U_np, dtype=float)
            new_V = np.array(V_np, dtype=float)
            new_W = np.array(W_np, dtype=float)
            eliminated = set()
            
            for null_vec in null_space:
                if time.time() > deadline - 10:
                    break
                # Find the column with the largest absolute coefficient
                # that hasn't been eliminated yet
                candidates = [c for c in range(rank) if c not in eliminated]
                if not candidates:
                    break
                max_idx = max(candidates, key=lambda c: abs(null_vec[c]))
                max_val = abs(null_vec[max_idx])
                if max_val < 1e-8:
                    continue
                
                # Express column max_idx as linear combination of others
                # null_vec[c] * term_c = 0 for all c
                # So term_max_idx = -sum_{c != max_idx} (null_vec[c] / null_vec[max_idx]) * term_c
                # We can absorb this into the other terms
                for c in candidates:
                    if c == max_idx:
                        continue
                    coeff = -null_vec[c] / null_vec[max_idx]
                    if abs(coeff) > 1e-10:
                        new_U[:, c] += coeff * new_U[:, max_idx]
                        new_V[:, c] += coeff * new_V[:, max_idx]
                        new_W[:, c] += coeff * new_W[:, max_idx]
                eliminated.add(max_idx)
            
            if eliminated:
                # Reconstruct without eliminated columns
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = new_U[:, keep]
                final_V = new_V[:, keep]
                final_W = new_W[:, keep]
                
                # Verify the new decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(new_rank):
                                total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                            if abs(total - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists with integer/half-integer entries
                    def to_fraction_list(arr):
                        result = []
                        for row in arr:
                            result.append([])
                            for val in row:
                                # Round to nearest half-integer
                                frac = round(val * 2) / 2
                                # If it's very close to an integer, make it integer
                                if abs(frac - round(frac)) < 1e-10:
                                    frac = int(round(frac))
                                result[-1].append(frac)
                        return result
                    
                    return {
                        "U": to_fraction_list(final_U),
                        "V": to_fraction_list(final_V),
                        "W": to_fraction_list(final_W)
                    }
    except Exception:
        pass
    
    # Try an alternative approach: use exact rational arithmetic to find dependencies
    # This may find dependencies that SVD misses due to numerical issues
    try:
        from fractions import Fraction
        
        U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
        
        # Build the matrix of flattened terms using exact arithmetic
        # Each term is a vector of length n*m * m*p * p*n
        dim = n * m * m * p * p * n
        terms = []
        for r in range(rank):
            term = [Fraction(0)] * dim
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        idx = i * (m*p) * (p*n) + j * (p*n) + k
                        term[idx] = U_frac[i][r] * V_frac[j][r] * W_frac[k][r]
            terms.append(term)
        
        # Find the rank of the matrix using Gaussian elimination with exact arithmetic
        # This is expensive but may find dependencies that SVD misses
        if time.time() < deadline - 30:
            # Simple Gaussian elimination to find null space
            # Represent as list of lists of Fractions
            matrix = [list(t) for t in terms]  # (rank, dim) - actually (42, 2304)
            # Transpose to (dim, rank) for row reduction
            matrix_T = [[matrix[r][c] for r in range(rank)] for c in range(dim)]
            
            # Row reduce to find pivot columns
            pivot_cols = []
            current_row = 0
            for col in range(rank):
                if current_row >= dim:
                    break
                # Find pivot
                pivot = None
                for row in range(current_row, dim):
                    if matrix_T[row][col] != 0:
                        pivot = row
                        break
                if pivot is None:
                    continue
                # Swap rows
                matrix_T[current_row], matrix_T[pivot] = matrix_T[pivot], matrix_T[current_row]
                # Eliminate below
                for row in range(current_row + 1, dim):
                    if matrix_T[row][col] != 0:
                        factor = matrix_T[row][col] / matrix_T[current_row][col]
                        for c in range(col, rank):
                            matrix_T[row][c] -= factor * matrix_T[current_row][c]
                pivot_cols.append(col)
                current_row += 1
            
            effective_rank_exact = len(pivot_cols)
            if effective_rank_exact < rank:
                # Find null space vectors
                null_vecs = []
                free_cols = [c for c in range(rank) if c not in pivot_cols]
                for free_col in free_cols:
                    if time.time() > deadline - 15:
                        break
                    null_vec = [Fraction(0)] * rank
                    null_vec[free_col] = Fraction(1)
                    # Back substitute
                    for i in range(len(pivot_cols) - 1, -1, -1):
                        col = pivot_cols[i]
                        total = Fraction(0)
                        for c in range(col + 1, rank):
                            if matrix_T[i][c] != 0:
                                total += matrix_T[i][c] * null_vec[c]
                        null_vec[col] = -total / matrix_T[i][col]
                    null_vecs.append(null_vec)
                
                # Use null vectors to eliminate terms
                new_U = np.array(U_np, dtype=float)
                new_V = np.array(V_np, dtype=float)
                new_W = np.array(W_np, dtype=float)
                eliminated = set()
                
                for null_vec in null_vecs:
                    if time.time() > deadline - 10:
                        break
                    candidates = [c for c in range(rank) if c not in eliminated]
                    if not candidates:
                        break
                    # Convert to float for finding max
                    null_vec_float = [float(nv) for nv in null_vec]
                    max_idx = max(candidates, key=lambda c: abs(null_vec_float[c]))
                    max_val = abs(null_vec_float[max_idx])
                    if max_val < 1e-8:
                        continue
                    
                    for c in candidates:
                        if c == max_idx:
                            continue
                        coeff = -float(null_vec[c]) / float(null_vec[max_idx])
                        if abs(coeff) > 1e-10:
                            new_U[:, c] += coeff * new_U[:, max_idx]
                            new_V[:, c] += coeff * new_V[:, max_idx]
                            new_W[:, c] += coeff * new_W[:, max_idx]
                    eliminated.add(max_idx)
                
                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for r in range(new_rank):
                                    total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                                if abs(total - 1.0) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        def to_fraction_list(arr):
                            result = []
                            for row in arr:
                                result.append([])
                                for val in row:
                                    frac = round(val * 2) / 2
                                    if abs(frac - round(frac)) < 1e-10:
                                        frac = int(round(frac))
                                    result[-1].append(frac)
                            return result
                        
                        return {
                            "U": to_fraction_list(final_U),
                            "V": to_fraction_list(final_V),
                            "W": to_fraction_list(final_W)
                        }
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
