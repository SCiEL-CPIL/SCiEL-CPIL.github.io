import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    
    base_rank = 42
    # Use Fraction for exact arithmetic
    U = [[Fraction(0) for _ in range(base_rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(base_rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(base_rank)] for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1 = (a+d)(e+h)
        U[a][start_col] += 1; U[d][start_col] += 1
        V[e][start_col] += 1; V[h][start_col] += 1
        W[c11][start_col] += 1; W[c22][start_col] += 1
        start_col += 1
        
        # M2 = (c+d)e
        U[c][start_col] += 1; U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1; W[c22][start_col] -= 1
        start_col += 1
        
        # M3 = a(f-h)
        U[a][start_col] += 1
        V[f][start_col] += 1; V[h][start_col] -= 1
        W[c12][start_col] += 1; W[c22][start_col] += 1
        start_col += 1
        
        # M4 = d(g-e)
        U[d][start_col] += 1
        V[g][start_col] += 1; V[e][start_col] -= 1
        W[c11][start_col] += 1; W[c21][start_col] += 1
        start_col += 1
        
        # M5 = (a+b)h
        U[a][start_col] += 1; U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1; W[c12][start_col] += 1
        start_col += 1
        
        # M6 = (c-a)(e+f)
        U[c][start_col] += 1; U[a][start_col] -= 1
        V[e][start_col] += 1; V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7 = (b-d)(g+h)
        U[b][start_col] += 1; U[d][start_col] -= 1
        V[g][start_col] += 1; V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col
    
    for k in range(3):
        A_L_idx = [0, 1, 4, 5]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        A_R_idx = [2, 3, 6, 7]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
        
    # Feature matrix F[i,j,k,r]
    num_i = n * m
    num_j = m * p
    num_k = p * n
    total_features = num_i * num_j * num_k
    
    F = [[Fraction(0) for _ in range(base_rank)] for _ in range(total_features)]
    
    idx = 0
    for i in range(num_i):
        for j in range(num_j):
            for k in range(num_k):
                for r in range(base_rank):
                    F[idx][r] = U[i][r] * V[j][r] * W[k][r]
                idx += 1
    
    # Gaussian Elimination to find dependencies
    # We want to find the rank of F and identify redundant columns.
    # Since F is 2304 x 42, we can do row reduction on F^T (42 x 2304) to find column dependencies?
    # No, we want to know if a column is in the span of others.
    # This is equivalent to finding the null space of F? No.
    # We want to find a subset of columns that spans the same space.
    # We can perform column reduction on F.
    # F is (Features x Rank).
    # We want to find a basis for the column space.
    
    # Convert to list of lists of Fractions for row reduction?
    # Actually, we can just do Gaussian elimination on the transpose F^T (Rank x Features).
    # The rank of F is the rank of F^T.
    # We can find a basis of the column space of F by doing row reduction on F^T?
    # No, row reduction on F^T gives a basis for the row space of F^T, which is the column space of F.
    # The non-zero rows of the RREF of F^T are a basis for the column space of F.
    # But we want to identify which original columns are redundant.
    
    # Better: Do column reduction on F.
    # We iterate through columns 0 to 41.
    # Maintain a set of independent columns.
    # For each new column, check if it is in the span of the previous independent columns.
    # If yes, it is redundant.
    
    independent_cols = []
    # We need to solve linear systems.
    # To be fast, we can use numpy for the heavy lifting but verify with fractions?
    # Or just use fractions for the 42 columns.
    # 2304 x 42 matrix.
    # We can form a matrix M of size (2304 x 42).
    # We perform Gaussian elimination on M to find the rank and pivot columns.
    
    # Let's use a simple Gaussian elimination on the transpose (42 x 2304) to find pivots.
    # Pivots in the row space of F^T correspond to independent columns in F.
    
    F_T = [[F[i][j] for i in range(total_features)] for j in range(base_rank)]
    
    # Row reduce F_T
    rank = 0
    pivots = []
    for col in range(total_features):
        if rank >= base_rank:
            break
        # Find pivot row
        pivot_row = -1
        for r in range(rank, base_rank):
            if F_T[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        # Swap
        F_T[rank], F_T[pivot_row] = F_T[pivot_row], F_T[rank]
        # Normalize
        pivot_val = F_T[rank][col]
        if pivot_val != 1:
            for c in range(col, total_features):
                F_T[rank][c] /= pivot_val
        # Eliminate
        for r in range(base_rank):
            if r != rank and F_T[r][col] != 0:
                factor = F_T[r][col]
                for c in range(col, total_features):
                    F_T[r][c] -= factor * F_T[rank][c]
        pivots.append(col)
        rank += 1
    
    # The pivots are the column indices in F_T that are independent.
    # These correspond to the row indices in F?
    # No. F_T is (Rank x Features).
    # The rows of F_T are the columns of F.
    # So the pivot columns in F_T (which are features) don't directly tell us which columns of F are independent.
    # We need to find which columns of F are linearly independent.
    # This is equivalent to finding the rank of F.
    # We can do column reduction on F directly.
    
    # Let's redo: Column reduction on F (2304 x 42).
    # We maintain a basis of independent columns.
    # Since 42 is small, we can just try to add each column to the basis.
    
    basis = []
    basis_indices = []
    
    for r in range(base_rank):
        if time.time() > deadline:
            break
        col_vec = [F[i][r] for i in range(total_features)]
        
        # Check if col_vec is in span of basis
        # Solve B * alpha = col_vec
        # B is (total_features x len(basis))
        # This is expensive.
        
        # Alternative: Use the RREF of the matrix formed by basis + col_vec.
        # If the rank increases, it's independent.
        
        # To speed up, we can use numpy for the check, and only use fractions if it's close?
        # Or just use fractions. 2304 x 42 is small enough for fractions if optimized?
        # 2304 * 42 * 42 operations ~ 4 million. Should be fast.
        
        # Let's use a simpler check:
        # Compute the rank of [Basis | Col] vs Rank of Basis.
        
        # We can maintain the RREF of the Basis.
        # This is complex to update.
        
        # Let's stick to the previous method: 
        # We know the rank is likely 36-38.
        # We can just try to remove columns using the float method first to get a candidate set,
        # then verify with fractions.
        
        pass
        
    # Given the time constraint and complexity, let's stick to the float method but with a better search strategy.
    # The float method found 37.
    # Let's try to find if 36 is possible by checking all pairs of removals more carefully?
    # Or just return the best found by the float method, but ensure it's valid with a final fraction check.
    
    # Re-run the float search to get best_indices
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    F_float = np.zeros((total_features, base_rank), dtype=np.float64)
    idx = 0
    for i in range(num_i):
        for j in range(num_j):
            for k in range(num_k):
                for r in range(base_rank):
                    F_float[idx, r] = U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
                idx += 1
                
    best_rank = base_rank
    best_indices = list(range(base_rank))
    
    # Greedy removal
    current_indices = list(range(base_rank))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_indices = list(current_indices)
        for r_idx in range(len(new_indices)):
            if time.time() > deadline:
                break
            r = new_indices[r_idx]
            S = [s for s in new_indices if s != r]
            if len(S) == 0:
                continue
            F_S = F_float[:, S]
            F_r = F_float[:, r]
            
            try:
                alpha, residuals, rank_S, sv = np.linalg.lstsq(F_S, F_r, rcond=None)
                reconstructed = F_S @ alpha
                residual_norm = np.linalg.norm(F_r - reconstructed)
                if residual_norm < 1e-10:
                    new_indices = S
                    improved = True
                    break
            except:
                pass
        if improved:
            current_indices = new_indices
            best_rank = len(current_indices)
            
    # Final verification with Fractions
    U_final = [[U[i][r] for r in best_indices] for i in range(n * m)]
    V_final = [[V[j][r] for r in best_indices] for j in range(m * p)]
    W_final = [[W[k][r] for r in best_indices] for k in range(p * n)]
    
    # Check validity
    valid = True
    for a in range(n):
        for c in range(p):
            for b in range(m):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                val = Fraction(0)
                for r in range(len(best_indices)):
                    val += U_final[i][r] * V_final[j][r] * W_final[k][r]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to full rank
        U_final = U
        V_final = V
        W_final = W
        
    return {"U": U_final, "V": V_final, "W": W_final}
