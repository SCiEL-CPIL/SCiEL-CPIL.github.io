import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    
    start_time = time.time()
    deadline = start_time + 130
    
    # Strategy: Build a structured decomposition using Strassen for 4x4x4 and schoolbook for 4x4x3,
    # then apply aggressive linear algebra reduction to find dependencies.
    # This is a structural change from pure schoolbook to a hybrid block approach + reduction.
    
    # Part 1: 4x4 * 4x4 multiplication using Strassen (Rank 49)
    # Part 2: 4x4 * 4x3 multiplication using Schoolbook (Rank 48)
    # Total initial rank: 97
    
    # We will build the full tensor explicitly with these components and then reduce.
    
    # --- Building Strassen 4x4x4 ---
    # Strassen 2x2x2 has rank 7.
    # 4x4x4 can be done with 49 mults (Strassen^2).
    # Indices for 4x4x4: i in 0..3, j in 0..3, k in 0..3.
    # We need U (16x49), V (16x49), W (16x49) for the first 4 columns of B.
    
    # Let's define the 49 terms for 4x4x4.
    # We can generate them by nesting the 2x2x2 Strassen.
    
    # M(a, b) for 2x2x2:
    # M1 = (a00 - a11) * (b11 - b00)
    # M2 = (a00 + a11) * b11
    # M3 = a11 * (b10 - b00)
    # M4 = a00 * (b01 - b11)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # For 4x4x4, we split A into 4 blocks of 2x2: A00, A01, A10, A11.
    # B into 4 blocks: B00, B01, B10, B11.
    # C00 = M(A00, B00) + M(A01, B10)
    # C01 = M(A00, B01) + M(A01, B11)
    # C10 = M(A10, B00) + M(A11, B10)
    # C11 = M(A10, B01) + M(A11, B11)
    
    # Each M is 7 terms. So 4 * 7 * 2 = 56? No.
    # Wait, Strassen 4x4x4 is rank 49.
    # The 49 terms are computed by applying Strassen to the 4x4 blocks.
    # It is effectively 7^2 = 49 multiplications.
    
    # Let's just use the known 49-term decomposition for 4x4x4.
    # Or simpler: just use schoolbook for 4x4x4 (16 terms? No, 4*4*4=64 terms) 
    # and schoolbook for 4x4x3 (4*4*3=48 terms). Total 112.
    # The current program does this.
    
    # The inspirations suggest rank 93 is possible.
    # 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3) = 97.
    # Can we do better on the 4x4x3 part?
    # 4x4x3 rank is at least 4*3 = 12? No, at least 4*4*3 / max(4,4,3) = 16?
    # Schoolbook is 48.
    # Maybe we can mix them.
    
    # Let's try to build the 97-rank decomposition explicitly.
    
    # 1. Strassen 4x4x4 (Rank 49)
    # We need to generate the U, V, W matrices for the first 4 columns of B.
    
    def _strassen_4x4x4():
        """
        Returns U, V, W for 4x4 * 4x4 -> 4x4.
        Shapes: (16, 49), (16, 49), (16, 49).
        """
        U = np.zeros((16, 49), dtype=np.float64)
        V = np.zeros((16, 49), dtype=np.float64)
        W = np.zeros((16, 49), dtype=np.float64)
        
        # We will use the recursive definition.
        # Term index t in 0..48.
        # t = 7 * t1 + t2, where t1, t2 in 0..6.
        # t1 corresponds to the "outer" Strassen step, t2 to the "inner".
        
        # This is complex to generate programmatically without bugs.
        # Let's use a known explicit list or a simpler method.
        # Actually, let's just use schoolbook for 4x4x4 (64 terms) and schoolbook for 4x4x3 (48 terms).
        # Total 112.
        # Then reduce.
        
        # The reduction step is the key. If we start with 112, can we get to 93?
        # The inspirations say "yes" (score 1.2043 implies rank ~ 112/1.2043 ~ 93).
        # So the reduction is effective.
        
        # Let's implement the 112-rank schoolbook and then a more aggressive reduction.
        
        pass

    # Let's stick to the 112-rank schoolbook but improve the reduction.
    # The current reduction only checks for duplicates.
    # We can check for linear dependencies in the tensor slices.
    
    # A term r is redundant if its corresponding vector in the tensor space is in the span of others.
    # The tensor T is of size 4*4*7 = 112.
    # Each term r defines a vector t_r in R^112.
    # If {t_r} is linearly dependent, we can reduce the rank.
    # We can compute the rank of the matrix M where M[:, r] = t_r.
    # If rank(M) < 112, we can remove redundant terms.
    
    # Let's build the 112-rank schoolbook.
    U = np.zeros((16, 112), dtype=np.float64)
    V = np.zeros((28, 112), dtype=np.float64)
    W = np.zeros((28, 112), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
                
    # Now, reduce the rank.
    # Construct the tensor matrix T_mat of shape (112, 112) where T_mat[i, r] is the i-th entry of term r.
    # i = k * 4 + i (output index)
    # Actually, the tensor is T_{i, j, k}.
    # Let's flatten the indices: idx = (i * 4 + j) * 7 + k? No.
    # The tensor is defined by T_{i, j, k} = 1.
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # We want to find the minimum number of terms.
    # This is equivalent to finding the rank of the 3-tensor.
    # We can unfold the tensor into a matrix.
    # Unfolding along the first mode: T_(1) of shape (4, 4*7) = (4, 28).
    # Rank of T_(1) is 4.
    # This doesn't help directly with the decomposition rank.
    
    # However, we can check for linear dependencies among the terms.
    # Let's represent each term r as a vector of length 112.
    # v_r[i, j, k] = U[i, j, r] * V[j, k, r] * W[k, i, r].
    # We can form a matrix M of shape (112, 112) where M[idx, r] = v_r[idx].
    # If the columns of M are linearly dependent, we can remove some.
    
    M = np.zeros((112, 112), dtype=np.float64)
    for r in range(112):
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    idx = (i * 4 + j) * 7 + k
                    M[idx, r] = U[i * 4 + j, r] * V[j * 7 + k, r] * W[k * 4 + i, r]
                    
    # Compute the rank of M.
    # Use SVD or QR.
    # Since M is 112x112, SVD is feasible.
    
    if time.time() < deadline - 10:
        try:
            # Use QR with column pivoting to find the rank and a basis.
            # Or just compute the rank.
            # If rank is R, we can keep R linearly independent columns.
            
            # Let's use np.linalg.matrix_rank.
            rank_M = np.linalg.matrix_rank(M, tol=1e-10)
            
            # If rank_M < 112, we can reduce.
            # We need to find which columns to keep.
            # We can use the pivots from QR.
            
            # scipy.linalg.qr with pivoting is better.
            from scipy.linalg import qr
            Q, R, piv = qr(M, pivoting=True)
            
            # The rank is the number of non-zero diagonal elements in R.
            # Let's find the pivot indices that correspond to independent columns.
            
            # Determine the rank from R.
            diag_R = np.abs(np.diag(R))
            threshold = 1e-10 * max(diag_R) if len(diag_R) > 0 else 1e-10
            num_pivots = np.sum(diag_R > threshold)
            
            if num_pivots < 112:
                # Keep the first 'num_pivots' pivots.
                keep_cols = piv[:num_pivots]
                
                U_new = U[:, keep_cols]
                V_new = V[:, keep_cols]
                W_new = W[:, keep_cols]
                
                # Verify the new decomposition.
                # It should be exact because we removed linearly dependent vectors.
                
                # Check if verification is fast enough.
                # 112 * 4 * 4 * 7 = 12544 checks. Fast.
                
                # Let's verify.
                valid = True
                for i in range(4):
                    for j in range(4):
                        for k in range(7):
                            val = 0
                            for r in range(num_pivots):
                                val += U_new[i * 4 + j, r] * V_new[j * 7 + k, r] * W_new[k * 4 + i, r]
                            if not np.isclose(val, 1.0, atol=1e-9):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        
                if valid:
                    U = U_new
                    V = V_new
                    W = W_new
        except Exception:
            pass
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
