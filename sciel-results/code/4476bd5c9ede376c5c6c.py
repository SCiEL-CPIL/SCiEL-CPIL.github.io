"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Initialize with Schoolbook (Rank 125)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # c = i*m*p + j*p + k
    U_list = []
    V_list = []
    W_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list.append((i, j))
                V_list.append((j, k))
                W_list.append((k, i))
    
    # Convert to numpy for speed
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    for c, (u_idx, v_idx, w_idx) in enumerate(zip(U_list, V_list, W_list)):
        i, j = u_idx
        j2, k = v_idx
        k2, i2 = w_idx
        U[i*m+j, c] = 1
        V[j*p+k, c] = 1
        W[k*n+i, c] = 1
        
    def check_validity(U, V, W, tol=1e-6):
        # Check if U * V * W reconstructs the MM tensor
        # MM[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Flatten indices
        # U is (n*m, R), V is (m*p, R), W is (p*n, R)
        # We need to check for all i,j,k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == (1 if i==i, j==j, k==k else 0)
        # Actually, the tensor is T[i, j, k] = 1 if it corresponds to (A[i,j] * B[j,k]).
        # The standard MM tensor T has T[i,j,k] = 1.
        # Let's verify a few random points or the whole thing if fast enough.
        # R is ~120. n*m*p is 125.
        # We can compute the full tensor and compare.
        # T_reconstructed[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient check:
        # Iterate over all i, j, k
        # This is 125 * R operations. 125 * 120 = 15000. Very fast.
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    val = np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
                    if abs(val - 1.0) > tol:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        return valid

    # 2. Iterative Reduction
    # We want to remove columns.
    # A column c can be removed if its contribution can be spanned by other columns.
    # Contribution of c: U[:, c] * V[:, c] * W[:, c] (tensor product)
    # We maintain a list of active columns.
    active_cols = list(range(n*m*p))
    
    # To speed up, we can try to remove columns in a specific order.
    # Greedy: try to remove column c if it's linearly dependent on others in the tensor space.
    # The tensor space is n*m*p dimensional (125).
    # We can represent each term T_c as a vector in R^{125}.
    # If T_c is in the span of {T_d | d != c}, we can remove c.
    
    # Construct the matrix of terms
    # Rows: 125 indices (i, j, k)
    # Cols: R terms
    # M[idx, c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    M = np.zeros((n*m*p, n*m*p))
    idx_map = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                M[idx, :] = U[u_row, :] * V[v_row, :] * W[w_row, :]
                
    # Now, we have M (125 x 125).
    # We want to find the minimum number of columns such that their span covers the "target" tensor.
    # Wait, the target tensor is T[i,j,k]=1.
    # The condition is that the sum of selected terms equals T.
    # Actually, the standard decomposition is a basis for the space of matrices? No.
    # The MM tensor is a specific vector in R^{125}.
    # We need sum_{c in S} alpha_c T_c = T_target.
    # In the schoolbook case, alpha_c = 1 for all c, and T_c are the basis vectors e_{i,j,k}.
    # So T_target = sum e_{i,j,k}.
    
    # So we are looking for a subset S of columns and coefficients alpha such that:
    # sum_{c in S} alpha_c T_c = T_target
    # And we want to minimize |S|.
    # Note: We can scale T_c by absorbing into U, V, W.
    # So we just need the span of {T_c | c in S} to contain T_target.
    
    # This is equivalent to: T_target is in the column space of M_S (M restricted to S).
    # We want to find the smallest S such that rank(M_S) == rank([M_S, T_target]).
    # Since T_target is the sum of basis vectors, and M is the identity matrix (for schoolbook),
    # any subset S that covers all indices? No.
    # If we use Strassen, T_c are not basis vectors. They are linear combinations.
    # But we started with schoolbook.
    # If we just pick columns, we can't "change" the columns, only select them.
    # Wait, the standard reduction allows modifying U, V, W.
    # But if we modify U, V, W, the terms T_c change.
    # The "Gaussian elimination" approach on the tensor network allows modifying the other terms to eliminate the dependence of one term.
    
    # Let's try a simpler heuristic first:
    # Can we find a set of columns S such that T_target is in span(T_S)?
    # For schoolbook, T_c = e_c. T_target = sum e_c.
    # T_target is in span(e_1, ..., e_R) iff all e_c are present?
    # No, span(e_1, ..., e_R) is the whole space. T_target is in the space.
    # But we need T_target = sum alpha_c e_c.
    # This implies alpha_c = 1 for all c.
    # So we can't remove any column from the *initial* schoolbook set without changing the other columns.
    
    # So we MUST modify the columns.
    # Algorithm:
    # 1. Start with Schoolbook.
    # 2. Pick a column c to remove.
    # 3. Find a linear combination of other columns that matches T_c?
    #    No, we want to eliminate T_c from the sum.
    #    We have T_target = sum_{d} T_d.
    #    We want T_target = sum_{d != c} T'_d.
    #    This implies T'_d = T_d + correction_d.
    #    We need sum_{d != c} (T_d + correction_d) = T_target.
    #    sum_{d != c} T_d + sum_{d != c} correction_d = T_target.
    #    (T_target - T_c) + sum_{d != c} correction_d = T_target.
    #    sum_{d != c} correction_d = T_c.
    #    So we need to express T_c as a linear combination of the *other* terms T_d (d != c).
    #    If T_c is in span({T_d | d != c}), we can "absorb" it.
    #    Specifically, if T_c = sum_{d != c} beta_d T_d, then:
    #    T_target = sum_{d != c} T_d + T_c = sum_{d != c} (1 + beta_d) T_d.
    #    We can define new terms T'_d = (1 + beta_d) T_d.
    #    And we can absorb the scaling (1+beta_d) into U_d, V_d, W_d.
    #    So if T_c is in the span of the others, we can remove it.
    
    # So the problem reduces to:
    # Find the largest subset of columns S such that for every c NOT in S, T_c is in span({T_d | d in S}).
    # Actually, we want to minimize |S|.
    # This is equivalent to finding a basis for the space spanned by {T_c} that is a subset of {T_c}?
    # No. The T_c are vectors in R^{125}.
    # We want to find a subset S such that T_target is in span({T_c | c in S}).
    # AND we want to minimize |S|.
    
    # Let's compute the rank of the matrix M (125x125).
    # If rank(M) < 125, there are dependencies.
    # But for schoolbook, M is Identity. Rank 125.
    # So no T_c is in the span of others.
    # So we cannot remove any column from the pure schoolbook set.
    
    # We need to *change* the basis.
    # This requires a search over transformations.
    # Given the time limit, a full search is hard.
    # However, the inspiration mentions "2+2+1 partition with Strassen".
    # This is a specific construction.
    # Let's try to implement the 2+2+1 Strassen construction directly, as it is known to be rank 117.
    # This is a "static" construction, not a search, but it's a different strategy than "just return schoolbook".
    
    # Strategy: Implement the 2+2+1 Strassen decomposition.
    # Partition 5x5 into 2x2, 2x2, 1x1 blocks?
    # A = [A11 A12 A13; A21 A22 A23; A31 A32 A33] where A11, A22 are 2x2, A33 is 1x1.
    # Indices:
    # 0,1 -> Block 1 (2x2)
    # 2,3 -> Block 2 (2x2)
    # 4 -> Block 3 (1x1)
    
    # We apply Strassen to the 2x2 blocks and schoolbook to 1x1 and cross terms?
    # Actually, the "2+2+1" likely refers to splitting the dimension 5 into 2, 2, 1.
    # The number of blocks is 3x3 = 9.
    # Blocks:
    # (0,0): 2x2 (Strassen) -> Rank 7 (instead of 8)
    # (0,1): 2x2 (Strassen) -> Rank 7
    # (0,2): 2x1 (Schoolbook) -> Rank 2 (2*1*1)
    # (1,0): 2x2 (Strassen) -> Rank 7
    # (1,1): 2x2 (Strassen) -> Rank 7
    # (1,2): 2x1 (Schoolbook) -> Rank 2
    # (2,0): 1x2 (Schoolbook) -> Rank 2
    # (2,1): 1x2 (Schoolbook) -> Rank 2
    # (2,2): 1x1 (Schoolbook) -> Rank 1
    
    # Total Rank = 4*7 + 4*2 + 1 = 28 + 8 + 1 = 37?
    # Wait. The blocks are not independent in the tensor decomposition sense unless we treat them as separate sub-problems.
    # Yes, matrix multiplication is block-wise.
    # C_ij = sum_k A_ik B_kj.
    # The tensor for C_11 depends on A_1k and B_k1.
    # If we partition indices, the terms for different output blocks C_ij are disjoint in the output index i,j.
    # So we can sum the ranks.
    
    # Let's verify the block sizes.
    # Output C is 5x5.
    # C[0:2, 0:2] depends on A[0:2, :] and B[:, 0:2].
    # This is a 2x5 * 5x2 multiplication.
    # This is NOT a 2x2 * 2x2 multiplication.
    # Strassen applies to square matrices of size N.
    # If we use Strassen on the whole 5x5, we can't directly.
    # But we can use Strassen on sub-blocks IF the other dimensions allow it.
    # A[0:2, 0:2] * B[0:2, 0:2] contributes to C[0:2, 0:2].
    # But A[0:2, 2:5] * B[2:5, 0:2] also contributes.
    # So the blocks are coupled.
    
    # The "2+2+1" strategy likely refers to a specific known decomposition for 5x5.
    # Known rank for 5x5 is 117 (or 116?).
    # Schoolbook 125.
    # 2+2+1 partition:
    # Split indices into sets I1={0,1}, I2={2,3}, I3={4}.
    # We have 9 output blocks C_ab for a,b in {1,2,3}.
    # Size of C_ab is |I_a| x |I_b|.
    # The contribution to C_ab is sum_c A_ac B_cb.
    # A_ac is |I_a| x |I_c|. B_cb is |I_c| x |I_b|.
    # So we are multiplying |I_a| x |I_c| by |I_c| x |I_b|.
    
    # We can apply Strassen if |I_a|=|I_c|=|I_b|=2.
    # This happens for (a,c,b) in {(1,1,1), (1,1,2), (1,2,1), (1,2,2), (2,1,1), (2,1,2), (2,2,1), (2,2,2)}.
    # These are the 2x2x2 sub-problems.
    # There are 8 such sub-problems?
    # Let's list them:
    # a=1, b=1: c=1 (2x2x2), c=2 (2x2x2), c=3 (2x1x2 - no)
    # a=1, b=2: c=1 (2x2x2), c=2 (2x2x2), c=3 (2x1x2 - no)
    # a=2, b=1: c=1 (2x2x2), c=2 (2x2x2), c=3 (2x1x2 - no)
    # a=2, b=2: c=1 (2x2x2), c=2 (2x2x2), c=3 (2x1x2 - no)
    # a=1, b=3: c=1 (2x2x1), c=2 (2x2x1), c=3 (2x1x1)
    # a=3, b=1: c=1 (1x2x2), c=2 (1x2x2), c=3 (1x1x2)
    # a=3, b=2: c=1 (1x2x2), c=2 (1x2x2), c=3 (1x1x2)
    # a=3, b=3: c=1 (1x2x1), c=2 (1x2x1), c=3 (1x1x1)
    
    # So we have 8 sub-problems of size 2x2x2.
    # Each can be done with Strassen (Rank 7) instead of Schoolbook (Rank 8).
    # Savings: 8 * (8-7) = 8.
    # Base Rank: 125.
    # New Rank: 125 - 8 = 117.
    
    # This matches the inspiration!
    # So we just need to implement this construction.
    
    # Construction:
    # We will build U, V, W by concatenating the factors for each sub-problem.
    # We need to map the local indices (0,1) to global indices.
    
    def get_strassen_factors(n_local):
        # Returns U, V, W for n_local x n_local multiplication using Strassen
        # n_local = 2
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # We need to express this as tensor decomposition.
        # U, V, W matrices of size (2*2, 7) = (4, 7).
        # U[i*2+j, k], V[j*2+k, k], W[k*2+i, k]
        
        U_loc = np.zeros((4, 7))
        V_loc = np.zeros((4, 7))
        W_loc = np.zeros((4, 7))
        
        # Indices for A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Indices for B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # Indices for C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        # Term 2: M2 = (A10+A11)B00
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1 # C21=M2+M4, C22=M1-M2...
        # Wait, let's check the W coefficients carefully.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-assign W based on this.
        # W[c_idx, term_idx]
        
        # Term 1 (M1):
        # A: A00, A11. B: B00, B11.
        # C: C00 (+1), C11 (-1).
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = -1
        
        # Term 2 (M2):
        # A: A10, A11. B: B00.
        # C: C10 (+1), C11 (-1).
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        # Term 3 (M3):
        # A: A00. B: B10, B11.
        # C: C01 (+1), C11 (+1).
        U_loc[0, 2] = 1
        V_loc[2, 2] = 1; V_loc[3, 2] = 1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        # Term 4 (M4):
        # A: A11. B: B01, B00.
        # C: C00 (+1), C10 (+1).
        U_loc[3, 3] = 1
        V_loc[1, 3] = 1; V_loc[0, 3] = -1 # B01 - B00
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        # Term 5 (M5):
        # A: A00, A01. B: B11.
        # C: C00 (-1), C01 (+1).
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        # Term 6 (M6):
        # A: A10, A00. B: B00, B01.
        # C: C11 (+1).
        U_loc[2, 5] = 1; U_loc[0, 5] = -1 # A10 - A00
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        # Term 7 (M7):
        # A: A01, A11. B: B10, B11.
        # C: C00 (+1).
        U_loc[1, 6] = 1; U_loc[3, 6] = -1 # A01 - A11
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        return U_loc, V_loc, W_loc

    # Get Strassen factors
    U_s, V_s, W_s = get_strassen_factors(2)
    
    # Define partitions
    # I1 = [0, 1], I2 = [2, 3], I3 = [4]
    partitions = [[0, 1], [2, 3], [4]]
    
    # Build global U, V, W
    # We will accumulate terms.
    # Each term is a column in U, V, W.
    # We can just concatenate the local factors, mapping indices.
    
    U_global = []
    V_global = []
    W_global = []
    
    # Helper to add a local block
    def add_block(a_idx, c_idx, b_idx, U_loc, V_loc, W_loc):
        # a_idx, c_idx, b_idx are indices into 'partitions'
        # I_a, I_c, I_b are the index sets
        I_a = partitions[a_idx]
        I_c = partitions[c_idx]
        I_b = partitions[b_idx]
        
        # U_loc is (|I_a|*|I_c|, R_loc)
        # V_loc is (|I_c|*|I_b|, R_loc)
        # W_loc is (|I_b|*|I_a|, R_loc)
        
        # We need to map local row indices to global.
        # Local U row (i, j) corresponds to global (I_a[i], I_c[j]).
        # Local V row (j, k) corresponds to global (I_c[j], I_b[k]).
        # Local W row (k, i) corresponds to global (I_b[k], I_a[i]).
        
        size_a = len(I_a)
        size_c = len(I_c)
        size_b = len(I_b)
        R_loc = U_loc.shape[1]
        
        U_block = np.zeros((n*m, R_loc))
        V_block = np.zeros((m*p, R_loc))
        W_block = np.zeros((p*n, R_loc))
        
        for r in range(size_a * size_c):
            i_loc = r // size_c
            j_loc = r % size_c
            i_glob = I_a[i_loc]
            j_glob = I_c[j_loc]
            U_block[i_glob * m + j_glob, :] = U_loc[r, :]
            
        for r in range(size_c * size_b):
            j_loc = r // size_b
            k_loc = r % size_b
            j_glob = I_c[j_loc]
            k_glob = I_b[k_loc]
            V_block[j_glob * p + k_glob, :] = V_loc[r, :]
            
        for r in range(size_b * size_a):
            k_loc = r // size_a
            i_loc = r % size_a
            k_glob = I_b[k_loc]
            i_glob = I_a[i_loc]
            W_block[k_glob * n + i_glob, :] = W_loc[r, :]
            
        U_global.append(U_block)
        V_global.append(V_block)
        W_global.append(W_block)

    # Iterate over all a, c, b
    for a in range(3):
        for c in range(3):
            for b in range(3):
                size_a = len(partitions[a])
                size_c = len(partitions[c])
                size_b = len(partitions[b])
                
                if size_a == 2 and size_c == 2 and size_b == 2:
                    # Use Strassen
                    add_block(a, c, b, U_s, V_s, W_s)
                else:
                    # Use Schoolbook
                    # Rank = size_a * size_c * size_b
                    R_sb = size_a * size_c * size_b
                    U_sb = np.zeros((size_a * size_c, R_sb))
                    V_sb = np.zeros((size_c * size_b, R_sb))
                    W_sb = np.zeros((size_b * size_a, R_sb))
                    
                    col = 0
                    for i in range(size_a):
                        for j in range(size_c):
                            for k in range(size_b):
                                U_sb[i * size_c + j, col] = 1
                                V_sb[j * size_b + k, col] = 1
                                W_sb[k * size_a + i, col] = 1
                                col += 1
                    add_block(a, c, b, U_sb, V_sb, W_sb)
                    
    # Concatenate
    U_final = np.hstack(U_global)
    V_final = np.hstack(V_global)
    W_final = np.hstack(W_global)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 140:
        # Fallback to schoolbook if something went wrong or took too long
        # (Unlikely, this is O(1))
        pass
        
    # Verify
    if not check_validity(U_final, V_final, W_final):
        # Fallback
        U_fb = np.zeros((n*m, n*m*p))
        V_fb = np.zeros((m*p, n*m*p))
        W_fb = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i*m*p + j*p + k
                    U_fb[i*m+j, c] = 1
                    V_fb[j*p+k, c] = 1
                    W_fb[k*n+i, c] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
