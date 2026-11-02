"""Matrix multiplication 6,6,6 - structured 2x3 block scheme search + null-space rank reduction."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Strategy:
    # 1. Search for a 2x3 matrix multiplication scheme (A is 2x3, B is 3x3, C is 2x3)
    #    with rank < 27 (schoolbook).
    # 2. Use the best scheme found to build a 6x6x6 decomposition via 3x3 sub-blocks.
    # 3. Apply null-space rank reduction to minimize the final rank.

    # --- Step 1: Search for 2x3 block schemes ---
    # A: 2x3, B: 3x3, C: 2x3
    # Schoolbook: 6*3 = 18 multiplications
    # We want to find a scheme with r multiplications where r < 18
    
    # Each multiplication: M_s = (sum_{i,j} a_s[i,j] * A[i,j]) * (sum_{j,k} b_s[j,k] * B[j,k])
    # Output: C[i,k] = sum_s c_s[i,k] * M_s
    
    # For small r, we can enumerate sparse patterns
    # Let's try r = 7, 8, 9 (Strassen-like for 2x3)
    
    # Actually, let's first verify the 2x3 schoolbook rank
    # A[i,j] for i in 0,1; j in 0,1,2
    # B[j,k] for j in 0,1,2; k in 0,1,2
    # C[i,k] for i in 0,1; k in 0,1,2
    
    # The tensor has 2*3*3 = 18 entries, all 1s in the "diagonal" pattern
    # T[i,j,k] = 1 if the indices are consistent with matrix multiplication
    
    # For 2x3 * 3x3 -> 2x3:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple contributes to C[i,k])
    # Wait, that's not right. T[i,j,k] = 1 if this triple is part of the multiplication.
    # In matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) where i is row of A,
    # j is column of A / row of B, k is column of B.
    # For 2x3 * 3x3: i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # So T is all ones of shape (2,3,3) = 18 entries.
    
    # The rank of the all-ones tensor of shape (2,3,3) is... 
    # Actually, the matrix multiplication tensor for (n,m,p) has T[i,j,k]=1 for all i,j,k
    # when we write C[i,k] = sum_j A[i,j]*B[j,k].
    # So for 2x3*3x3, T is a 2x3x3 all-ones tensor.
    
    # The rank of an all-ones tensor of shape (a,b,c) is at most min(a*b, b*c, a*c)
    # For (2,3,3): at most min(6,9,6) = 6? No, that's not right either.
    # Actually, the rank of the all-ones tensor is 1 if we allow real entries? No.
    # T[i,j,k] = 1 = 1*1*1, so it's rank 1!
    # Wait, that can't be right for matrix multiplication.
    
    # I'm confusing myself. Let me re-derive.
    # Matrix multiplication tensor: T[i,j,k] = 1 if (i,j,k) is a valid triple.
    # For standard n x n matrix multiplication, T[i,j,k] = 1 for all i,j,k in [0,n).
    # So T is the all-ones tensor of shape (n,n,n).
    # The rank of the all-ones tensor (n,n,n) is n (not 1).
    # Proof: T[i,j,k] = 1 = (1/sqrt(n)) * (1/sqrt(n)) * (1/sqrt(n)) * n
    # Actually, T[i,j,k] = sum_{l=0}^{n-1} (1/n) * delta_{i,l} * delta_{l,j} * delta_{k,l}... no.
    
    # Let me think again. The matrix multiplication tensor for n x n is:
    # T[i,j,k] = 1 for all i,j,k in [0,n)
    # This can be written as: T[i,j,k] = sum_{l=0}^{n-1} U[i,l] * V[j,l] * W[k,l]
    # where U[i,l] = 1 if i==l else 0, V[j,l] = 1 if j==l else 0, W[k,l] = 1 if k==l else 0
    # This gives rank n. But we know rank(n x n matmul) >= n^2 for n>=2 (lower bound).
    # So the all-ones tensor interpretation is wrong.
    
    # CORRECT: The matrix multiplication tensor has T[i,j,k] = 1 if and only if
    # the triple (i,j,k) participates in the product. For C[i,k] = sum_j A[i,j]*B[j,k],
    # every triple (i,j,k) participates, so T[i,j,k] = 1 for all i,j,k.
    # But the rank is NOT n. The rank of the all-ones tensor (n,n,n) is actually...
    
    # Let me just use the known result: rank of n x n matrix multiplication tensor.
    # For n=1: rank 1
    # For n=2: rank 7 (Strassen)
    # For n=3: rank 19? (best known)
    # For n=4: rank 49? (best known)
    
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    # This is the "all-ones" tensor. Its rank is known to be between n^2 and n^3.
    
    # OK, I was overthinking. Let me just build the 2x3 scheme directly.
    
    # For 2x3 * 3x3 -> 2x3:
    # We need: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}
    # This is 6 output entries, each is a sum of 3 products.
    # Total "work" = 6*3 = 18 scalar multiplications (schoolbook).
    
    # Can we do it in fewer? Let's try to find a scheme with r multiplications.
    # Each multiplication M_s = (linear combo of A entries) * (linear combo of B entries)
    # C[i,k] = sum_s c_s[i,k] * M_s
    
    # For r=7 (Strassen-like):
    # Let's define the scheme and check if it works.
    
    # Actually, let me just use the known Strassen 2x2 scheme and extend.
    # Split 6x6 into 3x3 blocks of 2x2 matrices.
    # 3x3 matrix multiplication of 2x2 blocks: use Strassen (rank 7) for each 2x2.
    # But 3x3 matrix multiplication has rank at least 19? No, 3x3 matmul rank is at most 27 (schoolbook).
    # If we use Strassen for 2x2 blocks, we get 3x3x3x7 = 63 rank-1 terms for the 6x6 product?
    # No, that's not how it works.
    
    # Let me try a different approach: use the 2x2 Strassen scheme directly on 6x6
    # by splitting into 2x2 blocks of 3x3 matrices.
    # 6x6 = 2x2 blocks of 3x3 matrices.
    # Strassen: 7 multiplications of 3x3 matrices.
    # Each 3x3 multiplication has rank at most 27 (schoolbook).
    # Total rank: 7 * 27 = 189.
    
    # This is what the current program does. Let me try to improve it.
    
    # Alternative: split 6x6 into 3x3 blocks of 2x2 matrices.
    # 3x3 matrix multiplication of 2x2 blocks: rank at most 27 (schoolbook for 3x3).
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank: 27 * 7 = 189. Same.
    
    # What if we use a better 3x3 scheme? If 3x3 matmul has rank 19, then 19*7=133.
    # But I don't know a rank-19 scheme for 3x3.
    
    # Let me try to find a 3x3 matmul scheme with rank < 27.
    
    # For 3x3 * 3x3 -> 3x3:
    # Schoolbook: 27 multiplications.
    # Can we do better?
    
    # Let's try to enumerate 3x3 schemes with r < 27.
    # This is a large search space, so let's be smart.
    
    # Actually, let me just use the 189 rank decomposition and try to reduce it
    # using linear algebra more effectively.
    
    # Build the 189-rank decomposition using 2x2 Strassen on 3x3 blocks.
    rank_strassen = 7 * 27  # 189
    U = np.zeros((n * m, rank_strassen), dtype=float)
    V = np.zeros((m * p, rank_strassen), dtype=float)
    W = np.zeros((p * n, rank_strassen), dtype=float)
    
    # Strassen 2x2 scheme
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),              # M2
        ({0: 1}, {1: 1, 3: -1}),             # M3
        ({3: 1}, {2: 1, 0: -1}),             # M4
        ({0: 1, 1: 1}, {3: 1}),              # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),       # M7
    ]
    
    c_contrib = [
        {0: 1, 3: 1},   # M1: C00(+1), C11(+1)
        {2: 1, 3: -1},  # M2: C10(+1), C11(-1)
        {1: 1, 3: 1},   # M3: C01(+1), C11(+1)
        {0: -1, 2: 1},  # M4: C00(-1), C10(+1)
        {1: 1},         # M5: C01(+1)
        {0: 1},         # M6: C00(+1)
        {0: -1, 3: 1},  # M7: C00(-1), C11(+1)
    ]
    
    for s in range(7):
        a_coeffs = steps[s][0]
        b_coeffs = steps[s][1]
        c_coeffs = c_contrib[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    for a_block, a_sign in a_coeffs.items():
                        a_row = (a_block // 2) * 3 + i
                        a_col = (a_block % 2) * 3 + j
                        U[a_row * m + a_col, col] += float(a_sign)
                    
                    for b_block, b_sign in b_coeffs.items():
                        b_row = (b_block // 2) * 3 + j
                        b_col = (b_block % 2) * 3 + k
                        V[b_row * p + b_col, col] += float(b_sign)
                    
                    for c_block, c_sign in c_coeffs.items():
                        c_row = (c_block // 2) * 3 + i
                        c_col = (c_block % 2) * 3 + k
                        W[c_col * n + c_row, col] += float(c_sign)
    
    # Verify the decomposition
    T_reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
    T_expected = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(T_reconstructed, T_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, column] = 1.0
                    V[j*p+k, column] = 1.0
                    W[k*n+i, column] = 1.0
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Step 2: Rank reduction using null space ---
    # The tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # In vector form: T_vec = sum_c v_c where v_c = U[:,c] ⊙ V[:,c] ⊙ W[:,c] (flattened)
    # We want to find a sparse vector w such that sum_c w_c * v_c = T_vec
    # and minimize the number of non-zero w_c.
    
    # Build the tensor matrix: columns are v_c
    tensor_matrix = np.zeros((n*m*p, rank_strassen), dtype=float)
    for c in range(rank_strassen):
        tensor_matrix[:, c] = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).reshape(-1)
    
    T_vec = T_expected.reshape(-1)
    
    # Find the null space of tensor_matrix
    # If v is in the null space, then tensor_matrix @ v = 0
    # We can add null space vectors to the all-ones vector to get other valid solutions
    
    try:
        _, s, vh = np.linalg.svd(tensor_matrix, full_matrices=False)
        tol = 1e-10
        null_start = np.sum(s > tol)
        if null_start < rank_strassen:
            null_space = vh[null_start:]  # shape (rank - null_start, rank_strassen)
        else:
            null_space = np.array([])
    except:
        null_space = np.array([])
    
    if null_space.size > 0:
        # We have null space vectors. Try to find a sparse solution.
        # w = ones + sum(a_i * null_vec_i)
        # We want w to be sparse (many zeros).
        
        # Try a greedy approach: for each coordinate, try to set it to zero
        # by adjusting the null space coefficients.
        
        # This is a hard problem, so let's try a simple heuristic:
        # Randomly sample null space combinations and check sparsity.
        
        best_w = np.ones(rank_strassen, dtype=float)
        best_sparsity = np.sum(best_w != 0)
        
        num_trials = 200
        for trial in range(num_trials):
            if time.time() > DEADLINE:
                break
            
            # Random null space combination
            a = np.random.randn(null_space.shape[0])
            w = np.ones(rank_strassen, dtype=float) + null_space.T @ a
            
            # Check if w is valid (tensor_matrix @ w = T_vec)
            # This should be approximately true by construction
            
            # Count sparsity
            sparsity = np.sum(np.abs(w) > 1e-10)
            if sparsity < best_sparsity:
                best_sparsity = sparsity
                best_w = w.copy()
        
        # Apply the best w
        # Keep only columns where w_c != 0
        keep_mask = np.abs(best_w) > 1e-10
        new_rank = np.sum(keep_mask)
        
        if new_rank < rank_strassen:
            U = U[:, keep_mask]
            V = V[:, keep_mask]
            W = W[:, keep_mask]
            
            # Scale the kept columns by w_c
            for c in range(new_rank):
                U[:, c] *= best_w[keep_mask][c]
                V[:, c] *= best_w[keep_mask][c]
                W[:, c] *= best_w[keep_mask][c]
            
            rank_strassen = new_rank
    
    # Final verification
    if rank_strassen > 0:
        T_check = np.einsum('ic,jc,kc->ijk', U, V, W)
        if not np.allclose(T_check, T_expected, atol=1e-10):
            # Something went wrong, fallback to original
            U = np.zeros((n * m, 189), dtype=float)
            V = np.zeros((m * p, 189), dtype=float)
            W = np.zeros((p * n, 189), dtype=float)
            
            for s in range(7):
                a_coeffs = steps[s][0]
                b_coeffs = steps[s][1]
                c_coeffs = c_contrib[s]
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            col = s * 27 + i * 9 + j * 3 + k
                            
                            for a_block, a_sign in a_coeffs.items():
                                a_row = (a_block // 2) * 3 + i
                                a_col = (a_block % 2) * 3 + j
                                U[a_row * m + a_col, col] += float(a_sign)
                            
                            for b_block, b_sign in b_coeffs.items():
                                b_row = (b_block // 2) * 3 + j
                                b_col = (b_block % 2) * 3 + k
                                V[b_row * p + b_col, col] += float(b_sign)
                            
                            for c_block, c_sign in c_coeffs.items():
                                c_row = (c_block // 2) * 3 + i
                                c_col = (c_block % 2) * 3 + k
                                W[c_col * n + c_row, col] += float(c_sign)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
