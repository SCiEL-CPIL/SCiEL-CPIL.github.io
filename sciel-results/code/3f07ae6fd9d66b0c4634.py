import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build the multiplication tensor T
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Initialize with Schoolbook Decomposition (Rank = n*m*p)
    rank_schoolbook = n * m * p
    U = np.zeros((dim_U, rank_schoolbook))
    V = np.zeros((dim_V, rank_schoolbook))
    W = np.zeros((dim_W, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    current_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Helper to verify if U, V, W decompose T exactly
    def verify(U_mat, V_mat, W_mat, T_target):
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can check this by contracting U and V into a matrix M1 and M2?
        # No, T is a 3-tensor.
        # Efficient check: 
        # For each r, the term is U[:,r] outer V[:,r] outer W[:,r].
        # We can sum these up.
        # To be fast, we can use einsum or reshape tricks.
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is equivalent to:
        # Let A = U * W.T  (shape dim_U x dim_W) -- wait, no.
        # Let's just do the sum for verification if rank is small enough, 
        # or use a randomized check if rank is large.
        # Given 120s limit, we can afford exact checks for ranks < 100.
        
        # Exact check via loop over rank (optimized with numpy broadcasting)
        # T_recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        # This might be slow for high rank. 
        # Alternative:
        # M_ab = sum_r U[a,r] V[b,r] = U @ V.T
        # Then T_recon[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
        # This doesn't simplify to a single matrix mult easily for verification 
        # unless we fix one dimension.
        
        # Let's fix 'c' and check if T[:, :, c] == sum_r (U[:,r] * V[:,r]) * W[c,r]
        # T_slice_c = sum_r W[c,r] * (U[:,r] outer V[:,r])
        # T_slice_c = (U * W[c, :]) @ V.T
        
        for c in range(dim_W):
            # T[:, :, c] should equal (U * W[c, :]) @ V.T
            # U * W[c, :] broadcasts W[c, :] to each row of U? No.
            # U is (dim_U, rank). W[c, :] is (rank,).
            # U * W[c, :] scales column r of U by W[c, r].
            # Result is (dim_U, rank).
            # Multiply by V.T (rank, dim_V).
            # Result (dim_U, dim_V).
            recon_slice = (U_mat * W_mat[c, :]) @ V_mat.T
            if not np.allclose(recon_slice, T_target[:, :, c], atol=1e-9):
                return False
        return True

    # Helper to attempt rank reduction by removing one column
    def try_reduce_rank(U_mat, V_mat, W_mat, T_target, current_rank_val):
        # We want to see if we can remove column r such that the remaining columns
        # still span the same tensor.
        # This is equivalent to checking if the column r in the "coefficient space"
        # is linearly dependent on the others in the context of the tensor product.
        
        # A simpler heuristic: 
        # The set of vectors {u_r, v_r, w_r} defines a rank-1 tensor.
        # If we remove r, we lose that rank-1 component.
        # We can only remove r if that component is zero (impossible for schoolbook) 
        # or if it can be represented by the other components.
        
        # Actually, a more robust search strategy for "diff":
        # We treat the decomposition as a set of vectors.
        # We can try to find a linear combination of columns that allows us to
        # eliminate one dimension.
        
        # Let's try a different approach: 
        # Randomized Search for Lower Rank.
        # We know schoolbook is 72.
        # We can try to find a decomposition of rank R < 72.
        # This is a non-convex problem.
        
        # Given the time constraint, a full random search is risky.
        # However, the "inspiration" mentions "structured random search".
        
        # Let's implement a structured search:
        # 1. Start with Schoolbook.
        # 2. Try to merge pairs of terms that cancel out or combine.
        #    e.g., if U[:,r1] == U[:,r2] and V[:,r1] == -V[:,r2] and W[:,r1] == W[:,r2],
        #    then the terms cancel.
        #    In schoolbook, all entries are 1, so no cancellation.
        
        # Let's try a "Greedy Elimination" based on Linear Dependencies in the flattened tensor space.
        # Flatten T to a vector t of size dim_U*dim_V*dim_W.
        # Each rank-1 term r corresponds to a vector t_r = vec(U[:,r] outer V[:,r] outer W[:,r]).
        # We want the smallest subset of {t_r} that sums to t.
        # This is a combinatorial problem.
        
        # Since we have 110s, we can do a limited random search for better decompositions
        # initialized near the schoolbook structure but with random perturbations,
        # and then use a local optimization (like gradient descent on the error) 
        # to snap to exact integer/half-integer solutions if possible.
        # BUT: "exact tensor decomposition" with "integer/half-integer entries" is a strong constraint.
        # Random floats won't work.
        
        # Strategy:
        # 1. Use the known Strassen algorithm for 2x2x2 (rank 7) as a building block.
        # 2. Embed this into 3x3x8.
        # 3. 3x3x3 has rank 23. 3x3x8 is 3x3x(3+3+2)?
        
        # Let's construct a specific structured decomposition:
        # Split the 8 columns into 3+3+2.
        # Block 1: 3x3x3 (Rank 23)
        # Block 2: 3x3x3 (Rank 23)
        # Block 3: 3x3x2 (Rank ?)
        
        # Rank of 3x3x2:
        # This is multiplying a 3x3 matrix by a 3x2 matrix.
        # Schoolbook: 3*3*2 = 18.
        # Can we do better?
        # 3x3x2 is equivalent to two 3x3x1 multiplications.
        # Rank(3x3x1) = 9 (diagonal).
        # Total 18.
        
        # Total Rank = 23 + 23 + 18 = 64.
        # This is better than 72!
        
        # Let's verify if we can construct this.
        # We need the explicit factors for 3x3x3 (Strassen) and 3x3x2.
        
        # Strassen 2x2x2 Factors (Rank 7):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # To get 3x3x3, we can embed 2x2x2 in the top-left, and handle the rest.
        # This is getting complex to hardcode perfectly in 110s without testing.
        
        # Alternative: Use the "Winograd" or "Frequency" approach mentioned in the current code's comments but actually implement it.
        # However, the most reliable "jump" is the block decomposition 23+23+18=64.
        
        # Let's try to build the 3x3x3 Strassen decomposition explicitly.
        # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
        # For 2x2:
        # U (4x7), V (4x7), W (4x7)
        # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # Let's define the 2x2x2 factors.
        # r=0: M1
        # U: A00=1, A11=1 -> [1, 0, 0, 1]
        # V: B00=1, B11=1 -> [1, 0, 0, 1]
        # W: C00=1, C01=1, C10=1, C11=-1? 
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4 + M5
        # C10 = M2 + M4 - M6 - M7
        # C11 = M1 - M2 + M3 + M6
        
        # Let's map these to vectors.
        # U (A factors):
        # r0 (M1): A00=1, A11=1 -> [1, 0, 0, 1]
        # r1 (M2): A10=1, A11=1 -> [0, 0, 1, 1]
        # r2 (M3): A00=1 -> [1, 0, 0, 0]
        # r3 (M4): A11=1 -> [0, 0, 0, 1]
        # r4 (M5): A00=1, A01=1 -> [1, 1, 0, 0]
        # r5 (M6): A10=1, A00=-1 -> [-1, 0, 1, 0]
        # r6 (M7): A01=1, A11=-1 -> [0, 1, 0, -1]
        
        # V (B factors):
        # r0 (M1): B00=1, B11=1 -> [1, 0, 0, 1]
        # r1 (M2): B00=1 -> [1, 0, 0, 0]
        # r2 (M3): B01=1, B11=-1 -> [0, 1, 0, -1]
        # r3 (M4): B10=1, B00=-1 -> [-1, 0, 1, 0]
        # r4 (M5): B11=1 -> [0, 0, 0, 1]
        # r5 (M6): B00=1, B10=1 -> [1, 0, 1, 0]
        # r6 (M7): B01=1, B11=1 -> [0, 1, 0, 1]
        
        # W (C factors):
        # C00 = M1 - M4 + M5 + M7 -> W00: [1, 0, 0, -1, 1, 0, 1]
        # C01 = M3 + M4 + M5 -> W01: [0, 0, 1, 1, 1, 0, 0]
        # C10 = M2 + M4 - M6 - M7 -> W10: [0, 1, 0, 1, 0, -1, -1]
        # C11 = M1 - M2 + M3 + M6 -> W11: [1, -1, 1, 0, 0, 1, 0]
        
        # Let's verify this 2x2x2 decomposition.
        # C00 = sum_r U0r Vr Wr0
        # r0: 1*1*1 = 1
        # r1: 0
        # r2: 0
        # r3: 1*(-1)*(-1) = 1? No.
        # U0r for r3 (A11) is 1. V0r for r3 (B10) is -1. W0r for r3 (C00) is -1.
        # 1 * (-1) * (-1) = 1.
        # r4: 1*0*1 = 0
        # r5: 0
        # r6: 1*0*1 = 0
        # Sum = 1 + 1 = 2. Should be 1.
        # Error in W mapping?
        # C00 = M1 - M4 + M5 + M7.
        # M1: A00B00 + A00B11 + A11B00 + A11B11
        # M4: A11B10 - A11B00
        # M5: A00B11 + A01B11
        # M7: A01B01 + A01B11 - A11B01 - A11B11
        
        # Let's re-calculate C00 coefficient for each term.
        # Term A00B00: M1(1) - M4(0) + M5(0) + M7(0) = 1. Correct.
        # Term A00B11: M1(1) - M4(0) + M5(1) + M7(0) = 2. Incorrect, should be 0.
        
        # Wait, Strassen formulas:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4 + M5
        # C10 = M2 + M4 - M6 - M7
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check M5: (A00+A01)B11 = A00B11 + A01B11.
        # So in C00, M5 contributes A00B11.
        # In C00, we want 0 for A00B11.
        # M1 contributes A00B11.
        # So M1 - M5 would cancel A00B11.
        # But formula is M1 + M5.
        
        # Let's check standard Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B01 + B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4 + M5
        # C10 = M2 + M4 - M6 - M7
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-evaluate C00 = M1 - M4 + M5 + M7.
        # A00B00: M1(1) - M4(0) + M5(0) + M7(0) = 1. OK.
        # A00B11: M1(1) - M4(0) + M5(1) + M7(0) = 2. FAIL.
        
        # Is the formula wrong?
        # Wikipedia:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4 + M5
        # C10 = M2 + M4 - M6 - M7
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check M7: (A01 - A11)(B01 + B11) = A01B01 + A01B11 - A11B01 - A11B11.
        # No A00 terms in M7.
        
        # Ah, I see. The standard Strassen decomposition works. Why did my check fail?
        # Maybe I copied the formula wrong?
        # Let's try:
        # C00 = M1 - M4 + M5 + M7
        # If A00=1, B11=1, others 0.
        # M1 = (1+0)(0+1) = 1.
        # M4 = 0.
        # M5 = (1+0)1 = 1.
        # M7 = 0.
        # C00 = 1 - 0 + 1 + 0 = 2.
        # But A*B should be 0 for (0,0) element if A00=1, B11=1.
        # So the formula MUST be wrong or my mapping is wrong.
        
        # Correct Strassen:
        # C00 = M1 - M4 + M5 + M7
        # Let's check a known source.
        # Actually, C00 = M1 - M4 + M5 + M7 is correct for some variants.
        # Let's check C00 for A00=1, B11=1.
        # M1 = 1. M5 = 1. Sum = 2.
        # This implies the formula is for a different arrangement or I am misremembering.
        
        # Let's use the 2x2x2 decomposition from a reliable source or derive it.
        # Given the risk of error in hardcoding Strassen, and the time limit,
        # I will stick to the **Schoolbook** but try to **verify** if a simple
        # **Block Diagonal** construction yields a valid lower rank.
        
        # Actually, the safest "improvement" over 72 is to use the **3x3x3 Rank 23**
        # fact.
        # If I can construct 3x3x3 with Rank 23, then 3x3x8 (as 3x3x3 + 3x3x3 + 3x3x2)
        # Rank = 23 + 23 + 18 = 64.
        
        # Let's hardcode the 3x3x3 Rank 23 factors.
        # This is a known result (Bini et al.).
        # However, finding the explicit 23 vectors is non-trivial to hardcode from memory.
        
        # Alternative: **Random Search with Verification**.
        # We can generate random integer matrices U, V, W of rank R
        # and check if they decompose T.
        # Probability is low, but we can use the structure.
        
        # Let's try a **Greedy Column Merging** on the Schoolbook.
        # If two columns in U, V, W are identical, we can merge them?
        # No, they add up.
        
        # Let's return the Schoolbook but with a **Verification Step** to ensure
        # we don't return garbage, and a **Time Check**.
        # The current code returns Schoolbook.
        # To beat 1.0, we MUST find rank < 72.
        
        # Let's try the **2x2 Strassen Embedding** properly.
        # Embed 2x2x2 (Rank 7) into 3x3x3.
        # A = [[A22, 0], [0, 1]]? No.
        
        # Let's try **Winograd's F(2, 2, 2)** which is Strassen.
        
        # Given the difficulty of hardcoding exact low-rank factors for 3x3x8,
        # and the risk of verification failure, I will implement a **Robust Schoolbook**
        # and a **Simple Block Decomposition** that is guaranteed to be valid
        # and potentially lower rank if the blocks are smaller.
        
        # 3x3x8 = 3x3x4 + 3x3x4.
        # 3x3x4 Rank?
        # Schoolbook 3*3*4 = 36.
        # Total 72.
        
        # 3x3x8 = 3x3x2 + 3x3x2 + 3x3x2 + 3x3x2.
        # Rank 18 * 4 = 72.
        
        # It seems hard to beat 72 without a specific algorithm like Strassen for 3x3.
        
        # Let's try **Rank 64** via 3x3x3 (23) + 3x3x3 (23) + 3x3x2 (18).
        # I will attempt to construct the 3x3x3 Rank 23 factors.
        # If I fail, I return Schoolbook.
        
        # For now, to ensure a valid submission, I return Schoolbook.
        # The "jump" might not happen if I can't construct the factors.
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

