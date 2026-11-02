"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Schoolbook rank is n*m*p = 60
    # We will construct a rank 20 decomposition using known low-rank algorithms
    # for smaller sub-problems.
    
    # Strategy:
    # 1. Split the 3x10 matrix B into B1 (3x2) and B2 (3x8).
    # 2. Compute A (2x3) * B1 (3x2) -> C1 (2x2). Use Strassen-like rank 5 algorithm.
    # 3. Compute A (2x3) * B2 (3x8) -> C2 (2x8). Use rank 12 algorithm (based on 2x4 rank 6).
    # 4. Total rank = 5 + 12 = 17? Or can we share some factors?
    #    Actually, let's look at the structure.
    #    A is 2x3.
    #    For 2x2 * 2x2, rank 5 is known (Strassen).
    #    For 2x4 * 4x4? No, A is 2x3.
    
    # Let's refine the construction.
    # We want to compute C = A B where A is 2x3, B is 3x10.
    # Let's split B into two 3x5 blocks? 
    # Or split the columns of B into 2+8?
    # 
    # Known results:
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,3) = 10? (Actually R(2,2,3) is not 10, let's check. 
    #   R(2,2,2)=7. R(2,2,3) >= 10? No, R(2,2,3) is likely 10 or 11.
    #   Wait, the tensor for 2x2*2x3 has size 2*2*3=12.
    #   Lower bound is max(2*2, 2*3, 2*3) = 6.
    #   Upper bound: R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 2 = 9?
    #   Let's verify R(2,2,1). A(2x2)*B(2x1) -> C(2x1).
    #   c0 = a00 b0 + a01 b1
    #   c1 = a10 b0 + a11 b1
    #   Rank 4 is schoolbook. Can we do better?
    #   This is essentially 2 linear forms. Rank 4 is minimal? 
    #   Actually, R(2,2,1) = 4 is not minimal. 
    #   Let's check R(2,1,1) = 2.
    #   R(2,2,1): C = A * b. 
    #   This is a linear map from A to C for fixed b? No, bilinear.
    #   It's equivalent to multiplying a 2x2 matrix by a vector.
    #   Rank is 4? 
    #   Let's look at R(2,2,3).
    #   Actually, there is a known algorithm for 2x2x3 of rank 10?
    #   Or maybe we should use a different split.
    
    # Let's try a direct construction for 2x3x10.
    # We can view the 3x10 part as three rows.
    # 
    # Alternative: Use the fact that R(2,3,3) is known.
    # R(2,3,3) <= R(2,2,3) + R(2,1,3)?
    # 
    # Let's look at the "Sum of Products" structure.
    # 
    # A more robust approach:
    # Use the known rank 7 algorithm for 2x2x2 (Strassen).
    # We can embed 2x3x10 into a larger problem or use blocking.
    # 
    # Let's try to construct a rank 20 solution explicitly.
    # 
    # Consider splitting the 3 middle dimension into 2+1.
    # A = [A2 | A1] where A2 is 2x2, A1 is 2x1.
    # B = [B2; B1] where B2 is 2x10, B1 is 1x10.
    # C = A2 B2 + A1 B1.
    # 
    # Term 1: A2 (2x2) * B2 (2x10).
    # This is 2x2 * 2x10.
    # We can split B2 into five 2x2 blocks: B2 = [B2_0, B2_1, B2_2, B2_3, B2_4].
    # C1 = A2 B2_0 + A2 B2_1 + ... + A2 B2_4.
    # Each A2 B2_i is 2x2 * 2x2. Rank 5 (Strassen).
    # Total rank for this part = 5 * 5 = 25.
    # 
    # Term 2: A1 (2x1) * B1 (1x10).
    # This is outer product of a 2-vector and a 10-vector.
    # C2 = A1 B1.
    # c0 = a0 b0, c1 = a1 b0, ..., c9 = a1 b9? No.
    # A1 is [a0; a1]. B1 is [b0, b1, ..., b9].
    # C2 is 2x10.
    # C2[0, k] = a0 * B1[k]
    # C2[1, k] = a1 * B1[k]
    # This requires 2 * 10 = 20 multiplications?
    # Or can we do better?
    # The tensor for 2x1 * 1x10 is rank 20?
    # Let's check.
    # We need to compute a0*b_k and a1*b_k for all k.
    # These are all independent bilinear forms?
    # Yes, the partial derivatives are independent.
    # So rank is 20.
    # Total rank = 25 + 20 = 45.
    # This is better than 60, but not minimal.
    
    # Let's try splitting the 3 middle dimension into 1+1+1?
    # A = [A1 A2 A3] (2x1 each)
    # B = [B1; B2; B3] (1x10 each)
    # C = A1 B1 + A2 B2 + A3 B3.
    # Each term is 2x1 * 1x10. Rank 20.
    # Total rank 60. No gain.
    
    # Let's try splitting the output columns.
    # 10 columns.
    # 
    # What if we use the rank 7 algorithm for 2x2x2 and apply it to 2x2x2 blocks?
    # 
    # Let's look at R(2,3,3).
    # If we can find a good algorithm for 2x3x3, we can extend it to 2x3x10.
    # 
    # Known: R(2,3,3) = 12?
    # Let's assume R(2,3,3) = 12.
    # Then for 2x3x10, we can split the 10 columns into three 3x3 blocks and one 3x1 block?
    # 10 = 3+3+3+1.
    # R(2,3,3) + R(2,3,3) + R(2,3,3) + R(2,3,1).
    # R(2,3,1): 2x3 * 3x1.
    # c0 = a00 b0 + a01 b1 + a02 b2
    # c1 = a10 b0 + a11 b1 + a12 b2
    # This is 2 linear forms of 3 variables.
    # Rank?
    # It's equivalent to multiplying a 2x3 matrix by a 3x1 vector.
    # This is a "dot product" for each row.
    # Rank 6 is schoolbook.
    # Can we do better?
    # The tensor is of size 2*3*1 = 6.
    # The partial derivatives are independent?
    # Yes, so rank is 6.
    # Total rank = 12 + 12 + 12 + 6 = 42.
    # Still not great.
    
    # Let's try to find a better algorithm for 2x3x3.
    # 
    # Actually, there is a known result: R(2,3,3) = 12 is not the best.
    # 
    # Let's try a different strategy: **Randomized Search with Exact Rounding**.
    # The previous program failed because random search doesn't converge to exact integers.
    # We can use a **Lattice Reduction** or **Integer Programming** approach?
    # No, too complex for 120s.
    
    # Let's try **Greedy Construction with Local Search**.
    # Start with rank 20.
    # Initialize U, V, W with random integers from {-1, 0, 1}.
    # Check if they satisfy the equations.
    # If not, fix one variable at a time to minimize error.
    
    # Actually, let's use the **Strassen Algorithm** structure directly.
    # 
    # We can compute 2x3 * 3x10 by viewing it as:
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x10) = [B_top (2x10); B_bottom (1x10)]
    # 
    # C = A_left B_top + A_right B_bottom.
    # 
    # Part 1: A_left (2x2) * B_top (2x10).
    # Split B_top into five 2x2 matrices: B_top = [B1, B2, B3, B4, B5].
    # C1 = A_left B1 + A_left B2 + A_left B3 + A_left B4 + A_left B5.
    # Each A_left Bi is 2x2 * 2x2.
    # Use Strassen (rank 7) for each?
    # Rank 7 * 5 = 35.
    # 
    # Part 2: A_right (2x1) * B_bottom (1x10).
    # Rank 20 (as derived before).
    # 
    # Total Rank = 35 + 20 = 55.
    # This is better than 60, but not minimal.
    
    # Can we share Strassen factors?
    # The Strassen algorithm for 2x2x2 uses 7 products.
    # If we apply it to 5 different B matrices, we use 35 products.
    # 
    # What if we use a different split for B_top?
    # 
    # Let's try to compute 2x2 * 2x10 directly.
    # A (2x2) * B (2x10).
    # This is equivalent to 5 separate 2x2 * 2x2 multiplications.
    # Is there a faster way?
    # The tensor size is 2*2*10 = 40.
    # Lower bound: max(4, 20, 20) = 20.
    # So rank is at least 20.
    # Can we achieve rank 20?
    # If we use schoolbook for each 2x2 block, we get 5 * 4 = 20.
    # So R(2,2,10) = 20?
    # Let's verify.
    # A = [[a, b], [c, d]]
    # B = [[e, f, g, h, i, j, k, l, m, n],
    #      [o, p, q, r, s, t, u, v, w, x]]
    # C = [[a e + b o, a f + b p, ...],
    #      [c e + d o, c f + d p, ...]]
    # 
    # Each column of C is a 2x1 vector.
    # C_col = A * B_col.
    # This is a 2x2 * 2x1 multiplication.
    # Rank of 2x2 * 2x1 is 4.
    # So for 10 columns, rank is 10 * 4 = 40?
    # Wait, my previous calculation was wrong.
    # R(2,2,1) = 4.
    # So R(2,2,10) = 10 * R(2,2,1) = 40.
    # 
    # So Part 1 rank is 40.
    # Part 2 rank is 20.
    # Total rank = 60.
    # No improvement.
    
    # We need a better algorithm for 2x2 * 2x1.
    # Is R(2,2,1) = 4 minimal?
    # c0 = a e + b o
    # c1 = c e + d o
    # 
    # Let's check if rank 3 is possible.
    # If rank 3, we need 3 products.
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
    # This is a linear map from the 4-dimensional space of A to the 2-dimensional space of C,
    # parameterized by the 2-dimensional space of B.
    # 
    # Actually, R(2,2,1) is the rank of the matrix multiplication tensor for 2x2 * 2x1.
    # This is the same as the rank of the tensor for 1x2 * 2x1? No.
    # 
    # Let's look up R(2,2,1).
    # It is known that R(2,2,1) = 4.
    # 
    # So we are stuck with 60 using simple blocking.
    
    # We need a more clever construction.
    # 
    # Let's try **Rank 20** via **Fourier Transform** or **Cyclic Group** methods?
    # 
    # Or, let's use the **Strassen Algorithm** for 2x2x2 and embed it differently.
    # 
    # What if we split the 3x10 matrix into 3x2 and 3x8?
    # A (2x3) * B (3x10).
    # 
    # Let's try to find a rank 20 solution by **Random Search with Integer Constraints**.
    # 
    # We will perform a random search over the set of matrices with entries in {-1, 0, 1} 
    # (or {-2, -1, 0, 1, 2}) and check for exactness.
    # 
    # Since the space is discrete, we can use a **Local Search** heuristic:
    # 1. Start with a random integer matrix of rank 20.
    # 2. Compute the error.
    # 3. If error > 0, pick a random entry and change it to a nearby integer.
    # 4. If error decreases, keep the change.
    # 5. Repeat until error is 0 or time limit.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(rank):
                            actual += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        if abs(actual - expected) > 1e-9:
                            return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using integer local search
    # Start with rank 20 (theoretical lower bound is max(6, 20, 20) = 20)
    
    def try_integer_local_search(r, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using integer local search."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random integers from {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n * m, r))
        V = np.random.randint(-1, 2, size=(m * p, r))
        W = np.random.randint(-1, 2, size=(p * n, r))
        
        # Scale? No, integers are integers.
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            expected = 1.0 if j1 == j2 else 0.0
                            actual = 0.0
                            for c in range(r):
                                actual += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                            err += (actual - expected) ** 2
            return err
        
        current_err = compute_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            if current_err == 0:
                break
            
            # Pick a random matrix and a random entry
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                row, col = np.random.randint(0, n * m), np.random.randint(0, r)
                old_val = U[row, col]
                # Try nearby integers
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    U[row, col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    else:
                        U[row, col] = old_val
            elif mat_choice == 1:
                row, col = np.random.randint(0, m * p), np.random.randint(0, r)
                old_val = V[row, col]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    V[row, col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    else:
                        V[row, col] = old_val
            else:
                row, col = np.random.randint(0, p * n), np.random.randint(0, r)
                old_val = W[row, col]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    W[row, col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                        break
                    else:
                        W[row, col] = old_val
            
            # Occasionally, try a larger jump
            if it % 100 == 0:
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    row, col = np.random.randint(0, n * m), np.random.randint(0, r)
                    U[row, col] = np.random.randint(-3, 4)
                elif mat_choice == 1:
                    row, col = np.random.randint(0, m * p), np.random.randint(0, r)
                    V[row, col] = np.random.randint(-3, 4)
                else:
                    row, col = np.random.randint(0, p * n), np.random.randint(0, r)
                    W[row, col] = np.random.randint(-3, 4)
        
        if current_err == 0:
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        return None
    
    # Try ranks from 20 to 30
    for r in range(20, 31):
        if time.time() > deadline - 20:
            break
        
        result = try_integer_local_search(r, max_iters=2000, seed=r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
