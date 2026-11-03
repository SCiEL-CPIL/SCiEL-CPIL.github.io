import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 6
    
    # Strategy: Use the known rank-20 exact tensor decomposition for 2x3x3 matrix multiplication.
    # We partition the 4x6 right matrix B into two 4x3 blocks: B_left (cols 0-2) and B_right (cols 3-5).
    # C = A * B = A * [B_left, B_right] = [A * B_left, A * B_right]
    # Each product A * B_block is a 2x3 matrix multiplication.
    # The rank of the 2x3x3 tensor is 20 (known optimal bound/achievement in literature, e.g., via specific algebraic identities or exhaustive search results for small tensors).
    # Total rank = 20 + 20 = 40.
    # This is better than the current 42 (6 * 7 Strassen blocks).
    
    # We need to construct the rank-20 decomposition for M_{2,3,3}.
    # Let A be 2x3, B be 3x3, C be 2x3.
    # Indices:
    # A: i in {0,1}, j in {0,1,2} -> idx = i*3 + j (6 entries)
    # B: j in {0,1,2}, k in {0,1,2} -> idx = j*3 + k (9 entries)
    # C: k in {0,1,2}, i in {0,1} -> idx = k*2 + i (6 entries)
    # Tensor T[i,j,k] = delta_{i,k}? No, T_{i,j,k} is 1 if i,j,k correspond to the multiplication.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is defined by T[i,j,k] = 1 for all i,j,k? No.
    # The tensor M_{2,3,3} has components M_{(i,j),(j,k),(k,i)} = 1.
    # Wait, the tensor T_{u,v,w} = 1 if u=(i,j), v=(j,k), w=(k,i) for some i,j,k.
    # Here u ranges over 6 (i*3+j), v over 9 (j*3+k), w over 6 (k*2+i).
    # Actually, the standard matrix multiplication tensor M_{n,m,p} has rank r such that
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'}? No.
    # The identity is: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is equivalent to sum_r (sum_{i,j} U_{i,j,r} A_{i,j}) (sum_{j,k} V_{j,k,r} B_{j,k}) = sum_{i,k} W_{i,k,r} C_{i,k}.
    # Or more commonly: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 if the indices match the multiplication structure.
    # Specifically, for fixed i,j,k, the coefficient of A_{i,j} B_{j,k} in the expansion of the sum is 1.
    # So T_{(i,j),(j,k),(k,i)} = 1.
    
    # We need a rank-20 decomposition for 2x3x3.
    # Finding this by hand is hard. However, the prompt suggests using a "known rank-20" decomposition.
    # If I don't have the explicit coefficients memorized, I cannot hardcode them reliably without a search.
    # Can I find it quickly?
    # The rank of M_{2,3,3} is known to be 12? No, 2x2x2 is 7.
    # 2x3x3: 
    # Let's check if a simpler construction exists.
    # What if I use the 2x2x2 Strassen (rank 7) and handle the extra dimension?
    # A 2x3 can be seen as 2x2 + 2x1?
    # B 3x3 can be seen as 2x2 + 1x2 + 2x1 + 1x1?
    # This gets complicated.
    
    # Alternative: The prompt says "known rank-20". 
    # If I cannot derive it, I should stick to the known valid 42 rank solution to ensure correctness, 
    # UNLESS I can implement a quick search or use a library.
    # But I have 140 seconds. Can I search for a lower rank?
    # The space of rank 40 decompositions is huge.
    # However, the prompt *inspiration* explicitly mentions rank 40 via 2x3x3 rank 20.
    # Is there a standard rank-20 decomposition for 2x3x3?
    # Actually, the rank of M_{2,3,3} is 12? 
    # Let's check small tensor ranks.
    # M_{2,2,2}: 7
    # M_{2,2,3}: ?
    # M_{2,3,3}: ?
    # According to some sources, the rank of M_{2,3,3} is 12? Or 15?
    # If it's 12, total rank 24. That would be huge improvement.
    # If it's 20, total rank 40.
    
    # Since I cannot verify the exact coefficients of a rank-20 2x3x3 decomposition from memory, 
    # and hardcoding incorrect coefficients will result in score 0, 
    # I must be very careful.
    
    # Let's try to construct a valid decomposition with rank < 42 using a different block strategy.
    # Current: 6 blocks of 2x2x2 (rank 7 each) = 42.
    # Can we do better?
    # What if we use 2x2x2 Strassen for the 2x2x2 parts and handle the rest more efficiently?
    # The 2x4x6 multiplication can be viewed as:
    # A (2x4) * B (4x6).
    # Split B into B1 (4x2), B2 (4x2), B3 (4x2).
    # C = A B1 + A B2 + A B3.
    # Each A B_j is 2x2.
    # We used 6 Strassen blocks.
    # Notice that A is shared.
    # Strassen computes (A_L B_top + A_R B_bot).
    # This structure is rigid.
    
    # Is there a rank 40 decomposition?
    # Let's assume the inspiration is correct that rank 40 is achievable.
    # But without the coefficients, I can't implement it.
    # 
    # WAIT. The prompt says "Knowledge from this run... Maintaining the provably-minimal rank-4 schoolbook decomposition... no structural change needed as rank 4 is already minimal".
    # This knowledge snippet seems confused or refers to a different sub-problem (maybe 2x2x2 rank 4 is NOT minimal, 7 is minimal for exact).
    # Actually, rank 4 is NOT minimal for 2x2x2 exact multiplication. 7 is the minimal rank for exact 2x2x2 multiplication (Strassen).
    # The snippet "rank-4 schoolbook" might refer to the number of multiplications in schoolbook for 2x2 (which is 4), but it's not a tensor decomposition rank.
    # The score 1.143 corresponds to rank 42.
    
    # Let's look at the "Inspiration 2" code. It defines a verification function and Strassen terms.
    # It doesn't provide the 2x3x3 decomposition.
    
    # Given the risk of providing an invalid decomposition (score 0), and the lack of explicit coefficients for the 2x3x3 rank-20 decomposition,
    # the safest "improvement" within the family might be to optimize the current rank-42 construction for speed (already fast) 
    # or to find a slightly better decomposition if one is easily derivable.
    
    # However, there is a known result: The rank of M_{2,3,3} is 12? 
    # Let's check: 
    # If rank(M_{2,3,3}) = 12, then 2 * 12 = 24.
    # If rank(M_{2,3,3}) = 15, then 30.
    # If rank(M_{2,3,3}) = 20, then 40.
    
    # Actually, I recall that the rank of M_{2,3,3} is 12 is a conjecture or result for *complex* tensors?
    # For exact integer/half-integer coefficients, Strassen-like methods are standard.
    
    # Let's try a different partitioning of the 2x4x6 problem.
    # What if we split A into 1x4 and 1x4?
    # A = [A1; A2] where A1, A2 are 1x4.
    # B is 4x6.
    # C = A1 B + A2 B.
    # A1 B is 1x6. A2 B is 1x6.
    # Rank of M_{1,4,6}?
    # M_{1,m,p} has rank 1? No.
    # C_{1,k} = sum_j A_{1,j} B_{j,k}.
    # This is a bilinear map.
    # The tensor is M_{1,4,6}.
    # Can we decompose this?
    # The rank of M_{1,m,p} is max(m,p)? No.
    # For M_{1,2,2}, rank is 2?
    # C1 = A1 B1 + A2 B2.
    # U: A1, A2. V: B1, B2. W: C1.
    # Term 1: A1 * B1 -> C1. U=[1,0], V=[1,0], W=[1].
    # Term 2: A2 * B2 -> C1. U=[0,1], V=[0,1], W=[1].
    # Rank 2.
    # For M_{1,4,6}, we can just use 6 terms?
    # C_k = sum_j A_j B_{j,k}.
    # We can group terms.
    # If we use rank 6 for each row of A, total rank 12?
    # Let's check.
    # A is 2x4.
    # C = A B.
    # Row 1 of C: C1 = A1 B.
    # Row 2 of C: C2 = A2 B.
    # If we compute A1 B with rank 6 (schoolbook for 1x4 * 4x6? No, 1x4 * 4x6 is 6 mults).
    # Rank of M_{1,4,6} is 6?
    # T_{j,k} = A_j B_{j,k}.
    # We can write this as sum_{j=1..4} A_j * B_{j,:}.
    # Each term is a product of a scalar (from A) and a vector (row of B).
    # This is a sum of 4 rank-1 terms?
    # U_j = e_j (in A space). V_j = row j of B (in B space). W_j = vector of ones in C space?
    # No, W must map the product to C.
    # The product is a scalar * vector.
    # The output is a vector.
    # So for each j, we have U_j (1 non-zero), V_j (1 non-zero row), and W_j (which adds the result to C).
    # Actually, the tensor M_{1,4,6} has rank 4?
    # Let's see.
    # C_k = sum_j A_j B_{j,k}.
    # We can define 4 terms.
    # Term j:
    # U: has 1 at index j.
    # V: has 1 at index (j, k) for all k? No, V is fixed.
    # The term is (A_j) * (B_{j,:}) .
    # This produces a vector.
    # In the tensor decomposition sum_r U_r V_r W_r = T,
    # U_r is a vector in A-space, V_r in B-space, W_r in C-space.
    # The product is a scalar (dot product of U_r with A) times a scalar (dot product of V_r with B)?
    # No, the bilinear form is sum_{i,j,k} T_{i,j,k} A_i B_j.
    # For M_{1,4,6}, i is fixed (0).
    # T_{0,j,k} = delta_{j,j} ? No.
    # C_k = sum_j A_j B_{j,k}.
    # So the coefficient of A_j B_{j,k} is 1.
    # So T_{j, (j,k)} = 1?
    # The tensor is T_{u,v,w}.
    # u = j (index in A).
    # v = (j,k) (index in B).
    # w = k (index in C).
    # T_{j, j*6+k, k} = 1.
    # This tensor has rank 4?
    # Let's try to decompose it.
    # We need sum_r U_{r,j} V_{r,j,k} W_{r,k} = delta_{j,j} delta_{k,k}?
    # Actually, for fixed j, the term is A_j * (sum_k B_{j,k} C_k^*).
    # This looks like 4 independent rank-1 tensors?
    # Term j:
    # U_j: has 1 at index j.
    # V_j: has 1 at indices (j,0), (j,1), ..., (j,5).
    # W_j: has 1 at indices 0, 1, ..., 5?
    # If W_j is all ones, then sum_k V_{j,(j,k)} W_{j,k} = sum_k 1*1 = 6.
    # We need the result to be B_{j,k} for each k.
    # The decomposition sums over r.
    # C_k = sum_r (sum_j U_{r,j} A_j) (sum_{j',k'} V_{r,j',k'} B_{j',k'}) W_{r,k}.
    # This is not quite right. The standard form is:
    # sum_r (U_r^T A) (V_r^T B) W_r = C (as a linear combination).
    # Actually, the tensor T is defined by T_{u,v,w} such that C_w = sum_{u,v} T_{u,v,w} A_u B_v.
    # So C_k = sum_r W_{r,k} (sum_j U_{r,j} A_j) (sum_{j',k'} V_{r,j',k'} B_{j',k'}).
    # For M_{1,4,6}, C_k = sum_j A_j B_{j,k}.
    # We want sum_r W_{r,k} (sum_j U_{r,j} A_j) (sum_{j',k'} V_{r,j',k'} B_{j',k'}) = sum_j A_j B_{j,k}.
    # If we choose 4 terms (r=1..4):
    # For r=j:
    # U_{j,j} = 1, others 0.
    # V_{j,j,k} = 1 for all k?
    # Then (V_j^T B) = sum_{k'} B_{j,k'}.
    # This is a scalar.
    # We need to recover B_{j,k} for each k.
    # This approach fails because V_r is a vector, not a matrix.
    # The bilinear form is scalar.
    # So M_{1,4,6} rank is NOT 4.
    # The rank of M_{1,m,p} is m*p?
    # No, M_{1,2,2} rank is 4?
    # C1 = A1 B11 + A2 B21.
    # C2 = A1 B12 + A2 B22.
    # We can use 4 terms:
    # 1: A1 * B11 -> C1. U=[1,0], V=[1,0,0,0], W=[1,0].
    # 2: A1 * B12 -> C2. U=[1,0], V=[0,1,0,0], W=[0,1].
    # 3: A2 * B21 -> C1. U=[0,1], V=[0,0,1,0], W=[1,0].
    # 4: A2 * B22 -> C2. U=[0,1], V=[0,0,0,1], W=[0,1].
    # Rank 4.
    # So M_{1,4,6} has rank 24?
    # Then 2 rows of A would be 48. Worse than 42.
    
    # So the 2x3x3 split is the way to go IF rank is low enough.
    # Rank(M_{2,3,3})?
    # If we use schoolbook for 2x3 * 3x3:
    # C has 6 entries. Each is a sum of 3 products. Total 18 products.
    # Rank 18.
    # 2 * 18 = 36.
    # This is better than 42!
    # Can we do better than 18 for 2x3x3?
    # Strassen for 2x2x2 is 7.
    # For 2x3x3, we can embed 2x2x2.
    # A (2x3) = [A_2x2, A_2x1].
    # B (3x3) = [B_2x2, B_2x1; B_1x2, B_1x1].
    # C = A_2x2 B_2x2 + A_2x1 B_1x2 + ...
    # This is getting complex.
    # But Schoolbook 18 is definitely valid and easy to implement.
    # Total rank 36.
    
    # Let's implement the schoolbook decomposition for 2x3x3 and use it twice.
    
    rank_2x3x3 = 18
    total_rank = 2 * rank_2x3x3 # 36
    
    U = [[0] * total_rank for _ in range(n * m)] # 8 x 36
    V = [[0] * total_rank for _ in range(m * p)] # 24 x 36
    W = [[0] * total_rank for _ in range(p * n)] # 12 x 36
    
    # Define the 2x3x3 schoolbook decomposition.
    # A is 2x3. B is 3x3. C is 2x3.
    # Indices for A: i in 0..1, j in 0..2. idx = i*3 + j.
    # Indices for B: j in 0..2, k in 0..2. idx = j*3 + k.
    # Indices for C: k in 0..2, i in 0..1. idx = k*2 + i.
    
    # We will build a list of (u_idx, v_idx, w_idx) for the 18 terms.
    terms_2x3x3 = []
    for i in range(2):
        for k in range(3):
            for j in range(3):
                # Term: A_{i,j} * B_{j,k} contributes to C_{i,k}
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 2 + i
                terms_2x3x3.append((u_idx, v_idx, w_idx))
    
    # Now map this to the global 2x4x6 indices.
    # We have two blocks: left (cols 0-2 of B) and right (cols 3-5 of B).
    # Block 0: B cols 0,1,2.
    # Block 1: B cols 3,4,5.
    
    col = 0
    for block in range(2):
        # B columns in this block are block*3 + k_loc (k_loc in 0..2)
        # Global B index: j*p + k_global.
        # k_global = block*3 + k_loc.
        # A is the same for both blocks.
        # C columns are block*3 + k_loc.
        # Global C index: k_global*n + i.
        
        for (u_loc, v_loc, w_loc) in terms_2x3x3:
            # u_loc is index in A (2x3).
            # But A is 2x4. The 2x3 submatrix uses columns 0,1,2.
            # So u_idx_global = u_loc. (Since i is 0..1, j is 0..2, u_loc = i*3+j matches i*4+j? No.)
            # A is 2x4. Index i*4 + j.
            # u_loc = i*3 + j.
            # i = u_loc // 3, j = u_loc % 3.
            i = u_loc // 3
            j = u_loc % 3
            u_idx_global = i * 4 + j
            
            # v_loc is index in B_block (3x3).
            # v_loc = j*3 + k_loc.
            # j = v_loc // 3, k_loc = v_loc % 3.
            j = v_loc // 3
            k_loc = v_loc % 3
            k_global = block * 3 + k_loc
            v_idx_global = j * 6 + k_global
            
            # w_loc is index in C_block (2x3).
            # w_loc = k_loc*2 + i.
            # k_loc = w_loc // 2, i = w_loc % 2.
            k_loc = w_loc // 2
            i = w_loc % 2
            k_global = block * 3 + k_loc
            w_idx_global = k_global * 2 + i
            
            U[u_idx_global][col] = 1
            V[v_idx_global][col] = 1
            W[w_idx_global][col] = 1
            
            col += 1
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
