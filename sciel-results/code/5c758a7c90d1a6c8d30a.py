import time
import numpy as np
import random
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Allow 110s to finish with margin
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We split 2x4x8 into two 2x4x4 problems.
    # Target: Find rank r decomposition for 2x4x4, total rank will be 2*r.
    # Known: Strassen gives 28 for 2x4x4 (4 blocks of 2x2x2). Total 56.
    # Known: Rank 20 for 2x4x4 exists.
    
    best_rank = None
    best_solution = None
    
    # --- Helper functions for search ---
    
    def generate_initial_decomp(m, p, n, rank, seed=None):
        """
        Generates a random sparse decomposition for m x p matrix mult.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        if seed is not None:
            random.seed(seed)
            
        # Sparse initialization
        density = 0.2
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            # Pick non-zero indices
            num_u = max(1, int(n * m * density))
            num_v = max(1, int(m * p * density))
            num_w = max(1, int(p * n * density))
            
            idx_u = random.sample(range(n * m), num_u)
            idx_v = random.sample(range(m * p), num_v)
            idx_w = random.sample(range(p * n), num_w)
            
            for i in idx_u:
                U[i, r] = random.choice([-1, 1])
            for i in idx_v:
                V[i, r] = random.choice([-1, 1])
            for i in idx_w:
                W[i, r] = random.choice([-1, 1])
                
        return U, V, W

    def compute_residual(U, V, W):
        """
        Computes || U*V*W - T_mult ||_F
        T_mult[i, j, k] = 1 if i==j else 0? 
        Actually, tensor T for matrix mult: 
        T is a tensor such that C_{ik} = sum_j A_{ij} B_{jk}.
        Decomposition: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
        Here indices:
        A is n x m. Flatten index for A: a = i*m + j.
        B is m x p. Flatten index for B: b = j*p + k.
        C is n x p. Flatten index for C: c = k*n + i. (Trace convention W[k*n+i, r])
        
        We want sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
        Where T[a, b, c] = 1 if the indices correspond to valid matrix multiplication alignment.
        a = i*m + j
        b = j*p + k
        c = k*n + i
        
        Valid if: 
        j_from_a == j_from_b
        i_from_a == i_from_c
        k_from_b == k_from_c
        """
        # Precompute target tensor
        # For 2x4x4:
        # n=2, m=4, p=4
        pass 
        
    def verify_decomposition(U, V, W, n, m, p):
        """
        Verifies if U, V, W correctly decompose the multiplication tensor.
        Returns True if exact.
        """
        # Target tensor values
        # T[a, b, c] = 1 if a% m == b // p AND a // m == c % n AND b % p == c // n
        # Note: 
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        
        rank = U.shape[1]
        
        # Iterate over all possible a, b, c
        # Optimization: Only check valid combinations where T=1
        # And check that all other combinations are 0
        
        # Create a sparse representation of the target
        # For 2x4x4, total combos = 8*16*8 = 1024. Very small.
        # For 2x4x8, total combos = 8*32*16 = 4096. Small.
        
        # We'll check 20 random valid targets and 20 random invalid targets
        # Or just all of them if small enough.
        
        all_a = range(n * m)
        all_b = range(m * p)
        all_c = range(p * n)
        
        # Build list of (a, b, c, expected_val)
        checks = []
        for a in all_a:
            i, j = divmod(a, m)
            for b in all_b:
                j2, k = divmod(b, p)
                if j == j2:
                    # Valid alignment
                    expected = 1
                    for c in all_c:
                        k2, i2 = divmod(c, n)
                        if k2 == k and i2 == i:
                            checks.append((a, b, c, 1))
                            break
                        # else expected 0 for this c? No, c is determined by i, k.
                        # Actually, for fixed a, b, there is only one c that has T=1.
                        # The rest are 0.
        # Simpler: Iterate all (a, b, c) triples
        max_iter = len(all_a) * len(all_b) * len(all_c)
        if max_iter < 10000:
            for a in all_a:
                i, j = divmod(a, m)
                for b in all_b:
                    j2, k = divmod(b, p)
                    for c in all_c:
                        k2, i2 = divmod(c, n)
                        expected = 1 if (j == j2 and i == i2 and k == k2) else 0
                        # Compute sum_r U[a,r] V[b,r] W[c,r]
                        # Use dot product
                        # val = np.dot(np.dot(U[:, r], V[:, r]), W[:, r])? No.
                        # val = sum_r U[a,r] * V[b,r] * W[c,r]
                        # This is equivalent to (U[:, r] * V[:, r]) dot W[:, r]
                        # Or: sum_r U[a,r] * (V[:, r] * W[:, r])[b]? No.
                        # Efficiently: 
                        # Precompute M_r = V[:, r] * W[:, r]
                        # Then val = sum_r U[a,r] * M_r[b]
                        
                        pass
        
        # Efficient verification for small matrices
        # Precompute for each rank r:
        # V[:, r] (m*p,) and W[:, r] (p*n,)
        # We need to check: 
        # For all a, c: sum_r U[a,r] * (sum_b' V[b',r] W[c',r] T[...]) ... 
        # Actually, let's just use the definition.
        
        # Residual check
        # For each r, let v_r = V[:, r], w_r = W[:, r].
        # Let M_r = v_r * w_r (element-wise, shape m*p)
        # Then C_calc[c] = sum_a (A_real[a] * U[a,r]) * ... no.
        
        # Standard check:
        # C_calc_{ik} = sum_j sum_r U_{i,r} V_{j,r} W_{k,r} ? 
        # No.
        # C_{ik} = sum_j A_{ij} B_{jk}
        # Decomp: C_{ik} = sum_r U_{i,r} V_{j,r} W_{k,r} ? 
        # Wait, the tensor is T_{a,b,c}.
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
        
        # Let's compute the left hand side LHS[c] for each c?
        # No, it's a trilinear form.
        
        # Method:
        # 1. Create a dummy matrix A (n x m) with random values.
        # 2. Create a dummy matrix B (m x p) with random values.
        # 3. Compute C_real = A @ B.
        # 4. Compute C_decomp using U, V, W.
        #    C_decomp[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # 5. Check if C_real == C_decomp exactly (integers).
        
        random.seed(42)
        A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)])
        B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)])
        
        C_real = A @ B
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            # u_r: length n*m
            # v_r: length m*p
            # w_r: length p*n
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            
            # Reconstruct contributions
            # C_decomp[i, k] += sum_j u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            # Iterate over j to optimize
            for j in range(m):
                # u_part: indices i*m+j for i in 0..n-1
                # v_part: indices j*p+k for k in 0..p-1
                # w_part: indices k*n+i for k in 0..p-1, i in 0..n-1
                
                # We can form a matrix W_block where W_block[k, i] = w[k*n+i]
                # Then C_decomp[i, k] contribution = u[i*m+j] * v[j*p+k] * w[k*n+i]
                
                # Sum over j:
                # C_decomp[i, k] = sum_j u[i*m+j] * v[j*p+k] * w[k*n+i]
                
                pass
        
        # Efficient calculation:
        # For each r:
        # Create W_matrix (p x n) from w
        # For each j:
        #   val_j = u[i*m+j] for all i -> vector u_j (n x 1)
        #   val_k = v[j*p+k] for all k -> vector v_j (1 x p)
        #   Outer product u_j * v_j (n x p)
        #   Multiply element-wise with W_matrix (p x n) transposed? 
        #   w_idx = k*n+i. W_matrix[k, i] = w[k*n+i].
        #   So element [i, k] of C_decomp += (u_i * v_k) * W_matrix[k, i]
        
        for r in range(rank):
            w = W[:, r]
            W_mat = w.reshape(p, n) # W_mat[k, i]
            W_mat_T = W_mat.T # W_mat_T[i, k]
            
            for j in range(m):
                u_vec = U[(slice(None), slice(None))] # This is slow in python loop
                # Better:
                # u_indices = [i*m+j for i in range(n)]
                # v_indices = [j*p+k for k in range(p)]
                
                u_vals = np.array([U[i * m + j, r] for i in range(n)])
                v_vals = np.array([V[j * p + k, r] for k in range(p)])
                
                # Outer product u_vals (n,1) and v_vals (1,p)
                outer = np.outer(u_vals, v_vals)
                
                # Element-wise multiply with W_mat_T (n, p)
                contrib = outer * W_mat_T
                
                C_decomp += contrib
                
        if np.all(C_real == C_decomp):
            return True
        return False

    # --- Search Loop for 2x4x4 ---
    
    n_sub, m_sub, p_sub = 2, 4, 4
    
    target_ranks = [20, 24, 28] # Try 20, then 24, fallback 28
    
    solution_found = False
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
            
        max_tries = 100 if target_rank < 28 else 10
        start_search = time.time()
        
        for attempt in range(max_tries):
            if time.time() > deadline - 20: # Reserve 20s for final assembly
                break
                
            U, V, W = generate_initial_decomp(n_sub, m_sub, p_sub, target_rank, seed=attempt)
            
            # Check
            if verify_decomposition(U, V, W, n_sub, m_sub, p_sub):
                solution_found = True
                best_rank = target_rank
                best_solution = (U, V, W)
                print(f"Found rank {target_rank} for 2x4x4")
                break
                
            # Optional: Local Search (Greedy Flip)
            # Since exact integer verification is brittle with random init, 
            # we might need a solver. 
            # Given time constraints, simple random restart is safer than complex local search 
            # which might get stuck.
            
        if solution_found:
            break

    if not solution_found:
        # Fallback: Hardcoded Strassen Rank 28 for 2x4x4
        # 2x4x4 splits into 4 blocks of 2x2x2.
        # Strassen 2x2x2 rank 7.
        # We need to map the 4 blocks to the 28 terms.
        
        print("Fallback to Strassen Rank 56")
        
        # Reuse the logic from the seed for Rank 56
        n, m, p = DIMENSIONS
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_U = [
            [(0, 1), (3, 1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (0, -1)],
            [(1, 1), (3, -1)],
        ]
        strassen_V = [
            [(0, 1), (3, 1)],
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(2, 1), (0, -1)],
            [(3, 1)],
            [(0, 1), (1, 1)],
            [(2, 1), (3, 1)],
        ]
        strassen_W = [
            [(0, 1), (3, 1)],
            [(1, 1), (3, -1)],
            [(2, 1), (3, 1)],
            [(0, 1), (1, 1)],
            [(0, -1), (2, 1)],
            [(3, 1)],
            [(0, 1)],
        ]
        
        def add_strassen_block(base_rank, i_off, j_off, k_off):
            for r in range(7):
                cr = base_rank + r
                for idx, val in strassen_U[r]:
                    li, lj = divmod(idx, 2)
                    U[(li + i_off) * 4 + (lj + j_off)][cr] += val
                for idx, val in strassen_V[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj + j_off) * 8 + (lk + k_off)][cr] += val
                for idx, val in strassen_W[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk + k_off) * 2 + (li + i_off)][cr] += val
        
        for j_block in range(2):
            for k_block in range(4):
                base = (j_block * 4 + k_block) * 7
                add_strassen_block(base, 0, j_block * 2, k_block * 2)
                
        # Convert to lists if needed, but seed used lists.
        # Return in the same format
        return {"U": U, "V": V, "W": W}

    # Assembly for 2x4x8
    # We have U_sub (8, r), V_sub (16, r), W_sub (8, r) for 2x4x4.
    # We need U_tot (8, 2r), V_tot (32, 2r), W_tot (16, 2r).
    
    U_sub, V_sub, W_sub = best_solution
    r = U_sub.shape[1]
    
    U_tot = np.zeros((8, 2 * r))
    V_tot = np.zeros((32, 2 * r))
    W_tot = np.zeros((16, 2 * r))
    
    # Block 1 (p=0..3)
    U_tot[:, 0:r] = U_sub
    V_tot[0:16, 0:r] = V_sub
    W_tot[0:8, 0:r] = W_sub
    
    # Block 2 (p=4..7)
    # A is same.
    U_tot[:, r:2*r] = U_sub
    # B is shifted. B[j, 4+k]. Index in flat V is j*8 + (4+k) = j*8 + 4 + k.
    # In V_sub, index was j*4 + k.
    # So we need to place V_sub[flat_index] into V_tot[flat_index_shifted]
    # V_sub is 16 elements (4x4).
    # V_tot for this block is rows 16:32? 
    # Wait. V_tot has shape (m*p, rank) = (4*8, rank) = (32, rank).
    # The "first" block of B corresponds to columns 0..3. Flat indices 0..15.
    # The "second" block of B corresponds to columns 4..7. Flat indices 16..31.
    
    # So:
    V_tot[16:32, r:2*r] = V_sub
    
    # C is split into cols 0..3 and 4..7.
    # W_tot has shape (p*n, rank) = (8*2, rank) = (16, rank).
    # First block C (cols 0..3): indices 0..7.
    # Second block C (cols 4..7): indices 8..15.
    
    W_tot[8:16, r:2*r] = W_sub
    
    # Convert to lists for return
    U_list = U_tot.tolist()
    V_list = V_tot.tolist()
    W_list = W_tot.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END

