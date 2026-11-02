"""Rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W, tol=1e-9):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        # Check all 16 required terms equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check all non-matching terms equal 0
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    a = i // m; b = i % m
                    bp = j // p; c = j % p
                    cp = k // n; ap = k % n
                    if a == ap and b == bp:
                        continue
                    s = 0.0
                    for r in range(R):
                        s += U[i][r] * V[j][r] * W[k][r]
                    if abs(s) > tol:
                        return False
        return True
    
    def build_rank14():
        """Build the known rank-14 Strassen decomposition."""
        R = 14
        U = [[0]*R for _ in range(nm)]
        V = [[0]*R for _ in range(mp)]
        W = [[0]*R for _ in range(pn)]
        
        # Left block Strassen (7 products): computes A*B_left (2x2 * 2x2)
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block Strassen (7 products): computes A*B_right (2x2 * 2x2)
        # N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # N2 = (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # N5 = (a00+a01)(b13)
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_structured():
        """
        Try to find a rank-13 decomposition by exploiting the shared A-structure.
        
        Key insight: In both Strassen blocks, the same A-linear-combinations appear:
        - a00+a11 (M1, N1)
        - a10+a11 (M2, N2)  
        - a00+a01 (M5, N5)
        - a10-a00 (M6, N6)
        - a01-a11 (M7, N7)
        
        If we can merge one pair of products across blocks, we get rank 13.
        This requires that the B-terms and W-terms align properly.
        
        Strategy: Try all possible pairings of the 5 shared A-combinations
        and check if a valid rank-13 decomposition exists.
        """
        if time.time() > deadline - 20:
            return None
        
        # The 5 shared A-combinations and their indices in U
        # A-comb 0: a00+a11 -> U[0], U[3]
        # A-comb 1: a10+a11 -> U[2], U[3]
        # A-comb 2: a00+a01 -> U[0], U[1]
        # A-comb 3: a10-a00 -> U[2], U[0]
        # A-comb 4: a01-a11 -> U[1], U[3]
        
        # For each shared A-comb, we have a left product (L) and right product (R)
        # L: U_comb * V_left * W_left
        # R: U_comb * V_right * W_right
        # Merged: U_comb * (V_left*W_left + V_right*W_right) -- but this isn't bilinear
        
        # Actually, merging requires: U_comb * V_merged * W_merged
        # where V_merged and W_merged are single vectors, not products.
        # This is only possible if V_left ⊗ W_left + V_right ⊗ W_right = V_merged ⊗ W_merged
        # i.e., the sum of two rank-1 matrices is rank 1.
        
        # This happens when V_left and V_right are linearly dependent AND
        # W_left and W_right are linearly dependent, with consistent scaling.
        
        # Let's check: for each pair, can we find scalars α, β such that:
        # V_left = α * V_merged, V_right = β * V_merged
        # W_left = β * W_merged, W_right = α * W_merged
        # (so that V_left⊗W_left + V_right⊗W_right = (αβ+βα) * V_merged⊗W_merged)
        
        # More precisely: V_left⊗W_left + V_right⊗W_merged... let me think differently.
        
        # The product term is: U_comb[r] * V_left[r] * W_left[r] + U_comb[r] * V_right[r] * W_right[r]
        # = U_comb[r] * (V_left[r]*W_left[r] + V_right[r]*W_right[r])
        
        # For this to be a single bilinear product U_merged * V_merged * W_merged,
        # we need V_left[r]*W_left[r] + V_right[r]*W_right[r] = V_merged[r]*W_merged[r]
        # for all r, with U_merged[r] = U_comb[r].
        
        # This is equivalent to: the vector (V_left[r]*W_left[r] + V_right[r]*W_right[r])_r
        # must be expressible as V_merged[r]*W_merged[r] for some V_merged, W_merged.
        
        # Let's define: for each shared A-comb c, we have:
        #   Left:  V_L_c (8-dim), W_L_c (8-dim)
        #   Right: V_R_c (8-dim), W_R_c (8-dim)
        #   U_c (4-dim)
        
        # The contribution to the tensor from comb c is:
        #   U_c ⊗ (V_L_c ∘ W_L_c + V_R_c ∘ W_R_c)  (where ∘ is element-wise)
        
        # For rank-13, we need to merge one pair. Let's try each of the 5 pairs.
        
        # First, extract the V and W vectors for each product in the rank-14 solution
        R14 = 14
        U14 = [[0]*R14 for _ in range(nm)]
        V14 = [[0]*R14 for _ in range(mp)]
        W14 = [[0]*R14 for _ in range(pn)]
        
        # Fill in the rank-14 solution
        U14[0][0]=1; U14[3][0]=1; V14[0][0]=1; V14[5][0]=1; W14[0][0]=1; W14[3][0]=1
        U14[2][1]=1; U14[3][1]=1; V14[0][1]=1; W14[1][1]=1; W14[3][1]=-1
        U14[0][2]=1; V14[1][2]=1; V14[5][2]=-1; W14[2][2]=1; W14[3][2]=1
        U14[3][3]=1; V14[4][3]=1; V14[0][3]=-1; W14[0][3]=1; W14[1][3]=1
        U14[0][4]=1; U14[1][4]=1; V14[5][4]=1; W14[0][4]=-1; W14[2][4]=1
        U14[2][5]=1; U14[0][5]=-1; V14[0][5]=1; V14[1][5]=1; W14[3][5]=1
        U14[1][6]=1; U14[3][6]=-1; V14[4][6]=1; V14[5][6]=1; W14[0][6]=1
        U14[0][7]=1; U14[3][7]=1; V14[2][7]=1; V14[7][7]=1; W14[4][7]=1; W14[7][7]=1
        U14[2][8]=1; U14[3][8]=1; V14[2][8]=1; W14[5][8]=1; W14[7][8]=-1
        U14[0][9]=1; V14[3][9]=1; V14[7][9]=-1; W14[6][9]=1; W14[7][9]=1
        U14[3][10]=1; V14[6][10]=1; V14[2][10]=-1; W14[4][10]=1; W14[5][10]=1
        U14[0][11]=1; U14[1][11]=1; V14[7][11]=1; W14[4][11]=-1; W14[6][11]=1
        U14[2][12]=1; U14[0][12]=-1; V14[2][12]=1; V14[3][12]=1; W14[7][12]=1
        U14[1][13]=1; U14[3][13]=-1; V14[6][13]=1; V14[7][13]=1; W14[4][13]=1
        
        # Product indices: 0-6 are left block, 7-13 are right block
        # Shared A-combs:
        # Comb 0 (a00+a11): left=0, right=7
        # Comb 1 (a10+a11): left=1, right=8
        # Comb 2 (a00+a01): left=4, right=11
        # Comb 3 (a10-a00): left=5, right=12
        # Comb 4 (a01-a11): left=6, right=13
        
        shared_pairs = [(0,7), (1,8), (4,11), (5,12), (6,13)]
        
        for merge_idx, (li, ri) in enumerate(shared_pairs):
            if time.time() > deadline - 10:
                break
            
            # Try to merge products li and ri
            # Extract vectors
            U_l = [U14[i][li] for i in range(nm)]
            V_l = [V14[j][li] for j in range(mp)]
            W_l = [W14[k][li] for k in range(pn)]
            
            U_r = [U14[i][ri] for i in range(nm)]
            V_r = [V14[j][ri] for j in range(mp)]
            W_r = [W14[k][ri] for k in range(pn)]
            
            # Check if U_l == U_r (they should be for shared A-combs)
            if U_l != U_r:
                continue
            
            # The combined contribution is U_l ⊗ (V_l∘W_l + V_r∘W_r)
            # We need V_l∘W_l + V_r∘W_r = V_m∘W_m for some V_m, W_m
            
            # Compute the element-wise products
            VLW_l = [V_l[j] * W_l[j] for j in range(mp)]  # This isn't right
            # Actually, the tensor contribution is:
            # T[i,j,k] += U_l[i] * V_l[j] * W_l[k] + U_r[i] * V_r[j] * W_r[k]
            # Since U_l == U_r:
            # T[i,j,k] += U_l[i] * (V_l[j] * W_l[k] + V_r[j] * W_r[k])
            
            # For this to be a single bilinear product U_m[i] * V_m[j] * W_m[k],
            # we need V_l[j]*W_l[k] + V_r[j]*W_r[k] = V_m[j]*W_m[k] for all j,k.
            
            # This means the matrix M = V_l W_l^T + V_r W_r^T must be rank 1.
            # (where V_l, W_l are column vectors)
            
            M = np.outer(V_l, W_l) + np.outer(V_r, W_r)
            # Check if M is rank 1
            try:
                rank_M = np.linalg.matrix_rank(M, tol=1e-9)
            except:
                rank_M = 8
            
            if rank_M == 1:
                # Find V_m, W_m such that M = V_m * W_m^T
                # Use SVD
                U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
                if len(s_vals) > 0 and s_vals[0] > 1e-9:
                    V_m = U_svd[:, 0] * s_vals[0]
                    W_m = Vt_svd[0, :]
                    # Scale to get integer/half-integer entries
                    # Try to find a scaling factor
                    found = False
                    for scale_num in range(1, 20):
                        for scale_den in [1, 2]:
                            V_m_scaled = [v * scale_num / scale_den for v in V_m]
                            W_m_scaled = [w * scale_den / scale_num for w in W_m]
                            # Check if all entries are close to integer or half-integer
                            def is_half_int(x):
                                return abs(x * 2 - round(x * 2)) < 1e-6
                            if all(is_half_int(x) for x in V_m_scaled) and all(is_half_int(x) for x in W_m_scaled):
                                V_m_final = [round(x * 2) / 2 for x in V_m_scaled]
                                W_m_final = [round(x * 2) / 2 for x in W_m_scaled]
                                # Build the rank-13 solution
                                R13 = 13
                                U13 = [[0]*R13 for _ in range(nm)]
                                V13 = [[0]*R13 for _ in range(mp)]
                                W13 = [[0]*R13 for _ in range(pn)]
                                
                                # Copy all products except li and ri
                                prod_map = {}
                                r_idx = 0
                                for r in range(R14):
                                    if r == li or r == ri:
                                        continue
                                    prod_map[r] = r_idx
                                    for i in range(nm):
                                        U13[i][r_idx] = U14[i][r]
                                    for j in range(mp):
                                        V13[j][r_idx] = V14[j][r]
                                    for k in range(pn):
                                        W13[k][r_idx] = W14[k][r]
                                    r_idx += 1
                                
                                # Add merged product
                                for i in range(nm):
                                    U13[i][r_idx] = U_l[i]
                                for j in range(mp):
                                    V13[j][r_idx] = V_m_final[j]
                                for k in range(pn):
                                    W13[k][r_idx] = W_m_final[k]
                                
                                if verify(U13, V13, W13):
                                    return {"U": U13, "V": V13, "W": W13}
        
        return None
    
    def try_rank13_random():
        """Try random search for rank-13 with half-integer entries."""
        if time.time() > deadline - 15:
            return None
        
        random.seed(42)
        R = 13
        values = [-1, -0.5, 0, 0.5, 1]
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Start from rank-14 and try to merge two products
            U = [[0]*R for _ in range(nm)]
            V = [[0]*R for _ in range(mp)]
            W = [[0]*R for _ in range(pn)]
            
            # Initialize with random half-integer values
            for i in range(nm):
                for r in range(R):
                    U[i][r] = random.choice(values)
            for j in range(mp):
                for r in range(R):
                    V[j][r] = random.choice(values)
            for k in range(pn):
                for r in range(R):
                    W[k][r] = random.choice(values)
            
            # Local optimization: try to fix the tensor
            # This is a non-convex problem, so use a simple hill-climbing approach
            # Compute the error
            def compute_error(U, V, W):
                err = 0.0
                R = len(U[0])
                for i in range(nm):
                    for j in range(mp):
                        for k in range(pn):
                            a = i // m; b = i % m
                            bp = j // p; c = j % p
                            cp = k // n; ap = k % n
                            target = 1.0 if (a == ap and b == bp) else 0.0
                            s = 0.0
                            for r in range(R):
                                s += U[i][r] * V[j][r] * W[k][r]
                            err += (s - target) ** 2
                return err
            
            # Simple local search: try flipping individual entries
            best_err = compute_error(U, V, W)
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(nm):
                    for r in range(R):
                        old_val = U[i][r]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            U[i][r] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err - 1e-12:
                                best_err = err
                                improved = True
                            else:
                                U[i][r] = old_val
                for j in range(mp):
                    for r in range(R):
                        old_val = V[j][r]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            V[j][r] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err - 1e-12:
                                best_err = err
                                improved = True
                            else:
                                V[j][r] = old_val
                for k in range(pn):
                    for r in range(R):
                        old_val = W[k][r]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            W[k][r] = new_val
                            err = compute_error(U, V, W)
                            if err < best_err - 1e-12:
                                best_err = err
                                improved = True
                            else:
                                W[k][r] = old_val
                
                if best_err < 1e-9:
                    if verify(U, V, W):
                        return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try rank 14 first (known good)
    result = build_rank14()
    if result is not None:
        # Try to find rank 13
        r13 = try_rank13_structured()
        if r13 is not None:
            return r13
        
        r13_rand = try_rank13_random()
        if r13_rand is not None:
            return r13_rand
        
        return result
    
    # Fallback: should not reach here
    R = 16
    U = [[0]*R for _ in range(nm)]
    V = [[0]*R for _ in range(mp)]
    W = [[0]*R for _ in range(pn)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
