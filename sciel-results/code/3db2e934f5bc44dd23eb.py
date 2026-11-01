import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # The tensor for 2x2 * 2xp
    # U: 4 x R_sub, V: 2p x R_sub, W: 2p x R_sub
    # Indices:
    # a = i*m + j (i in 0..1, j in 0..1) -> 4
    # b = j*p + k (j in 0..1, k in 0..p-1) -> 2p
    # c = k*n + i (k in 0..p-1, i in 0..1) -> 2p
    
    def verify_sub(U, V, W, p_sub):
        """Verify exact decomposition for 2x2*2xp_sub."""
        R = U.shape[1]
        # Nonzeros: (a, b, c) where a=i*2+j, b=j*p_sub+k, c=2*k+i
        # Zeros: everything else
        # We can just compute the full tensor and check.
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        for i in range(2):
            for j in range(2):
                for k in range(p_sub):
                    a = i * 2 + j
                    b = j * p_sub + k
                    c = k * 2 + i
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        return False
                    # Check zeros for this (a,b) pair? 
                    # It's faster to just check the whole tensor at the end if we want to be safe,
                    # but checking nonzeros first is a good filter.
        
        # Check zeros
        # T is 4 x 2p x 2p
        # For each a, b:
        #   if (a//2, a%2) and (b//p, b%p) don't match j, then T[a,b,:] should be 0?
        # No, T[a,b,c] is 0 unless i=a//2, j=a%2, k=c//2, and b=j*p+k.
        # So for fixed a,b, there is at most one c where T[a,b,c] can be non-zero.
        # That c is determined by i=a//2, k=b%p. c = k*2+i.
        # If b%p != k such that j=b//p matches a%2, then T[a,b,:] must be all 0.
        
        for a in range(4):
            i1, j1 = a // 2, a % 2
            for b in range(2 * p_sub):
                j2, k1 = b // p_sub, b % p_sub
                if i1 == i2 and j1 == j2: # Wait, i2 is from c.
                    pass
                # Actually, simpler:
                # T[a,b,c] is non-zero ONLY if:
                # i = a//2, j = a%2
                # j = b//p_sub, k = b%p_sub
                # c = k*2 + i
                # So for fixed a,b, the only potentially non-zero c is c_star = (b%p_sub)*2 + (a//2).
                # But we must also have b//p_sub == a%2.
                if b // p_sub != a % 2:
                    # All entries in T[a,b,:] must be 0
                    if np.any(np.abs(T[a, b, :]) > 1e-9):
                        return False
                else:
                    # Only T[a,b,c_star] should be 1, others 0
                    c_star = (b % p_sub) * 2 + (a // 2)
                    if abs(T[a, b, c_star] - 1.0) > 1e-9:
                        return False
                    # Check other c in this row
                    for c in range(2 * p_sub):
                        if c != c_star and abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    def try_find_sub_decomp(p_sub, R, num_trials=1000, seed=None):
        """
        Try to find a rank-R decomposition for 2x2*2xp_sub.
        Strategy:
        1. Randomly initialize U (4 x R) with entries in {-1, 0, 1}.
        2. For a fixed U, the problem is bilinear in V and W.
        3. We can alternate:
           - Fix V, solve for W (linear system).
           - Fix W, solve for V (linear system).
        4. Or, fix U and V, solve for W.
        
        Let's use the linear system approach:
        For fixed U and V, we have 4*2p equations for W (2p*R unknowns).
        Actually, for each c, W[c,:] is a vector of length R.
        The equations are: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
        This is a system of 4*2p equations in 2p*R unknowns.
        Wait, W has 2p rows. Each row W[c,:] has R unknowns. Total 2p*R unknowns.
        The equations are coupled? No.
        For a specific c, the equation involves W[c,0..R-1].
        The coefficients are U[a,r] * V[b,r].
        So for each c, we have a system of 4*2p equations?
        No. The equation is for ALL (a,b,c).
        sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
        For a fixed c, this equation must hold for ALL (a,b).
        So for fixed c, we have 4*2p equations in R unknowns (W[c,0..R-1]).
        This system is usually overdetermined (4*2p > R).
        We need to find V such that for all c, the system is consistent.
        
        Alternative: Fix U. Randomly initialize V.
        Solve for W using least squares? No, we need EXACT solution.
        If we use least squares, we might get a non-integer solution.
        
        Better approach for small R:
        Enumerate U columns from {-1,0,1}^4.
        For each U, try to find V, W.
        
        Let's try a different linear algebra trick.
        T = sum_r u_r v_r w_r^T (outer products).
        This is a rank-R decomposition of the tensor.
        
        Let's try the "Fix U, Solve V, Solve W" loop.
        1. Random U (integers).
        2. Random V (integers).
        3. Solve for W:
           For each c, we have a linear system A_c W[c,:] = b_c.
           Where A_c is a matrix of size (4*2p) x R?
           No. The equation is sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
           Let's vectorize.
           Let M_c be a matrix where row index is (a,b) and col index is r.
           M_c[(a,b), r] = U[a,r] * V[b,r].
           Then M_c @ W[c,:] = T[:, :, c].reshape(-1).
           We need to solve M_c @ W[c,:] = T[:, :, c].reshape(-1) for W[c,:].
           If M_c is full rank and the system is consistent, we have a unique solution.
           If R < 4*2p, it's overdetermined. We need the system to be consistent.
           
        So the algorithm is:
        1. Pick U (4xR, integers).
        2. Pick V (2p x R, integers).
        3. For each c in 0..2p-1:
             Form M_c (4*2p x R).
             Check if T[:,:,c].reshape(-1) is in the column space of M_c.
             If yes, solve for W[c,:].
             If no, fail.
        4. If all c succeed, we have W. Check if W entries are integers/half-integers?
           The problem allows integer/half-integer.
           If the solution W is rational, we can scale?
           Actually, if U, V are integers, and T is integer, the solution W from a linear system
           with integer coefficients will be rational.
           We can check if W is close to integers or half-integers.
           
        Let's implement this.
        """
        if seed is not None:
            random.seed(seed)
            
        for trial in range(num_trials):
            # Generate random U with entries in {-1, 0, 1}
            # To ensure diversity, we can force some structure, but random is okay for small R.
            U = np.random.choice([-1, 0, 1], size=(4, R))
            # Ensure U is not trivial (e.g., all zeros)
            if np.all(U == 0):
                continue
                
            # Generate random V with entries in {-1, 0, 1}
            V = np.random.choice([-1, 0, 1], size=(2 * p_sub, R))
            if np.all(V == 0):
                continue
            
            W = np.zeros((2 * p_sub, R))
            success = True
            
            for c in range(2 * p_sub):
                # Construct M_c: (4 * 2p_sub) x R
                # Row index: a * 2p_sub + b
                # Col index: r
                # M_c[idx, r] = U[a, r] * V[b, r]
                
                # We can construct M_c efficiently
                # U is 4xR, V is 2p x R
                # M_c is outer product of U and V, reshaped?
                # M_c[a, b, r] = U[a, r] * V[b, r]
                # This is a 3D array 4 x 2p x R.
                # We want to solve for W[c, r].
                # Equation: sum_r M_c[a, b, r] * W[c, r] = T[a, b, c]
                
                # Let's build the matrix A (size 4*2p x R)
                A = np.zeros((4 * 2 * p_sub, R))
                b_vec = np.zeros(4 * 2 * p_sub)
                
                for a in range(4):
                    i1, j1 = a // 2, a % 2
                    for b in range(2 * p_sub):
                        j2, k1 = b // p_sub, b % p_sub
                        idx = a * (2 * p_sub) + b
                        # T[a, b, c] is 1 if i1 == i2 and j1 == j2
                        # i2 = c % 2, k2 = c // 2
                        # j2 must equal j1, and k2 must equal k1?
                        # Wait, T[a,b,c] = 1 iff:
                        # a = i*2+j, b = j*p+k, c = k*2+i
                        # So i = a//2, j = a%2
                        # j = b//p, k = b%p
                        # k = c//2, i = c%2
                        # Consistency: a//2 == c%2 and a%2 == b//p and b%p == c//2
                        
                        i_val = a // 2
                        j_val = a % 2
                        j_b = b // p_sub
                        k_b = b % p_sub
                        i_c = c % 2
                        k_c = c // 2
                        
                        if i_val == i_c and j_val == j_b and k_b == k_c:
                            b_vec[idx] = 1.0
                        else:
                            b_vec[idx] = 0.0
                        
                        # Fill A row
                        for r in range(R):
                            A[idx, r] = U[a, r] * V[b, r]
                
                # Solve A @ w = b_vec
                # A is (4*2p) x R. b_vec is (4*2p).
                # We need an exact solution.
                # Use np.linalg.lstsq? It gives least squares.
                # We need to check if the residual is 0.
                
                if R == 0:
                    if np.any(b_vec != 0):
                        success = False
                        break
                    continue
                
                # Solve
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                except:
                    success = False
                    break
                
                # Check residual
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        success = False
                        break
                else:
                    # If underdetermined or exact, check A @ sol - b_vec
                    res = np.linalg.norm(A @ sol - b_vec)
                    if res > 1e-9:
                        success = False
                        break
                
                # Check if sol is close to integer or half-integer
                # The problem allows integer/half-integer.
                # Let's check if 2*sol is close to integer.
                sol_2 = 2 * sol
                if not np.allclose(sol_2, np.round(sol_2), atol=1e-7):
                    # It's not a valid half-integer solution.
                    # We could try to snap it, but if the linear system has a unique solution,
                    # and it's not rational with denom 2, it's invalid.
                    # However, if the system is underdetermined, lstsq gives the min-norm solution.
                    # There might be another solution that IS integer/half-integer.
                    # For now, we reject it. This might miss valid solutions.
                    success = False
                    break
                
                W[c, :] = sol
            
            if success:
                # Verify the full decomposition
                if verify_sub(U, V, W, p_sub):
                    return U, V, W
        
        return None

    # We want to find a sub-block decomposition.
    # Total p=15.
    # We can split 15 into blocks.
    # If we find a rank R3 for 2x2x3, we can use 5 blocks. Total rank 5*R3.
    # If we find a rank R2 for 2x2x2 (Strassen is 7), and R1 for 2x2x1 (Schoolbook is 4).
    # 15 = 7*2 + 1. Rank = 7*7 + 4 = 53. (Current best)
    # 15 = 5*3. Rank = 5*R3.
    # If R3 < 11, we beat 53.
    # Lower bound for 2x2x3: max(4, 6, 6) = 6.
    # Strassen for 2x2x2 is 7. Adding 1 column schoolbook (4) gives 11.
    # Can we do 2x2x3 in rank 10? Or 9? Or 8?
    
    best_result = None
    best_rank = 60 # Schoolbook for 15
    
    # Try to find 2x2x3 decomposition
    # Start with R=8, 9, 10
    for R3 in range(8, 11):
        # Give it some time
        time_limit = 30
        start = time.time()
        U3, V3, W3 = try_find_sub_decomp(3, R3, num_trials=500, seed=42+R3)
        if U3 is not None:
            total_rank = 5 * R3
            if total_rank < best_rank:
                # Construct full 2x2x15
                U_full = np.zeros((4, total_rank))
                V_full = np.zeros((30, total_rank))
                W_full = np.zeros((30, total_rank))
                
                for block in range(5):
                    k_offset = block * 3
                    for r in range(R3):
                        col = block * R3 + r
                        U_full[:, col] = U3[:, r]
                        # V3 index: j*3+k_local
                        # V_full index: j*15+k_global
                        for j in range(2):
                            for k_local in range(3):
                                V_full[j * 15 + (k_offset + k_local), col] = V3[j * 3 + k_local, r]
                        # W3 index: k_local*2+i
                        # W_full index: k_global*2+i
                        for k_local in range(3):
                            for i in range(2):
                                W_full[(k_offset + k_local) * 2 + i, col] = W3[k_local * 2 + i, r]
                
                # Final verification
                T_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(15):
                            a = i * 2 + j
                            b = j * 15 + k
                            c = k * 2 + i
                            if abs(T_full[a, b, c] - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Check zeros (optional, verify_sub did it for sub, but let's be safe)
                    # ... (skip for brevity, assume valid if nonzeros match and rank is low enough to not have accidental zeros? No, must check)
                    # Actually, verify_sub checked zeros for the sub-block.
                    # Since the blocks are disjoint in k, the zero check for the full tensor
                    # reduces to the zero check for each sub-block.
                    # T_full[a,b,c] is non-zero only if b and c are in the same block.
                    # If they are in different blocks, the sum is 0 because U,V,W are zero for other blocks?
                    # No, U is shared. But V and W are block-diagonal in k.
                    # V_full[b, col] is non-zero only if b's k is in the block.
                    # So if b and c are in different blocks, V_full[b, col] and W_full[c, col] cannot be non-zero simultaneously for the same col?
                    # Yes, because col corresponds to a specific block.
                    # So T_full[a,b,c] = sum_blocks sum_r U[a,r] V_full[b, block*r+r] W_full[c, block*r+r]
                    # If b is in block 1 and c is in block 2, then for block 1, W_full[c, ...] is 0.
                    # For block 2, V_full[b, ...] is 0.
                    # So T_full[a,b,c] = 0.
                    # Thus, if sub-blocks are valid, full tensor is valid.
                    
                    best_rank = total_rank
                    best_result = {
                        "U": U_full.tolist(),
                        "V": V_full.tolist(),
                        "W": W_full.tolist()
                    }
                    break # Found a better one, stop searching for this R3? Or try lower R3?
        
        if time.time() - start > time_limit:
            break
        
        if best_rank < 53:
            break

    if best_result:
        return best_result
    
    # Fallback: 53-rank construction
    def make_strassen_block(k0, k1, col_start, U, V, W):
        col = col_start
        U[0, col] = 1; U[3, col] = 1
        V[k0, col] = 1; V[15+k1, col] = 1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[k0, col] = 1
        W[2*k0, col] = 1; W[2*k1, col] = 1
        col += 1
        U[0, col] = 1
        V[15+k1, col] = 1; V[k0, col] = -1
        W[2*k1+1, col] = 1; W[2*k1, col] = 1
        col += 1
        U[3, col] = 1
        V[15+k0, col] = 1; V[15+k1, col] = -1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[15+k1, col] = 1
        W[2*k0, col] = -1; W[2*k1+1, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[15+k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[15+k0, col] = 1; V[15+k1, col] = 1
        W[2*k0, col] = 1
        col += 1
        return col
    
    def make_schoolbook_block(k, col_start, U, V, W):
        col = col_start
        U[0, col] = 1; V[k, col] = 1; W[2*k, col] = 1; col += 1
        U[1, col] = 1; V[15+k, col] = 1; W[2*k, col] = 1; col += 1
        U[2, col] = 1; V[k, col] = 1; W[2*k+1, col] = 1; col += 1
        U[3, col] = 1; V[15+k, col] = 1; W[2*k+1, col] = 1; col += 1
        return col
    
    rank_53 = 7*7 + 4
    U = np.zeros((4, rank_53))
    V = np.zeros((30, rank_53))
    W = np.zeros((30, rank_53))
    
    col = 0
    for block_idx in range(7):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        col = make_strassen_block(k0, k1, col, U, V, W)
    col = make_schoolbook_block(14, col, U, V, W)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
