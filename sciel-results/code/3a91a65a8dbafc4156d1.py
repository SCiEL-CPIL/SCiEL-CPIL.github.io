import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Tensor T_{a,b,c,d,i,k} is 1 if a==i, b==d, c==k else 0
    # Decomposition: sum_l U_{a*m+b, l} V_{c*p+d, l} W_{k*n+i, l}
    
    # Precompute the target tensor as a sparse dictionary for speed
    # Key: (a, b, c, d, i, k) -> Value: 1 if a==i and c==k and b==d
    # We will iterate over the non-zero entries of the target tensor to check them.
    # There are n * m * p = 2 * 3 * 6 = 36 non-zero entries.
    # For each entry, we compute the sum.
    
    # To speed up verification, we can just check the 36 non-zero entries and a few random zero entries.
    # If the sum is correct for non-zeros and random zeros, it's likely correct.
    # But for exactness, we should check all.
    # 1296 entries is small enough to check all if done efficiently.
    
    # Let's use a vectorized check.
    # U: (nm, r), V: (mp, r), W: (pn, r)
    # We want to check if sum_r U[a*m+b, r] * V[c*p+d, r] * W[k*n+i, r] == Target[a,b,c,d,i,k]
    
    # This is equivalent to checking if the tensor T reconstructed from U,V,W matches the target.
    # We can't easily vectorize the 6D comparison without creating the 6D array.
    # 2*3*3*6*2*6 = 1296 floats. This is tiny.
    
    T_recon = np.zeros((n, m, m, p, n, p))
    for l in range(r):
        u = U[:, l].reshape(n, m)
        v = V[:, l].reshape(m, p)
        w = W[:, l].reshape(p, n)
        # T[a, b, c, d, i, k] += u[a, b] * v[c, d] * w[k, i]
        # Broadcasting:
        # u[a,b] -> (n, m, 1, 1, 1, 1)
        # v[c,d] -> (1, 1, m, p, 1, 1)
        # w[k,i] -> (1, 1, 1, 1, n, p)
        
        T_recon += u[:, :, None, None, None, None] * v[None, None, :, :, None, None] * w[None, None, None, None, :, :]
        
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            if a == i and c == k and b == d:
                                T_target[a, b, c, d, i, k] = 1.0
                                
    return np.allclose(T_recon, T_target, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy:
    # The current best known is 34.
    # Schoolbook is 36.
    # The inspiration mentions a Rank 22 solution using two copies of Rank 11 for M_{2,3,3}.
    # Wait, M_{2,3,3} has dimensions 2x3, 3x3. Rank 11?
    # Let's verify the Rank 11 claim for M_{2,3,3}.
    # If R(2,3,3) = 11, then we can combine two blocks to get R(2,3,6) <= 22?
    # M_{2,3,6} can be viewed as M_{2,3,3} \otimes M_{1,1,2}? No.
    # M_{2,3,6} is multiplying (2x3) by (3x6).
    # We can split the 6 columns into two groups of 3.
    # Then M_{2,3,6} = M_{2,3,3} (for cols 0-2) + M_{2,3,3} (for cols 3-5).
    # If we have a rank 11 decomposition for M_{2,3,3}, we can use two independent copies to get rank 22.
    # This would be a massive improvement over 34.
    # Is R(2,3,3) actually 11?
    # Known bounds: R(2,3,3) <= 17 (Strassen-like?).
    # Actually, R(2,2,2)=7.
    # R(2,3,3): 
    #   Schoolbook: 2*3*3 = 18.
    #   Can we do better?
    #   Let's try to find a rank 11 decomposition for M_{2,3,3}.
    #   If we find it, we can construct a rank 22 solution for M_{2,3,6} easily.
    #   Even rank 12 for M_{2,3,3} would give rank 24 for M_{2,3,6}, which is better than 34.
    
    # Let's search for a low-rank decomposition of M_{2,3,3}.
    # Dimensions: n=2, m=3, p=3.
    # U: (6, r), V: (9, r), W: (6, r).
    
    def search_m233_rank(max_rank, deadline):
        # We will perform a randomized search for integer/half-integer solutions.
        # But random search is hard.
        # Let's use a known structure or heuristic.
        # For M_{2,3,3}, we can try to extend M_{2,2,2} (rank 7) by adding columns.
        # M_{2,2,2} uses 7 terms.
        # M_{2,3,3} has 3x the output size of M_{2,2,2} in one dimension? No.
        
        # Let's try a greedy construction.
        # Start with schoolbook rank 18.
        # Try to merge terms.
        
        # Alternatively, use the known result if available.
        # A paper "Low rank decompositions of the matrix multiplication tensor" might have it.
        # For now, let's assume we can find rank <= 17.
        # The current program uses rank 17 blocks.
        # So R(2,3,3) <= 17 is already established in the current code (U233_seed has 17 cols).
        # The current program uses 2 * 17 = 34.
        
        # Can we do better than 17 for M_{2,3,3}?
        # If we find rank 16 for M_{2,3,3}, we get rank 32 for M_{2,3,6}.
        # If we find rank 15, we get rank 30.
        
        # Let's try to find a rank 16 decomposition for M_{2,3,3}.
        # We will use a local search on the rank 17 solution provided in the current code.
        
        # Extract the rank 17 solution from the current code's get_rank34_solution
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        # Verify seed
        if not verify_decomposition(U233_seed, V233_seed, W233_seed, 2, 3, 3):
            return None, None, None, 17

        # Try to reduce rank from 17 to 16.
        # We can try to merge two terms in the M_{2,3,3} block.
        # If we succeed, we get rank 16 for M_{2,3,3}.
        # Then we can construct M_{2,3,6} with rank 16 + 16 = 32?
        # Or can we share terms between the two blocks?
        
        # Let's first try to find a rank 16 M_{2,3,3}.
        # We will use the "Pair Merging" search on the M_{2,3,3} block.
        
        def check_merge_m233(j, k, a, b, c):
            u_new = U233_seed[:, j] + a * U233_seed[:, k]
            v_new = V233_seed[:, j] + b * V233_seed[:, k]
            w_new = W233_seed[:, j] + c * W233_seed[:, k]
            
            # Check if T_j + T_k = T_new
            # We can just build the new matrices and verify.
            # But that's slow.
            # Let's just verify the full decomposition if we find a candidate.
            # For speed, check a few random points.
            for _ in range(20):
                aa = np.random.randint(0, 2)
                bb = np.random.randint(0, 3)
                cc = np.random.randint(0, 3)
                dd = np.random.randint(0, 3)
                ii = np.random.randint(0, 2)
                kk = np.random.randint(0, 3)
                
                target = 1.0 if (aa == ii and cc == kk and bb == dd) else 0.0
                # Sum over all l except j, k
                s = 0.0
                for l in range(17):
                    if l == j or l == k:
                        continue
                    s += U233_seed[aa*3+bb, l] * V233_seed[cc*3+dd, l] * W233_seed[kk*2+ii, l]
                
                # Add new term
                s += u_new[aa*3+bb] * v_new[cc*3+dd] * w_new[kk*2+ii]
                
                if not np.isclose(s, target, atol=1e-9):
                    return False
            return True

        best_rank_m233 = 17
        best_U, best_V, best_W = U233_seed, V233_seed, W233_seed
        
        # Search for rank 16
        for j in range(17):
            if time.time() > deadline:
                break
            for k in range(j+1, 17):
                if time.time() > deadline:
                    break
                for a in range(-2, 3):
                    for b in range(-2, 3):
                        for c in range(-2, 3):
                            if check_merge_m233(j, k, a, b, c):
                                # Build new rank 16 solution
                                U_new = np.zeros((6, 16))
                                V_new = np.zeros((9, 16))
                                W_new = np.zeros((6, 16))
                                
                                cols_to_keep = [x for x in range(17) if x != j and x != k]
                                for new_idx, old_idx in enumerate(cols_to_keep):
                                    U_new[:, new_idx] = U233_seed[:, old_idx]
                                    V_new[:, new_idx] = V233_seed[:, old_idx]
                                    W_new[:, new_idx] = W233_seed[:, old_idx]
                                    
                                U_new[:, 15] = U233_seed[:, j] + a * U233_seed[:, k]
                                V_new[:, 15] = V233_seed[:, j] + b * V233_seed[:, k]
                                W_new[:, 15] = W233_seed[:, j] + c * W233_seed[:, k]
                                
                                if verify_decomposition(U_new, V_new, W_new, 2, 3, 3):
                                    best_rank_m233 = 16
                                    best_U, best_V, best_W = U_new, V_new, W_new
                                    break
                if best_rank_m233 == 16:
                    break
            if best_rank_m233 == 16:
                break
                
        # Now construct M_{2,3,6} from M_{2,3,3}
        # If we found rank 16, we can use two blocks of rank 16 -> rank 32.
        # Can we do better?
        # Maybe we can share some terms?
        # The two blocks operate on disjoint sets of columns in V and W.
        # V1 acts on cols 0-2, V2 acts on cols 3-5.
        # W1 acts on rows 0-1 of output? No, W is (p*n, rank).
        # For M_{2,3,3}, W is (6, rank). Indices k*n+i where k in 0..2, i in 0..1.
        # For M_{2,3,6}, W is (12, rank). Indices k*n+i where k in 0..5, i in 0..1.
        # Block 1 uses W indices 0..5 (k=0,1,2).
        # Block 2 uses W indices 6..11 (k=3,4,5).
        # Since the W indices are disjoint, the terms in Block 1 and Block 2 are "independent" in the W dimension.
        # Thus, we cannot easily merge terms across blocks unless we introduce cross-terms.
        # So rank 2 * best_rank_m233 is a solid upper bound.
        
        rank_m236 = 2 * best_rank_m233
        
        U_comb = np.zeros((6, rank_m236))
        V_comb = np.zeros((18, rank_m236))
        W_comb = np.zeros((12, rank_m236))
        
        U_comb[:, :best_rank_m233] = best_U
        U_comb[:, best_rank_m233:] = best_U
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :best_rank_m233] = best_V[j * 3 + k]
                V_comb[j * 6 + 3 + k, best_rank_m233:] = best_V[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :best_rank_m233] = best_W[k * 2 + i]
                W_comb[(k + 3) * 2 + i, best_rank_m233:] = best_W[k * 2 + i]
                
        # Verify final solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
        # Fallback to rank 34 if verification fails (should not happen)
        U0, V0, W0 = np.zeros((6, 34)), np.zeros((18, 34)), np.zeros((12, 34))
        # Reconstruct rank 34 from seed
        U0[:, :17] = U233_seed
        U0[:, 17:] = U233_seed
        for j in range(3):
            for k in range(3):
                V0[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V0[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W0[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W0[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }

    # Execute search
    sol = search_m233_rank(16, deadline)
    
    if sol[0] is not None:
        return sol
    
    # If search fails, return rank 34
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U0 = np.zeros((6, 34))
    V0 = np.zeros((18, 34))
    W0 = np.zeros((12, 34))
    
    U0[:, :17] = U233_seed
    U0[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V0[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V0[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W0[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W0[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }

if __name__ == "__main__":
    construct()
