import numpy as np
import time
import itertools
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Target T[a,b,c,d,i,k] = 1 if a==i, b==d, c==k else 0
    """
    r = U.shape[1]
    # Full reconstruction check
    T_recon = np.zeros((n, m, m, p, n, p))
    for l in range(r):
        u_l = U[:, l].reshape(n, m)
        v_l = V[:, l].reshape(m, p)
        w_l = W[:, l].reshape(p, n)
        # Tensor term: u_l[a,b] * v_l[c,d] * w_l[k,i]
        # Note: W index in target is W[k*n+i, l] based on problem statement W[k*n+i,r]
        # So w_l is reshaped to (p, n) where row is k, col is i.
        # T_recon[a, b, c, d, i, k] += U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
        term = np.einsum('ab,cd,ki->abcdki', u_l, v_l, w_l)
        T_recon += term
        
    # Target tensor
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                # Non-zero at d=b, i=a, k=c
                T_target[a, b, c, b, a, c] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Dimensions of factor matrices
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # Schoolbook rank is n*m*p = 36.
    
    # Strategy: 
    # 1. Try to find a rank 33, 32, ... solution using a "structured" search.
    # The inspiration suggests using field extensions. 
    # Specifically, p=6 = 3*2. 
    # If we can find a decomposition of 2x3x3 of rank r_33, 
    # can we combine two such decompositions to get 2x3x6?
    # The current seed uses two copies of rank 17 (total 34).
    # If we can find a rank 16 decomposition for 2x3x3, total rank 32.
    # If we can find a rank 15, total 30.
    # Known: Strassen rank for 3x3x3 is 19? No, 3x3x3 rank is 19? 
    # Actually, rank of 3x3x3 is 19? No, it's 19 for 3x3x3? 
    # Wait, Strassen is 7 for 2x2x2. 
    # For 3x3x3, the best known is 19? Or 22? 
    # Actually, the rank of 3x3x3 matrix multiplication is 19? 
    # Let's check: Strassen's algorithm generalizes. 
    # However, the seed program uses rank 17 for 2x3x3. 
    # Note: 2x3x3 is NOT 3x3x3. 
    # 2x3x3 tensor has dimensions n=2, m=3, p=3. 
    # Rank of 2x3x3: 
    # Schoolbook is 2*3*3 = 18.
    # The seed found 17. 
    # Can we find 16? 15?
    
    # Let's try a local search to improve the 2x3x3 rank from 17 to 16.
    # If we find 16, we can combine two copies to get rank 32 for 2x3x6.
    # Score would be 36/32 = 1.125. (Current is 36/34 = 1.0588).
    
    # If we find 15, rank 30. Score 36/30 = 1.2.
    
    # Let's implement a search for 2x3x3 rank 16.
    
    def find_2x3x3_rank(target_rank):
        """
        Finds a rank decomposition for 2x3x3.
        Returns U, V, W or None.
        """
        n, m, p = 2, 3, 3
        U_dim, V_dim, W_dim = n*m, m*p, p*n
        # U: (6, r), V: (9, r), W: (6, r)
        
        # Target tensor T[a,b,c,d,i,k] = delta_{a,i} delta_{b,d} delta_{c,k}
        # Non-zeros: a in [0,2), b in [0,3), c in [0,3).
        # i=a, d=b, k=c.
        # 18 non-zero entries.
        
        # We use a random restart gradient descent with a "snap to integer" heuristic.
        # Since the solution likely has integer/half-integer entries, we can search in real space
        # and then round.
        
        for restart in range(500):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(U_dim, target_rank)
            V = np.random.randn(V_dim, target_rank)
            W = np.random.randn(W_dim, target_rank)
            
            # Normalize to avoid scaling issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Gradient Descent
            # Error E = sum_{nonzero} (sum_l U_ab,l V_cb,l W_ca,l - 1)^2 + sum_{zero} (sum_l ...)^2
            # We'll use a subset of zeros to keep it fast.
            
            # Precompute non-zero indices
            nz_indices = []
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        nz_indices.append((a, b, c, b, a, c))
                        
            # Precompute a fixed set of zero indices (sample)
            all_indices = set(itertools.product(range(n), range(m), range(m), range(p), range(n), range(p)))
            nz_set = set(nz_indices)
            zero_indices = list(all_indices - nz_set)
            # Shuffle and take first 100
            np.random.shuffle(zero_indices)
            zero_sample = zero_indices[:100]
            
            def compute_grad(U, V, W):
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                # For each term l, the contribution to error is quadratic.
                # Let S_abcdki = sum_l U_ab,l V_cd,l W_ki,l
                # Target T_abcdki.
                # dE/dU_ab,l = 2 * (S_abcdki - T_abcdki) * V_cd,l * W_ki,l
                # We only need to compute this for non-zero and sampled zero indices.
                
                # Efficiently:
                # For each index in (nz + zero_sample):
                #   idx_u, idx_v, idx_w
                #   val = dot(U[idx_u], V[idx_v] * W[idx_w])
                #   err_val = val - target
                #   gU[idx_u] += 2 * err_val * (V[idx_v] * W[idx_w])
                #   gV[idx_v] += 2 * err_val * (U[idx_u] * W[idx_w])
                #   gW[idx_w] += 2 * err_val * (U[idx_u] * V[idx_v])
                
                # This is O(N_sample * r).
                # N_sample ~ 118. r ~ 16. 118*16 = 1888 ops. Very fast.
                
                for (a, b, c, d, i, k) in nz_indices:
                    idx_u = a * m + b
                    idx_v = c * p + d
                    idx_w = k * n + i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    err_val = val - 1.0
                    gU[idx_u] += 2 * err_val * (V[idx_v] * W[idx_w])
                    gV[idx_v] += 2 * err_val * (U[idx_u] * W[idx_w])
                    gW[idx_w] += 2 * err_val * (U[idx_u] * V[idx_v])
                    
                for (a, b, c, d, i, k) in zero_sample:
                    idx_u = a * m + b
                    idx_v = c * p + d
                    idx_w = k * n + i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    err_val = val
                    gU[idx_u] += 2 * err_val * (V[idx_v] * W[idx_w])
                    gV[idx_v] += 2 * err_val * (U[idx_u] * W[idx_w])
                    gW[idx_w] += 2 * err_val * (U[idx_u] * V[idx_v])
                    
                return gU, gV, gW

            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    return None
                
                gU, gV, gW = compute_grad(U, V, W)
                
                # Adaptive learning rate?
                # Simple momentum
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Check error occasionally
                if step % 100 == 0:
                    # Full check
                    T_recon = np.zeros((n, m, m, p, n, p))
                    for l in range(target_rank):
                        u_l = U[:, l].reshape(n, m)
                        v_l = V[:, l].reshape(m, p)
                        w_l = W[:, l].reshape(p, n)
                        T_recon += np.einsum('ab,cd,ki->abcdki', u_l, v_l, w_l)
                    
                    T_target = np.zeros((n, m, m, p, n, p))
                    for a in range(n):
                        for b in range(m):
                            for c in range(m):
                                T_target[a, b, c, b, a, c] = 1.0
                                
                    err = np.linalg.norm(T_recon - T_target)
                    if err < 1e-6:
                        # Snap to integers/half-integers
                        # Try rounding to nearest 0.5
                        U_int = np.round(U * 2) / 2
                        V_int = np.round(V * 2) / 2
                        W_int = np.round(W * 2) / 2
                        
                        # Verify snapped
                        if check_decomposition(U_int, V_int, W_int, n, m, p):
                            return U_int, V_int, W_int
                        # Try rounding to integers
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if check_decomposition(U_int, V_int, W_int, n, m, p):
                            return U_int, V_int, W_int

            # If GD didn't converge, try a different init?
            # Or accept if error is low enough and snap?
            
        return None

    # Fallback Rank 34 Solution
    def get_rank_34_solution():
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        # Rank 17 decomposition for 2x3x3
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
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    U_fallback, V_fallback, W_fallback = get_rank_34_solution()
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = 34

    # Try to find better 2x3x3 ranks
    # If we find rank 16, we can build rank 32.
    # If we find rank 15, we can build rank 30.
    
    found_16 = find_2x3x3_rank(16)
    if found_16:
        U16, V16, W16 = found_16
        # Build rank 32 solution for 2x3x6
        U32 = np.zeros((6, 32))
        V32 = np.zeros((18, 32))
        W32 = np.zeros((12, 32))
        
        U32[:, :16] = U16
        U32[:, 16:] = U16
        
        for j in range(3):
            for k in range(3):
                V32[j * 6 + k, :16] = V16[j * 3 + k]
                V32[j * 6 + 3 + k, 16:] = V16[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W32[k * 2 + i, :16] = W16[k * 2 + i]
                W32[(k + 3) * 2 + i, 16:] = W16[k * 2 + i]
                
        if check_decomposition(U32, V32, W32, n, m, p):
            best_U, best_V, best_W = U32, V32, W32
            best_rank = 32

    if best_rank > 32:
        found_15 = find_2x3x3_rank(15)
        if found_15:
            U15, V15, W15 = found_15
            U30 = np.zeros((6, 30))
            V30 = np.zeros((18, 30))
            W30 = np.zeros((12, 30))
            
            U30[:, :15] = U15
            U30[:, 15:] = U15
            
            for j in range(3):
                for k in range(3):
                    V30[j * 6 + k, :15] = V15[j * 3 + k]
                    V30[j * 6 + 3 + k, 15:] = V15[j * 3 + k]
                    
            for k in range(3):
                for i in range(2):
                    W30[k * 2 + i, :15] = W15[k * 2 + i]
                    W30[(k + 3) * 2 + i, 15:] = W15[k * 2 + i]
                    
            if check_decomposition(U30, V30, W30, n, m, p):
                best_U, best_V, best_W = U30, V30, W30
                best_rank = 30

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    construct()
