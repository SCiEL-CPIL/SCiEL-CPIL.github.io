"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    def verify(T):
        """Check if tensor T is all ones."""
        return np.allclose(T, 1.0, atol=1e-9)
    
    def compute_tensor(U, V, W):
        """Compute the resulting tensor from factor matrices."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return T
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W
    
    U, V, W = schoolbook()
    rank = U.shape[1]
    
    # Build initial Strassen-based decomposition for a better starting point
    def build_strassen_decomp():
        """Build decomposition using Strassen on 2x2x2 blocks where possible."""
        U_list = []
        V_list = []
        W_list = []
        
        def add_term(u, v, w):
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
        
        def add_strassen(i0, i1, j0, j1, k0, k1):
            # M1 = (A00+A11)(B00+B11)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i0*m+j0] = 1; u[i1*m+j1] = 1
            v[j0*p+k0] = 1; v[j1*p+k1] = 1
            w[k0*n+i0] = 1; w[k1*n+i1] = 1
            add_term(u, v, w)
            
            # M2 = (A10+A11)B00
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i1*m+j0] = 1; u[i1*m+j1] = 1
            v[j0*p+k0] = 1
            w[k0*n+i1] = 1; w[k1*n+i1] = -1
            add_term(u, v, w)
            
            # M3 = A00(B01-B11)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i0*m+j0] = 1
            v[j0*p+k1] = 1; v[j1*p+k1] = -1
            w[k1*n+i0] = 1; w[k1*n+i1] = 1
            add_term(u, v, w)
            
            # M4 = A11(B10-B00)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i1*m+j1] = 1
            v[j1*p+k0] = 1; v[j0*p+k0] = -1
            w[k0*n+i0] = 1; w[k0*n+i1] = 1
            add_term(u, v, w)
            
            # M5 = (A00+A01)B11
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i0*m+j0] = 1; u[i0*m+j1] = 1
            v[j1*p+k1] = 1
            w[k1*n+i0] = 1; w[k0*n+i0] = -1
            add_term(u, v, w)
            
            # M6 = (A10-A00)(B00+B01)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i1*m+j0] = 1; u[i0*m+j0] = -1
            v[j0*p+k0] = 1; v[j0*p+k1] = 1
            w[k1*n+i1] = 1
            add_term(u, v, w)
            
            # M7 = (A01-A11)(B10+B11)
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i0*m+j1] = 1; u[i1*m+j1] = -1
            v[j1*p+k0] = 1; v[j1*p+k1] = 1
            w[k0*n+i0] = 1
            add_term(u, v, w)
        
        def add_schoolbook(i, j, k):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            u[i*m+j] = 1
            v[j*p+k] = 1
            w[k*n+i] = 1
            add_term(u, v, w)
        
        # Try all possible 2x2x2 blocks
        i_pairs = [(0,1), (1,2)]
        j_pairs = [(0,1), (1,2), (2,3), (3,4)]
        k_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)]
        
        # Greedy: pick non-overlapping 2x2x2 blocks
        used_i = set()
        used_j = set()
        used_k = set()
        blocks = []
        
        # Sort by potential savings (all same) but try to maximize coverage
        random.seed(42)
        all_blocks = [(i0,i1,j0,j1,k0,k1) 
                      for i0,i1 in i_pairs 
                      for j0,j1 in j_pairs 
                      for k0,k1 in k_pairs]
        random.shuffle(all_blocks)
        
        for (i0,i1,j0,j1,k0,k1) in all_blocks:
            if i0 in used_i or i1 in used_i: continue
            if j0 in used_j or j1 in used_j: continue
            if k0 in used_k or k1 in used_k: continue
            used_i.add(i0); used_i.add(i1)
            used_j.add(j0); used_j.add(j1)
            used_k.add(k0); used_k.add(k1)
            blocks.append((i0,i1,j0,j1,k0,k1))
        
        for (i0,i1,j0,j1,k0,k1) in blocks:
            add_strassen(i0,i1,j0,j1,k0,k1)
        
        # Fill remaining with schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check if this (i,j,k) is covered by a Strassen block
                    covered = False
                    for (i0,i1,j0,j1,k0,k1) in blocks:
                        if i in (i0,i1) and j in (j0,j1) and k in (k0,k1):
                            covered = True
                            break
                    if not covered:
                        add_schoolbook(i,j,k)
        
        U_mat = np.array(U_list).T
        V_mat = np.array(V_list).T
        W_mat = np.array(W_list).T
        return U_mat, V_mat, W_mat
    
    U, V, W = build_strassen_decomp()
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Verify initial
    T = compute_tensor(U, V, W)
    if not verify(T):
        # Fallback to schoolbook
        U, V, W = schoolbook()
        rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank
        T = compute_tensor(U, V, W)
    
    # Randomized rank reduction
    # Strategy: try to merge pairs of terms into one
    # For terms a and b, try U_new = alpha*U_a + beta*U_b, etc.
    # This is expensive, so use a smarter approach:
    
    # Alternative: try removing one term at a time (greedy)
    # Then try "swap" operations: replace term c with a linear combo of others
    
    # First: greedy single-term removal
    def try_remove(U, V, W, idx):
        """Try removing term idx. Returns (success, new_U, new_V, new_W)."""
        rank = U.shape[1]
        if rank <= 1:
            return False, U, V, W
        U_new = np.delete(U, idx, axis=1)
        V_new = np.delete(V, idx, axis=1)
        W_new = np.delete(W, idx, axis=1)
        T_new = compute_tensor(U_new, V_new, W_new)
        if verify(T_new):
            return True, U_new, V_new, W_new
        return False, U, V, W
    
    # Greedy removal pass
    progress = True
    while progress and time.time() < deadline - 20:
        progress = False
        for idx in range(U.shape[1]):
            if time.time() >= deadline - 20:
                break
            success, U_new, V_new, W_new = try_remove(U, V, W, idx)
            if success:
                U, V, W = U_new, V_new, W_new
                progress = True
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = U.shape[1]
                break
    
    # Now try random pair-merging: replace two terms with one
    # For terms i and j, try U_new_c = a*U_i + b*U_j, V_new_c = a*V_i + b*V_j, W_new_c = a*W_i + b*W_j
    # This only works if the cross terms cancel out, which is very restrictive.
    # Better: try random linear combinations of ALL remaining terms to find a lower-rank equivalent.
    
    # Actually, let's try a different approach: random perturbation + projection
    # Or: try to find if any term is a linear combination of others (in the tensor sense)
    
    # Simpler and more effective: try all pairs (i,j) and see if we can replace them
    # with a single term by solving a system.
    # For terms i and j, we want to find a single term c such that:
    # U_i * V_i * W_i + U_j * V_j * W_j = U_c * V_c * W_c (as tensors)
    # This is very restrictive and rarely works.
    
    # Instead, let's use a more powerful technique: 
    # Randomized SVD-like approach on the tensor
    # Or: try to find a basis change that reveals redundancies
    
    # Let's try: for each pair of terms, try to eliminate one by expressing it
    # as a combination of others. This requires solving a linear system.
    
    # Actually, the most effective approach for small ranks:
    # Try all subsets of terms to remove (up to size 2-3) and check validity.
    
    # For efficiency, let's do random sampling of pairs to remove
    random.seed(123)
    max_rank = U.shape[1]
    
    for attempt in range(200000):
        if time.time() >= deadline - 15:
            break
        
        # Try removing a random single term
        idx = random.randint(0, U.shape[1] - 1)
        success, U_new, V_new, W_new = try_remove(U, V, W, idx)
        if success:
            U, V, W = U_new, V_new, W_new
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Try removing pairs of random terms
    for attempt in range(50000):
        if time.time() >= deadline - 10:
            break
        if U.shape[1] <= 1:
            break
        i = random.randint(0, U.shape[1] - 1)
        j = random.randint(0, U.shape[1] - 1)
        if i == j:
            continue
        # Try removing both i and j
        if i > j:
            i, j = j, i
        U_new = np.delete(np.delete(U, i, axis=1), j-1, axis=1)
        V_new = np.delete(np.delete(V, i, axis=1), j-1, axis=1)
        W_new = np.delete(np.delete(W, i, axis=1), j-1, axis=1)
        T_new = compute_tensor(U_new, V_new, W_new)
        if verify(T_new):
            U, V, W = U_new, V_new, W_new
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Final: try to add random terms and then remove more (explore higher rank temporarily)
    # This is a local search: add a random term, then greedily remove terms
    
    for attempt in range(10000):
        if time.time() >= deadline - 8:
            break
        if U.shape[1] >= 105:
            continue
        
        # Generate a random sparse term
        u_rand = np.zeros(n*m)
        v_rand = np.zeros(m*p)
        w_rand = np.zeros(p*n)
        
        # Randomly set a few entries
        for _ in range(random.randint(1, 4)):
            u_rand[random.randint(0, n*m-1)] = random.choice([-1, 1])
        for _ in range(random.randint(1, 4)):
            v_rand[random.randint(0, m*p-1)] = random.choice([-1, 1])
        for _ in range(random.randint(1, 4)):
            w_rand[random.randint(0, p*n-1)] = random.choice([-1, 1])
        
        # Add this term
        U_ext = np.column_stack([U, u_rand])
        V_ext = np.column_stack([V, v_rand])
        W_ext = np.column_stack([W, w_rand])
        
        # Now greedily remove terms from the extended set
        U_cur, V_cur, W_cur = U_ext, V_ext, W_ext
        progress = True
        while progress and time.time() < deadline - 8:
            progress = False
            for idx in range(U_cur.shape[1]):
                if time.time() >= deadline - 8:
                    break
                success, U_new, V_new, W_new = try_remove(U_cur, V_cur, W_cur, idx)
                if success:
                    U_cur, V_cur, W_cur = U_new, V_new, W_new
                    progress = True
                    break
        
        # Accept if we ended up with fewer terms
        if U_cur.shape[1] < U.shape[1]:
            U, V, W = U_cur, V_cur, W_cur
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END

