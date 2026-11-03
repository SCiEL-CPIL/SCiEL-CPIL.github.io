"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # Initial permutation: Random or Rank-matching?
    # Rank matching is a good start.
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute W = a + a.T
    W = a + a.T
    
    # Initialize B_perm: B_perm[f, pos] = b[f, p[pos]]
    # b is (n, n). p is (n,).
    # We want a matrix where column pos is b[:, p[pos]].
    B_perm = b[:, p]
    
    # Initialize D = W @ B_perm
    D = W @ B_perm
    
    best_delta = 0
    best_i = -1
    best_j = -1
    
    # Precompute indices for pairs
    # We will iterate i, j
    
    # Helper to compute corrections for a pair (i, j)
    # This is O(1) per pair if we access arrays
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    while time.time() - start_time < timeout:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Iterate over all pairs i < j
        # To speed up, we can vectorize the delta calculation?
        # Delta(i, j) = (D[i, j] - D[i, i]) + (D[j, i] - D[j, j]) - C_dots + A_pairs
        
        # Let's compute the main part M[i, j] = D[i, j] + D[j, i] - D[i, i] - D[j, j]
        # M is not symmetric in indices i,j of the matrix, but the value for pair {i,j} is M[i,j].
        # We want min over i < j.
        
        # M[i, j] = D[i, j] + D[j, i] - D[i, i] - D[j, j]
        
        # We can construct the matrix M for all i, j.
        # M = D + D.T - D.diagonal()[:, None] - D.diagonal()[None, :]
        
        M = D + D.T - D.diagonal()[:, None] - D.diagonal()[None, :]
        
        # Now we need to add corrections C_dots and A_pairs.
        # These depend on p[i], p[j].
        # f_i = p[i], f_j = p[j].
        
        # C_dots = w_i[i] d_i[i] + w_i[j] d_i[j] + w_j[i] d_j[i] + w_j[j] d_j[j]
        # A_pairs = a[i,i](b[f_j,f_j]-b[f_i,f_i]) + a[i,j](b[f_j,f_i]-b[f_i,f_j]) + a[j,i](b[f_i,f_j]-b[f_j,f_i]) + a[j,j](b[f_i,f_i]-b[f_j,f_j])
        
        # Note: w_i = W[i, :]
        
        # This part is hard to vectorize completely without n^3 memory or complex indexing.
        # However, n=256, so n^2 = 65k pairs.
        # We can loop i, j in Python? 65k iterations is fast.
        
        # Let's loop i, j to find the best.
        
        # Optimization: Only check upper triangle.
        
        # Pre-extract necessary values
        # b_fi_fi = b[p[i], p[i]] etc.
        
        # We can compute the correction matrix Corr[i, j]
        
        # Let's try a simple loop first to ensure correctness, then optimize if needed.
        
        # Vectorized correction calculation:
        # f_i = p[i]
        # f_j = p[j]
        
        # b_ii = b[p, p] -> vector of length n? No, b[p[i], p[i]] is scalar.
        # Let's create arrays:
        # b_pp = b[p, p] # b[p[i], p[i]]
        # b_pj = b[p[i], p[j]] # This is a matrix B_pair[i, j] = b[p[i], p[j]]
        
        B_pair = b[p, p] # Matrix where B_pair[i, j] = b[p[i], p[j]]
        
        # d_i[i] = b[f_j, f_i] - b[f_i, f_i] = B_pair[j, i] - B_pair[i, i]
        # d_i[j] = b[f_j, f_j] - b[f_i, f_j] = B_pair[j, j] - B_pair[i, j]
        # d_j[i] = b[f_i, f_i] - b[f_j, f_i] = B_pair[i, i] - B_pair[j, i]
        # d_j[j] = b[f_i, f_j] - b[f_j, f_j] = B_pair[i, j] - B_pair[j, j]
        
        # w_i[i] = W[i, i]
        # w_i[j] = W[i, j]
        # w_j[i] = W[j, i]
        # w_j[j] = W[j, j]
        
        # C_dots[i, j] = W[i, i] * (B_pair[j, i] - B_pair[i, i]) + W[i, j] * (B_pair[j, j] - B_pair[i, j]) + W[j, i] * (B_pair[i, i] - B_pair[j, i]) + W[j, j] * (B_pair[i, j] - B_pair[j, j])
        
        # A_pairs[i, j] = a[i, i] * (B_pair[j, j] - B_pair[i, i]) + a[i, j] * (B_pair[j, i] - B_pair[i, j]) + a[j, i] * (B_pair[i, j] - B_pair[j, i]) + a[j, j] * (B_pair[i, i] - B_pair[j, j])
        
        # Note: B_pair[j, i] = B_pair[i, j] because b is symmetric.
        # So B_pair[j, i] - B_pair[i, j] = 0.
        # Thus, the middle terms in A_pairs vanish if b is symmetric.
        # And in C_dots:
        # W[i, i] * (B_pair[i, j] - B_pair[i, i]) + W[i, j] * (B_pair[j, j] - B_pair[i, j]) + W[j, i] * (B_pair[i, i] - B_pair[i, j]) + W[j, j] * (B_pair[i, j] - B_pair[j, j])
        
        # Let's compute these matrices.
        
        # B_pair is symmetric.
        
        # Term 1 of C: W[i, i] * (B_pair[j, i] - B_pair[i, i])
        # We need W_diag = W.diagonal()
        # T1[i, j] = W_diag[i] * (B_pair[j, i] - B_pair[i, i])
        
        # Term 2 of C: W[i, j] * (B_pair[j, j] - B_pair[i, j])
        # T2[i, j] = W[i, j] * (B_pair[j, j] - B_pair[i, j])
        
        # Term 3 of C: W[j, i] * (B_pair[i, i] - B_pair[j, i])
        # T3[i, j] = W[j, i] * (B_pair[i, i] - B_pair[j, i])
        
        # Term 4 of C: W[j, j] * (B_pair[i, j] - B_pair[j, j])
        # T4[i, j] = W_diag[j] * (B_pair[i, j] - B_pair[j, j])
        
        # C_dots = T1 + T2 + T3 + T4
        
        # A_pairs:
        # Term 1: a[i, i] * (B_pair[j, j] - B_pair[i, i])
        # A_diag = a.diagonal()
        # T5[i, j] = A_diag[i] * (B_pair[j, j] - B_pair[i, i])
        
        # Term 2: a[i, j] * 0 = 0
        # Term 3: a[j, i] * 0 = 0
        # Term 4: a[j, j] * (B_pair[i, i] - B_pair[j, j])
        # T6[i, j] = A_diag[j] * (B_pair[i, i] - B_pair[j, j])
        
        # A_pairs = T5 + T6
        
        # Correction = -C_dots + A_pairs
        
        # Total Delta[i, j] = M[i, j] + Correction[i, j]
        
        # We want min Delta[i, j] for i < j.
        
        # Let's compute these arrays.
        
        W_diag = W.diagonal()
        A_diag = a.diagonal()
        
        # B_pair[i, i] is vector B_diag = B_pair.diagonal()
        B_diag = B_pair.diagonal()
        
        # T1: W_diag[:, None] * (B_pair.T - B_diag[:, None])
        # B_pair[j, i] is B_pair.T[i, j]
        T1 = W_diag[:, None] * (B_pair.T - B_diag[:, None])
        
        # T2: W * (B_diag[None, :] - B_pair)
        # B_pair[j, j] is B_diag[j]. B_pair[i, j] is B_pair[i, j].
        T2 = W * (B_diag[None, :] - B_pair)
        
        # T3: W.T * (B_diag[:, None] - B_pair)
        # W[j, i] is W.T[i, j]. B_pair[i, i] is B_diag[i]. B_pair[j, i] is B_pair.T[i, j].
        T3 = W.T * (B_diag[:, None] - B_pair.T)
        
        # T4: W_diag[None, :] * (B_pair - B_diag[None, :])
        T4 = W_diag[None, :] * (B_pair - B_diag[None, :])
        
        C_dots = T1 + T2 + T3 + T4
        
        # T5: A_diag[:, None] * (B_diag[None, :] - B_diag[:, None])
        T5 = A_diag[:, None] * (B_diag[None, :] - B_diag[:, None])
        
        # T6: A_diag[None, :] * (B_diag[:, None] - B_diag[None, :])
        T6 = A_diag[None, :] * (B_diag[:, None] - B_diag[None, :])
        
        A_pairs = T5 + T6
        
        Correction = -C_dots + A_pairs
        
        Delta = M + Correction
        
        # Set diagonal and lower triangle to inf to ignore
        Delta = np.triu(Delta, k=1)
        Delta = np.where(np.isinf(Delta), np.inf, Delta)
        
        # Find min
        min_val = np.min(Delta)
        
        if min_val >= 0:
            break
        
        # Find index
        idx = np.unravel_index(np.argmin(Delta), Delta.shape)
        i, j = idx
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        
        # Update B_perm
        old_bi = b[:, p[i]]
        old_bj = b[:, p[j]]
        B_perm[:, i] = old_bj
        B_perm[:, j] = old_bi
        
        # Update D
        # Delta B has columns i and j changed.
        # Delta B[:, i] = old_bj - B_perm[:, i] (before update) ?
        # Actually, we just swapped.
        # New B_perm[:, i] = old_bj
        # New B_perm[:, j] = old_bi
        # Old B_perm[:, i] = old_bi
        # Old B_perm[:, j] = old_bj
        
        # Delta B[:, i] = old_bj - old_bi
        # Delta B[:, j] = old_bi - old_bj = - Delta B[:, i]
        
        dBi = old_bj - old_bi
        dBj = -dBi
        
        # Delta D = W[:, i, None] * dBi[None, :] + W[:, j, None] * dBj[None, :]
        # Wait, D = W @ B_perm.
        # D_new = D + W @ Delta B.
        # (W @ Delta B)[k, m] = W[k, i] Delta B[m, i] + W[k, j] Delta B[m, j]
        # Delta B[m, i] = dBi[m]
        # Delta B[m, j] = dBj[m]
        
        Delta_D = W[:, i, None] * dBi[None, :] + W[:, j, None] * dBj[None, :]
        D += Delta_D
        
    return p.tolist()

# EVOLVE-BLOCK-END
