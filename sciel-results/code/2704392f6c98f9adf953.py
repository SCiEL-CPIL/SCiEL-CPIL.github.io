import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook Decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_curr, V_curr, W_curr = schoolbook()
    rank_curr = U_curr.shape[1]
    best_rank = rank_curr
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()

    def verify_decomposition(U, V, W):
        """Fast verification using einsum."""
        D = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(D, T, atol=1e-9)

    def is_rank_1_tensor(tens, tol=1e-8):
        """
        Check if a 3-way tensor of shape (I, J, K) is rank 1.
        Uses all three matricizations for robustness.
        Returns (True, u, v, w) if rank 1, else (False, None, None, None).
        """
        I, J, K = tens.shape
        
        # Check mode-1 matricization: (I, J*K)
        mat1 = tens.reshape(I, J * K)
        U_svd, s, Vt = np.linalg.svd(mat1, full_matrices=False)
        
        if s[1] > tol:
            return False, None, None, None
            
        # Extract u (first left singular vector)
        u = U_svd[:, 0]
        scale = s[0]
        
        # The right part reshaped to (J, K)
        M = (scale * Vt[0, :]).reshape(J, K)
        
        # Check if M is rank 1
        U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
        if s_m[1] > tol:
            return False, None, None, None
            
        v = U_m[:, 0]
        w = Vt_m[0, :]
        
        # Reconstruct to verify
        recon = np.einsum('i,j,k->ijk', u, v, s_m[0] * w)
        if not np.allclose(recon, tens, atol=tol):
            return False, None, None, None
            
        # Additional check: mode-2 matricization should also be rank 1
        mat2 = tens.transpose(1, 0, 2).reshape(J, I * K)
        s2 = np.linalg.svd(mat2, full_matrices=False, compute_uv=False)
        if s2[1] > tol:
            return False, None, None, None
            
        # Mode-3 matricization
        mat3 = tens.transpose(2, 0, 1).reshape(K, I * J)
        s3 = np.linalg.svd(mat3, full_matrices=False, compute_uv=False)
        if s3[1] > tol:
            return False, None, None, None
            
        return True, u, v, s_m[0] * w

    def is_half_integer(arr, atol=1e-6):
        """Check if all entries are close to half-integers."""
        return np.allclose(arr * 2, np.round(arr * 2), atol=atol)

    def try_scale_to_half_int(u, v, w):
        """
        Try to find scalings such that all entries become half-integers.
        We can scale u by alpha, v by 1/alpha, w unchanged (product invariant).
        Also try other combinations.
        """
        # Try simple scaling factors
        for alpha in [1.0, 0.5, 2.0, 0.25, 4.0, 1.0/3.0, 3.0, 0.5/3.0, 1.5]:
            u_s = u * alpha
            v_s = v / alpha
            if is_half_integer(u_s) and is_half_integer(v_s) and is_half_integer(w):
                return u_s, v_s, w
        
        # Try scaling w instead
        for beta in [1.0, 0.5, 2.0, 0.25, 4.0, 1.0/3.0, 3.0]:
            w_s = w * beta
            u_s = u / beta
            if is_half_integer(u_s) and is_half_integer(v) and is_half_integer(w_s):
                return u_s, v, w_s
        
        # Try scaling v
        for gamma in [1.0, 0.5, 2.0, 0.25, 4.0, 1.0/3.0, 3.0]:
            v_s = v * gamma
            w_s = w / gamma
            if is_half_integer(u) and is_half_integer(v_s) and is_half_integer(w_s):
                return u, v_s, w_s
        
        return None, None, None

    # Precompute D_full = sum of all rank-1 terms (initially T itself)
    D_current = T.copy()
    
    # Search Loop: Try to reduce rank by 1 (remove 2, add 1)
    attempts = 0
    max_attempts = 5000  # Increased due to faster per-iteration cost
    pairs_per_attempt = 3  # Try multiple pairs per rank level
    
    while rank_curr > 20 and time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Pick several random pairs to try
        idxs = list(range(rank_curr))
        random.shuffle(idxs)
        
        found = False
        for pair_idx in range(min(pairs_per_attempt, rank_curr - 1)):
            if time.time() >= deadline:
                break
                
            i_rem = idxs[pair_idx]
            j_rem = idxs[(pair_idx + 1) % rank_curr]
            if i_rem == j_rem:
                j_rem = idxs[(pair_idx + 2) % rank_curr]
                if i_rem == j_rem:
                    continue
            
            # Compute D_others = D_current - term_i - term_j
            # This is much faster than looping over all terms
            u_i = U_curr[:, i_rem]
            v_i = V_curr[:, i_rem]
            w_i = W_curr[:, i_rem]
            term_i = np.einsum('a,b,c->abc', u_i, v_i, w_i)
            
            u_j = U_curr[:, j_rem]
            v_j = V_curr[:, j_rem]
            w_j = W_curr[:, j_rem]
            term_j = np.einsum('a,b,c->abc', u_j, v_j, w_j)
            
            D_others = D_current - term_i - term_j
            Delta = T - D_others
            
            # Check if Delta is rank 1
            is_r1, u_new, v_new, w_new = is_rank_1_tensor(Delta)
            
            if is_r1:
                # Try to get half-integer entries
                u_h, v_h, w_h = try_scale_to_half_int(u_new, v_new, w_new)
                
                if u_h is None:
                    # Check if already half-integer
                    if is_half_integer(u_new) and is_half_integer(v_new) and is_half_integer(w_new):
                        u_h, v_h, w_h = u_new, v_new, w_new
                    else:
                        continue
                
                # Construct new U, V, W
                mask = [k for k in range(rank_curr) if k != i_rem and k != j_rem]
                U_new = np.column_stack((U_curr[:, mask], u_h))
                V_new = np.column_stack((V_curr[:, mask], v_h))
                W_new = np.column_stack((W_curr[:, mask], w_h))
                
                new_rank = rank_curr - 1
                
                # Verify
                if verify_decomposition(U_new, V_new, W_new):
                    # Update D_current
                    D_current = D_current - term_i - term_j + np.einsum('a,b,c->abc', u_h, v_h, w_h)
                    
                    U_curr = U_new
                    V_curr = V_new
                    W_curr = W_new
                    rank_curr = new_rank
                    if rank_curr < best_rank:
                        best_rank = rank_curr
                        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    found = True
                    break
        
        if not found and rank_curr > 20:
            # If we tried multiple pairs and none worked, continue to next attempt
            # This is already handled by the outer loop
            continue

    # Convert to lists for output
    def to_list(arr):
        return arr.tolist()
        
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
