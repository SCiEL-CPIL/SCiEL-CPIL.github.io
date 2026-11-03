import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k'
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
    best_U, best_V, best_W = U_curr, V_curr, W_curr

    def verify_decomposition(U, V, W):
        """Fast verification using einsum."""
        D = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(D, T, atol=1e-9)

    def is_rank_1_tensor(tens, tol=1e-9):
        """
        Check if a 3-way tensor of shape (I, J, K) is rank 1.
        Returns (True, u, v, w) if rank 1, else (False, None, None, None).
        u: shape (I,), v: shape (J,), w: shape (K,)
        """
        I, J, K = tens.shape
        
        # Matricize to (I, J*K)
        mat1 = tens.reshape(I, J * K)
        # SVD
        U_svd, s, Vt = np.linalg.svd(mat1, full_matrices=False)
        
        if s[1] > tol:
            return False, None, None, None
            
        # Extract u (first left singular vector)
        u = U_svd[:, 0]
        # Scale factor
        scale = s[0]
        
        # The right part is v * w^T (reshaped to J x K)
        # Vt[0, :] is the first right singular vector of shape (J*K,)
        right_vec = Vt[0, :] / scale # Wait, T = u * s * v^T. So T = u * (s * v^T).
        # Actually, T = u * s * Vt[0]. 
        # Let M = tens. M = u * (s * Vt[0]).reshape(J, K)
        M = (s[0] * Vt[0, :]).reshape(J, K)
        
        # Check if M is rank 1
        U_m, s_m, Vt_m = np.linalg.svd(M, full_matrices=False)
        if s_m[1] > tol:
            return False, None, None, None
            
        v = U_m[:, 0]
        w = Vt_m[0, :]
        
        # Reconstruct to check scaling
        # M = v * (s_m[0] * w)
        # T = u * M = u * v * (s_m[0] * w)
        # So u_final = u, v_final = v, w_final = s_m[0] * w
        # Check if u * v * w_final == T
        recon = np.einsum('i,j,k->ijk', u, v, s_m[0] * w)
        if not np.allclose(recon, tens, atol=tol):
             return False, None, None, None
             
        return True, u, v, s_m[0] * w

    # Search Loop: Try to reduce rank by 1 (remove 2, add 1)
    attempts = 0
    max_attempts = 2000 # Adjust based on speed
    
    while rank_curr > 20 and time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Pick 2 random distinct indices to remove
        idxs = list(range(rank_curr))
        random.shuffle(idxs)
        i_rem, j_rem = idxs[0], idxs[1]
        
        # Create new U, V, W without these columns
        # We want to compute Delta = T - sum_{k != i_rem, j_rem} U[:,k] V[:,k] W[:,k]
        # Instead of summing, we can compute the full sum and subtract the two terms.
        # D_full = einsum(U, V, W)
        # D_without = D_full - term_i - term_j
        
        # Calculate full tensor (expensive if done every time, but rank is small)
        # Optimization: Keep D_current updated? 
        # For now, compute D_without directly to avoid accumulation error.
        
        # Get the terms to remove
        u_i = U_curr[:, i_rem]
        v_i = V_curr[:, i_rem]
        w_i = W_curr[:, i_rem]
        term_i = np.einsum('a,b,c->abc', u_i, v_i, w_i)
        
        u_j = U_curr[:, j_rem]
        v_j = V_curr[:, j_rem]
        w_j = W_curr[:, j_rem]
        term_j = np.einsum('a,b,c->abc', u_j, v_j, w_j)
        
        # Compute sum of all other terms
        # This is O(Rank * Size). Rank ~36, Size ~1300. 36*1300 = 46k ops. Very fast.
        D_others = np.zeros_like(T)
        for k in range(rank_curr):
            if k == i_rem or k == j_rem:
                continue
            uk = U_curr[:, k]
            vk = V_curr[:, k]
            wk = W_curr[:, k]
            D_others += np.einsum('a,b,c->abc', uk, vk, wk)
            
        Delta = T - D_others
        
        # Check if Delta is rank 1
        is_r1, u_new, v_new, w_new = is_rank_1_tensor(Delta)
        
        if is_r1:
            # We found a candidate term (u_new, v_new, w_new)
            # We need to ensure the values are "nice" (integer/half-integer)
            # The problem statement says "integer/half-integer entries".
            # If u_new, v_new, w_new are rational, we can scale them.
            # However, the product u*v*w must match Delta exactly.
            # If we scale u by s, v by 1/s, w by 1, the product is invariant.
            # We need to find a scaling such that all entries are in {0, 0.5, 1, 1.5, ...}
            
            # Simple check: are the entries already close to half-integers?
            def is_half_integer(arr):
                # arr * 2 should be close to integer
                return np.allclose(arr * 2, np.round(arr * 2), atol=1e-6)
            
            if is_half_integer(u_new) and is_half_integer(v_new) and is_half_integer(w_new):
                # Construct new U, V, W
                # Remove cols i_rem, j_rem
                mask = [k for k in range(rank_curr) if k != i_rem and k != j_rem]
                U_new = U_curr[:, mask]
                V_new = V_curr[:, mask]
                W_new = W_curr[:, mask]
                
                # Add new col
                U_new = np.column_stack((U_new, u_new))
                V_new = np.column_stack((V_new, v_new))
                W_new = np.column_stack((W_new, w_new))
                
                new_rank = rank_curr - 1
                
                # Verify
                if verify_decomposition(U_new, V_new, W_new):
                    U_curr = U_new
                    V_curr = V_new
                    W_curr = W_new
                    rank_curr = new_rank
                    if rank_curr < best_rank:
                        best_rank = rank_curr
                        best_U, best_V, best_W = U_curr, V_curr, W_curr
                    # Continue search with new rank
                    continue 
            else:
                # Try to scale to half-integers?
                # This is a harder Diophantine problem. Skip for now.
                pass

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
