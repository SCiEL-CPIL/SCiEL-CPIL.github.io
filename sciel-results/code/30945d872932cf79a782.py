import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 6, 6, 6
    bs = 2
    nb = n // bs  # 3
    R = nb ** 3 * 7  # 189

    # Pre-compute the 7 Strassen terms' local patterns for 2x2 blocks
    terms = []
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    ))
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1, (1,1):-1}
    ))
    terms.append((
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(0,1):1, (1,1):1}
    ))
    terms.append((
        {(1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,0):1, (1,0):1}
    ))
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,0):-1, (0,1):1}
    ))
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    ))
    terms.append((
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1}
    ))

    # Build U, V, W
    U = np.zeros((n*m, R), dtype=np.int64)
    V = np.zeros((m*p, R), dtype=np.int64)
    W = np.zeros((p*n, R), dtype=np.int64)

    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for (upat, vpat, wpat) in terms:
                    for (lr, lc), cv in upat.items():
                        U[(ib*bs+lr)*m + (kb*bs+lc), col] += cv
                    for (lr, lc), cv in vpat.items():
                        V[(kb*bs+lr)*p + (jb*bs+lc), col] += cv
                    for (lr, lc), cv in wpat.items():
                        W[(jb*bs+lc)*n + (ib*bs+lr), col] += cv
                    col += 1

    # Fast verification using numpy einsum
    def verify_fast(Um, Vm, Wm):
        U_r = Um.reshape(n, m, -1)
        V_r = Vm.reshape(m, p, -1)
        W_r = Wm.reshape(p, n, -1)
        T = np.einsum('ikt,kjt,jit->ijk', U_r, V_r, W_r)
        return np.all(T == 1)

    assert verify_fast(U, V, W), "Initial decomposition is invalid"

    # Vectorized greedy removal: for each column, check if removing it preserves the tensor
    # T[i,j,k] = sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]
    # For column t: contribution C_t[i,j,k] = U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]
    # We need: sum_{t' != t} C_{t'}[i,j,k] == 1 for all i,j,k
    # i.e., C_t[i,j,k] == sum_t C_t[i,j,k] - 1 = 1 - 1 = 0 ... no that's wrong
    # Actually: total = sum_t C_t = 1. So sum_{t'!=t} C_{t'} = 1 - C_t.
    # We need 1 - C_t[i,j,k] == 1 for all i,j,k, i.e., C_t[i,j,k] == 0 for all i,j,k.
    # So a column can be removed iff its contribution tensor is identically zero!

    # This is a huge simplification! Let's verify this logic.
    # If C_t[i,j,k] = 0 for all i,j,k, then removing it doesn't change the sum.
    # But C_t[i,j,k] = U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]
    # This is zero for all i,j,k only if for every (i,j,k), at least one of U[i*m+k,t], V[k*p+j,t], W[j*n+i,t] is zero.
    # That's a strong condition. But it means we can check each column's contribution in O(1) per entry.

    # Actually wait - the condition is that the SUM without column t equals 1.
    # Sum without t = total - C_t = 1 - C_t. We need 1 - C_t = 1, so C_t = 0.
    # Yes! So a column is removable iff its individual contribution tensor is all zeros.

    # But that's too restrictive. The column might not be individually zero, but its contribution
    # might be "absorbed" by other columns. However, since each column contributes independently,
    # the only way removing it preserves the sum is if its contribution is zero.
    
    # Hmm, but that's only true if we're looking at it from the perspective of the sum.
    # Actually, I think I was wrong. Let me reconsider.
    # T = sum_t C_t where C_t is the rank-1 tensor from column t.
    # T[i,j,k] = 1 for all i,j,k.
    # If we remove column t: T' = T - C_t. We need T'[i,j,k] = 1 for all i,j,k.
    # So T[i,j,k] - C_t[i,j,k] = 1, meaning C_t[i,j,k] = T[i,j,k] - 1 = 0.
    # So yes, C_t must be identically zero.
    
    # This means the greedy approach as originally written (checking if removing column t still gives all-ones)
    # is equivalent to checking if C_t is all zeros. Let's use this for a fast check.

    def column_contribution_is_zero(Um, Vm, Wm, t):
        """Check if the rank-1 tensor from column t is identically zero."""
        U_col = Um[:, t]  # shape (n*m,)
        V_col = Vm[:, t]  # shape (m*p,)
        W_col = Wm[:, t]  # shape (p*n,)
        
        # C[i,j,k] = U[i*m+k] * V[k*p+j] * W[j*n+i]
        # For this to be zero for all i,j,k, we need that for every (i,j,k),
        # at least one factor is zero.
        
        # Quick check: if U_col has any nonzero, V_col has any nonzero, W_col has any nonzero,
        # we still need to check the full tensor. But we can short-circuit.
        
        if not np.any(U_col) or not np.any(V_col) or not np.any(W_col):
            return True
        
        # Build the full contribution tensor
        U_r = U_col.reshape(n, m)
        V_r = V_col.reshape(m, p)
        W_r = W_col.reshape(p, n)
        
        # C[i,j,k] = U_r[i,k] * V_r[k,j] * W_r[j,i]
        # Compute efficiently
        C = np.einsum('ik,kj,ji->ijk', U_r, V_r, W_r)
        return not np.any(C)

    # Greedy removal using the fast check
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_r = R

    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        removed = False
        for t in range(current_r - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if column_contribution_is_zero(current_U, current_V, current_W, t):
                mask = np.ones(current_r, dtype=bool)
                mask[t] = False
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_r -= 1
                removed = True
                break
        if not removed:
            break

    # Final verification
    assert verify_fast(current_U, current_V, current_W), "Final decomposition is invalid"

    return {
        "U": current_U.tolist(),
        "V": current_V.tolist(),
        "W": current_W.tolist()
    }
