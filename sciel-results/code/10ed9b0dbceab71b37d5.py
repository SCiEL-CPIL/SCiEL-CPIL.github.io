import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 6, 6, 6
    bs = 2
    nb = n // bs  # 3
    R = nb ** 3 * 7  # 189

    # Pre-compute the 7 Strassen terms' local patterns for 2x2 blocks
    # Each term is (u_pattern, v_pattern, w_pattern) where each pattern
    # is a dict mapping (row, col) -> coefficient
    terms = []
    # M1 = (X00+X11)(Y00+Y11)
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    ))
    # M2 = (X10+X11)Y00
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1, (1,1):-1}
    ))
    # M3 = X00(Y01-Y11)
    terms.append((
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(0,1):1, (1,1):1}
    ))
    # M4 = X11(Y10-Y00)
    terms.append((
        {(1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,0):1, (1,0):1}
    ))
    # M5 = (X00+X01)Y11
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,0):-1, (0,1):1}
    ))
    # M6 = (X10-X00)(Y00+Y01)
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    ))
    # M7 = (X01-X11)(Y10+Y11)
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

    # Verify correctness of the decomposition
    def verify(Um, Vm, Wm):
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    total = 0
                    for t in range(Um.shape[1]):
                        total += Um[i*m+k, t] * Vm[k*p+j, t] * Wm[j*n+i, t]
                    if total != 1:
                        return False
        return True

    assert verify(U, V, W), "Initial decomposition is invalid"

    # Greedy rank reduction: try to remove columns
    # We use a smarter approach: check if a column can be removed
    # by computing the tensor residual without that column

    def tensor_without(Um, Vm, Wm, exclude):
        """Compute the tensor T[i,j,k] = sum_{t != exclude} U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t]"""
        r = Um.shape[1]
        # For efficiency, compute per-row contributions
        # U_row: shape (n*m, r), V_col: shape (m*p, r), W_out: shape (p*n, r)
        # T[i,j,k] = sum_t U[i*m+k,t] * V[k*p+j,t] * W[j*n+i,t]
        
        # Build the full tensor and zero out column exclude
        # This is O(n*m*p*r) per check which is 216*r
        # For r~189, that's ~40k ops per check, times ~189 checks = ~7.6M ops
        # Fast enough
        
        T = np.ones((n, p, m), dtype=np.int64)  # target is all 1s
        # Actually let's just check directly
        keep_mask = [c for c in range(r) if c != exclude]
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    total = 0
                    for t in keep_mask:
                        total += Um[i*m+k, t] * Vm[k*p+j, t] * Wm[j*n+i, t]
                    if total != 1:
                        return False
        return True

    # Faster verification using numpy
    def verify_fast(Um, Vm, Wm, exclude=None):
        r = Um.shape[1]
        if exclude is not None:
            mask = np.ones(r, dtype=bool)
            mask[exclude] = False
            Um_c = Um[:, mask]
            Vm_c = Vm[:, mask]
            Wm_c = Wm[:, mask]
        else:
            Um_c = Um
            Vm_c = Vm
            Wm_c = Wm
        
        # T[i,j,k] = sum_t U[i*m+k,t] * V[k*p+j,t] * W[j*n+i,t]
        # Reshape: U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # We need to check all (i,j,k) triples
        
        # Method: for each t, compute outer product contribution
        # U_t[i*m+k] * V_t[k*p+j] * W_t[j*n+i]
        # Let's use einsum
        # U shape: (n*m, r), reshape to (n, m, r)
        # V shape: (m*p, r), reshape to (m, p, r)
        # W shape: (p*n, r), reshape to (p, n, r)
        
        U_r = Um_c.reshape(n, m, -1)
        V_r = Vm_c.reshape(m, p, -1)
        W_r = Wm_c.reshape(p, n, -1)
        
        # T[i,j,k] = sum_t U_r[i,k,t] * V_r[k,j,t] * W_r[j,i,t]
        # Use einsum: 'ikt,kjt,jit->ijk'
        T = np.einsum('ikt,kjt,jit->ijk', U_r, V_r, W_r)
        
        # Check all entries are 1
        return np.all(T == 1)

    # Test the fast verifier
    assert verify_fast(U, V, W), "Fast verifier failed on initial decomposition"

    # Now do greedy removal
    current_U = U
    current_V = V
    current_W = W
    current_r = R
    
    # Try removing each column, starting from the last
    # To be efficient, we can do multiple passes
    for pass_num in range(5):
        if time.time() > deadline - 10:
            break
        removed = False
        for t in range(current_r - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            if verify_fast(current_U, current_V, current_W, exclude=t):
                # Remove column t
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
