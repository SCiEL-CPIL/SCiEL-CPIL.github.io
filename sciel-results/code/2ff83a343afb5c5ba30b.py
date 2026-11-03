import numpy as np
import time
from scipy.linalg import null_space

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

    # Greedy removal: for each column, check if removing it preserves the tensor
    # A column can be removed iff its contribution tensor is identically zero
    def column_contribution_is_zero(Um, Vm, Wm, t):
        U_col = Um[:, t]
        V_col = Vm[:, t]
        W_col = Wm[:, t]
        
        if not np.any(U_col) or not np.any(V_col) or not np.any(W_col):
            return True
        
        U_r = U_col.reshape(n, m)
        V_r = V_col.reshape(m, p)
        W_r = W_col.reshape(p, n)
        
        C = np.einsum('ik,kj,ji->ijk', U_r, V_r, W_r)
        return not np.any(C)

    # First pass: remove zero-contribution columns
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_r = R

    for pass_num in range(5):
        if time.time() > deadline - 20:
            break
        removed = False
        for t in range(current_r - 1, -1, -1):
            if time.time() > deadline - 20:
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

    # Second pass: try to find linear dependencies between columns
    # If columns t1 and t2 are linearly dependent in a way that their contributions cancel,
    # we might be able to remove one. But this is complex.
    # Instead, try a different approach: look for pairs of columns whose sum has zero contribution.
    # This is expensive, so only do it if we have time.
    
    if time.time() < deadline - 30 and current_r > 1:
        # Try removing pairs of columns whose combined contribution is zero
        removed_pair = False
        for t1 in range(current_r - 1):
            if time.time() > deadline - 10:
                break
            for t2 in range(t1 + 1, current_r):
                if time.time() > deadline - 10:
                    break
                # Check if C_t1 + C_t2 = 0
                U1 = current_U[:, t1].reshape(n, m)
                U2 = current_U[:, t2].reshape(n, m)
                V1 = current_V[:, t1].reshape(m, p)
                V2 = current_V[:, t2].reshape(m, p)
                W1 = current_W[:, t1].reshape(p, n)
                W2 = current_W[:, t2].reshape(p, n)
                
                C1 = np.einsum('ik,kj,ji->ijk', U1, V1, W1)
                C2 = np.einsum('ik,kj,ji->ijk', U2, V2, W2)
                
                if not np.any(C1 + C2):
                    # Remove both columns
                    mask = np.ones(current_r, dtype=bool)
                    mask[t1] = False
                    mask[t2] = False
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_r -= 2
                    removed_pair = True
                    break
            if removed_pair:
                break
        if removed_pair:
            # Restart the single-column removal
            for pass_num in range(3):
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
