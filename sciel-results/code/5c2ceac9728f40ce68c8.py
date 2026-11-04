import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=int)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1,  0,  1,  0,  1]
    ], dtype=int)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=int)

    # Kronecker product gives 4x4 from 2x2: rank 49
    U_kron = np.kron(U2, U2)  # shape (16, 49)
    V_kron = np.kron(V2, V2)  # shape (16, 49)
    W_kron = np.kron(W2, W2)  # shape (16, 49)

    # Permute rows to standard row-major ordering
    def get_permutation_for_U():
        perm = []
        for S in range(16):
            i = S // 4
            j = S % 4
            i_out, i_in = i // 2, i % 2
            j_out, j_in = j // 2, j % 2
            block = i_out * 2 + j_out
            sub = i_in * 2 + j_in
            perm.append(block * 4 + sub)
        return perm

    def get_permutation_for_V():
        perm = []
        for S in range(16):
            j = S // 4
            k = S % 4
            j_out, j_in = j // 2, j % 2
            k_out, k_in = k // 2, k % 2
            block = j_out * 2 + k_out
            sub = j_in * 2 + k_in
            perm.append(block * 4 + sub)
        return perm

    def get_permutation_for_W():
        perm = []
        for S in range(16):
            k = S // 4
            i = S % 4
            k_out, k_in = k // 2, k % 2
            i_out, i_in = i // 2, i % 2
            block = k_out * 2 + i_out
            sub = k_in * 2 + i_in
            perm.append(block * 4 + sub)
        return perm

    perm_U = get_permutation_for_U()
    perm_V = get_permutation_for_V()
    perm_W = get_permutation_for_W()

    U = U_kron[perm_U, :].astype(float)
    V = V_kron[perm_V, :].astype(float)
    W = W_kron[perm_W, :].astype(float)

    # Verification function
    def verify(U, V, W):
        max_err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(U[i * 4 + j, :], V[j * 4 + k, :] * W[k * 4 + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Initial verification
    max_err = verify(U, V, W)
    if max_err > 1e-9:
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j, col] = 1.0
                    V_fb[j * 4 + k, col] = 1.0
                    W_fb[k * 4 + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Precompute contributions: contrib[t, c] where t = i*16 + j*4 + k
    contrib = np.zeros((64, 49))
    t = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                contrib[t, :] = U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :]
                t += 1
    
    # Greedy elimination using precomputed contributions
    remaining_mask = np.ones(49, dtype=bool)
    
    # Sort columns by number of non-zero contributions (easier to remove)
    nonzero_counts = (np.abs(contrib) > 1e-15).sum(axis=0)
    order = np.argsort(nonzero_counts)
    
    for c in order:
        if time.time() - t0 > deadline - 15.0:
            break
        if not remaining_mask[c]:
            continue
        
        # Try removing column c
        remaining_mask[c] = False
        
        # Check if all 64 equations still hold
        remaining_cols = np.where(remaining_mask)[0]
        if len(remaining_cols) == 0:
            remaining_mask[c] = True
            continue
        
        sums = contrib[:, remaining_cols].sum(axis=1)
        
        if np.max(np.abs(sums - 1.0)) < 1e-9:
            pass  # Column c can be removed
        else:
            remaining_mask[c] = True  # Restore
    
    # Extract final matrices
    final_cols = np.where(remaining_mask)[0]
    U_final = U[:, final_cols]
    V_final = V[:, final_cols]
    W_final = W[:, final_cols]
    
    if time.time() - t0 > deadline - 5.0:
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # Final verification
    max_err = verify(U_final, V_final, W_final)
    if max_err > 1e-9:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

if __name__ == "__main__":
    construct()
