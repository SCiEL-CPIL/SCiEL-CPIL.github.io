import time
import numpy as np
import random

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    # Strassen 2x2*2x2 factors
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    prods = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Part 1: 4x4*4x4 Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2
    block_products = [
        (0, 0, 0, 4, 0, 4),
        (0, 2, 2, 4, 0, 4),
        (2, 0, 0, 4, 2, 4),
        (2, 2, 2, 4, 2, 4),
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    # Fast verification using precomputed tensor
    tensor = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i * m + j, j * p + k, k * n + i] = 1
    
    # Get indices where tensor is 1 for fast checking
    tensor_indices = np.argwhere(tensor == 1)
    # Get indices where tensor is 0 (for checking zeros) - we'll check a subset or use complement logic
    # Actually, checking all zeros is expensive. We can rely on the fact that if the non-zero parts match, 
    # and the rank is low, it's likely correct, but for exactness we need to check zeros too.
    # However, the tensor is sparse. 
    # Let's precompute the "support" of the tensor.
    support = set(map(tuple, tensor_indices))
    
    # For zero checking, we can't precompute all zeros easily in a set for 16x24x24.
    # But we can just iterate over all and check.
    # Optimization: If we verify the non-zero entries are 1, and we know the decomposition is "exact", 
    # do we need to check zeros? Yes, the definition requires exact match.
    
    def verify_fast(U_, V_, W_):
        """Fast verification using precomputed tensor."""
        rank = U_.shape[1]
        if rank == 0:
            return False
            
        # Check non-zero entries
        for idx in tensor_indices:
            i, j, k = idx
            s = 0
            # Vectorized dot product for speed
            # sum_r U[i,r] * V[j,r] * W[k,r]
            # This is element-wise multiply then sum
            prod = U_[i, :] * V_[j, :]
            s = np.dot(prod, W_[k, :])
            if s != 1:
                return False
        
        # Check zero entries (sample or full?)
        # Full check is 16*24*24 = 9216 checks. Each is a dot product of length rank.
        # If rank is ~70, that's 9216 * 70 ops. ~600k ops. Fast enough.
        # Let's do a full check but efficiently.
        # We can vectorize the zero check too?
        # For each (i,j,k) where tensor[i,j,k]==0, check sum == 0.
        
        # Iterate over all possible (i,j,k)
        # To speed up, we can check only if we haven't already verified it (which we haven't, tensor is 0)
        # Let's just loop over all indices and skip the ones in support.
        
        # Actually, generating all indices and filtering is slow.
        # Let's just iterate i, j, k.
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    if tensor[i, j, k] == 1:
                        continue
                    # Check if sum is 0
                    # s = np.dot(U_[i, :] * V_[j, :], W_[k, :])
                    # if s != 0: return False
                    
                    # Optimization: if any of U[i], V[j], W[k] is all zeros, sum is 0.
                    # But they are unlikely to be all zeros in a minimal decomposition.
                    
                    # Let's try to vectorize this check in batches?
                    # Or just use numpy dot.
                    s = np.dot(U_[i, :] * V_[j, :], W_[k, :])
                    if s != 0:
                        return False
        return True

    def try_remove_fast(U_, V_, W_, r):
        """Try removing column r; return (U,V,W) if valid, else None."""
        U2 = np.delete(U_, r, axis=1)
        V2 = np.delete(V_, r, axis=1)
        W2 = np.delete(W_, r, axis=1)
        if verify_fast(U2, V2, W2):
            return U2, V2, W2
        return None

    # Greedy rank reduction with fast verification
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while time.time() < deadline:
        ncol = best_U.shape[1]
        removed = False
        for r in range(ncol):
            if time.time() > deadline:
                break
            res = try_remove_fast(best_U, best_V, best_W, r)
            if res is not None:
                best_U, best_V, best_W = res
                removed = True
                break
        if not removed:
            break

    # Try random restarts with small perturbations
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        best_rank = best_U.shape[1]
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Start from current best
            U_c, V_c, W_c = best_U.copy(), best_V.copy(), best_W.copy()
            
            # Try removing multiple columns at once with random order
            ncol = U_c.shape[1]
            order = rng.permutation(ncol)
            
            for r in order:
                if time.time() > deadline - 5:
                    break
                if r >= U_c.shape[1]:
                    continue
                U2 = np.delete(U_c, r, axis=1)
                V2 = np.delete(V_c, r, axis=1)
                W2 = np.delete(W_c, r, axis=1)
                if verify_fast(U2, V2, W2):
                    U_c, V_c, W_c = U2, V2, W2
            
            if U_c.shape[1] < best_rank:
                best_U, best_V, best_W = U_c, V_c, W_c
                best_rank = U_c.shape[1]

    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
