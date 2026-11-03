import numpy as np
import time
import itertools

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 120

    # Target tensor T: T[i*m+j, j*p+k, k*n+i] = 1
    # We seek rank-r decomposition: T = sum_{t=1}^r U_t ⊗ V_t ⊗ W_t
    # such that for all (i,j,k):
    #   sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 if (i,j,k) valid
    #   and the decomposition is exact (0 elsewhere)

    # Strategy: Start with the schoolbook decomposition (rank 80), then attempt
    # to reduce the rank by finding linear dependencies or using known
    # fast matrix multiplication techniques.

    # 1. Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # 2. Initial decomposition: schoolbook (rank = n*m*p = 80)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
            U[i*m+j, t] = 1.0
            V[j*p+k, t] = 1.0
            W[k*n+i, t] = 1.0
        return U, V, W

    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    def verify(U, V, W):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zeros
        for a in range(n*m):
            ia, ja = divmod(a, m)
            for b in range(m*p):
                jb, kb = divmod(b, p)
                for c in range(p*n):
                    kc, ic = divmod(c, n)
                    if ja != jb or ia != ic or kb != kc:
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
        return True

    # 3. Try Strassen-based decomposition for the 4x4 block, then extend to 4x5
    # For 4x4 * 4x4: use 2-level Strassen (rank 49)
    # For the remaining columns (4x1 blocks): add rank 4 each
    # Total: 49 + 4*4 = 49 + 16 = 65
    # Actually, let's be more careful.

    # A 4x4 * 4x5 multiplication can be viewed as:
    # [4x4] * [4x5] = [4x5]
    # Split the 4x5 into five 4x1 blocks:
    # [4x4] * [4x1 | 4x1 | 4x1 | 4x1 | 4x1]
    # Each 4x4 * 4x1 requires rank 4 (no known faster method for this specific shape)
    # The 4x4 * 4x4 part (first 4 columns of output) uses Strassen rank 49
    # But wait, we need to multiply 4x4 by each of the 5 columns of the 4x5 matrix.
    # So we have 5 independent 4x4 * 4x1 multiplications.
    # Total rank = 5 * 4 = 20? No, that's not right either.

    # Let me think differently. The matrix multiplication tensor for m x n * n x p
    # has rank at most m*n*p. We can use known fast algorithms.

    # For 4x4 * 4x4, the best known rank is 49 (2-level Strassen) or possibly less.
    # Actually, the rank of 4x4 * 4x4 is known to be at most 49, and it's conjectured
    # to be less. The best proven upper bound is 49.

    # For our problem 4x4 * 4x5:
    # We can write this as a block multiplication.
    # Let A be 4x4, B be 4x5.
    # Split B into [B1 B2] where B1 is 4x2 and B2 is 4x3.
    # A * B1 is 4x2, A * B2 is 4x3.
    # This doesn't help directly.

    # Alternative: use the fact that we can reuse computations.
    # Let's just use a direct search approach with a time limit.

    # 4. Randomized local search for rank reduction
    # Start from schoolbook and try to find linear dependencies.

    def try_reduce_rank(U, V, W, current_rank, max_iters=10000):
        """Try to reduce rank by finding removable columns."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        active = list(range(current_rank))

        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            if len(active) < 2:
                break

            # Try removing a random column
            r = random.choice(active)
            new_active = [x for x in active if x != r]

            # Check if the remaining columns still span the target tensor
            # This is expensive, so we do a quick check first
            T_rebuilt = np.zeros((n*m, m*p, p*n))
            for t in new_active:
                T_rebuilt += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])

            if np.allclose(T_rebuilt, T, atol=1e-9):
                active = new_active

        if len(active) < current_rank:
            return U[:, active], V[:, active], W[:, active], len(active)
        return U, V, W, current_rank

    # For efficiency, let's use a structured approach.
    # Known: rank(4x4*4x4) <= 49
    # We can decompose the 4x4*4x5 as follows:
    # Write B = [B1, B2, B3, B4, B5] where each Bi is 4x1
    # C = A*B = [A*B1, A*B2, A*B3, A*B4, A*B5]
    # Each A*Bi is a 4x4 * 4x1 multiplication.
    # The rank of 4x4 * 4x1 is at most 4 (schoolbook), and I believe this is tight.
    # So total rank <= 5 * 4 = 20? That seems too low.

    # Wait, I think I'm confusing myself. Let me use a different approach.
    # The matrix multiplication tensor T_{i,j,k} for (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # For A: 4x4, B: 4x5
    # T has indices (i,j) for A, (j,k) for B, (i,k) for C
    # The tensor has size (4*4) x (4*5) x (5*4) = 16 x 20 x 20

    # The schoolbook rank is 4*4*5 = 80.

    # Let's try to use a Kronecker product approach or known fast algorithms.

    # Actually, let's just do a more careful local search.
    import random

    random.seed(42)
    U, V, W = schoolbook()
    current_rank = n * m * p

    # Precompute the tensor contributions for each term
    def tensor_contrib(U_col, V_col, W_col):
        return np.einsum('a,b,c->abc', U_col, V_col, W_col)

    # Try greedy removal with random restarts
    for restart in range(20):
        if time.time() > deadline - 30:
            break

        U_curr, V_curr, W_curr = schoolbook()
        active = list(range(n * m * p))
        T_target = T.copy()

        # Randomly shuffle the order
        random.shuffle(active)

        for r_idx in active:
            if time.time() > deadline - 30:
                break

            # Try removing column r_idx
            new_active = [x for x in active if x != r_idx]

            # Quick check: rebuild tensor
            T_test = np.zeros((n*m, m*p, p*n))
            for t in new_active:
                T_test += tensor_contrib(U_curr[:, t], V_curr[:, t], W_curr[:, t])

            if np.allclose(T_test, T_target, atol=1e-9):
                active = new_active
                T_target = T_test

        if len(active) < current_rank:
            U_curr = U_curr[:, active]
            V_curr = V_curr[:, active]
            W_curr = W_curr[:, active]
            current_rank = len(active)
            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            best_rank = current_rank

    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
