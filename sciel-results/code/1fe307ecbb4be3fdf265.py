"""Rewritten program: structured Strassen-like decomposition for 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 3, 3, 6

    # Start with schoolbook (rank 54)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank, n, m, p, tol=1e-9):
        """Verify exact decomposition using numpy for speed."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # For each (i,j,k) check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = U[:, c]  # shape (n*m,)
            v = V[:, c]  # shape (m*p,)
            w = W[:, c]  # shape (p*n,)
            u3 = u.reshape(n, m)
            v3 = v.reshape(m, p)
            w3 = w.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u3[i, j] * (v3[j, :] * w3[:, i])
        return np.all(np.abs(T - 1.0) <= tol)

    def verify_decomp_fast(U, V, W, rank):
        """Fast verification for 3x3x6."""
        n, m, p = 3, 3, 6
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            u3 = u.reshape(3, 3)
            v3 = v.reshape(3, 6)
            w3 = w.reshape(6, 3)
            for i in range(3):
                for j in range(3):
                    T[i, j, :] += u3[i, j] * (v3[j, :] * w3[:, i])
        return np.all(np.abs(T - 1.0) <= 1e-9)

    # ============================================================
    # Strategy 1: Try to find a low-rank 3x3x3 decomposition
    # and use it to build a 3x3x6 decomposition
    # ============================================================

    def find_3x3x3_decomposition(max_rank=23, min_rank=15, max_time=60):
        """Find a rank-r decomposition for 3x3x3 matrix multiplication."""
        best = None
        best_r = max_rank + 1

        def verify_3x3x3(U, V, W, r):
            U = np.array(U, dtype=np.float64)
            V = np.array(V, dtype=np.float64)
            W = np.array(W, dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True

        for r in range(max_rank, min_rank - 1, -1):
            if time.time() > start_time + max_time:
                break
            for attempt in range(30):
                if time.time() > start_time + max_time:
                    break
                rng = np.random.RandomState(attempt * 7919 + r * 31)
                vals = [-1, 0, 1]
                U = rng.choice(vals, size=(9, r))
                V = rng.choice(vals, size=(9, r))
                W = rng.choice(vals, size=(9, r))
                if verify_3x3x3(U, V, W, r):
                    best = (U, V, W, r)
                    best_r = r
                    break
        return best

    # Try to find 3x3x3 decomposition
    t3 = time.time()
    dec3 = find_3x3x3_decomposition(max_rank=23, min_rank=15, max_time=30)

    if dec3 is not None:
        U3, V3, W3, r3 = dec3
        # Build 3x3x6 decomposition from 3x3x3
        # Use 2*r3 terms: r3 for block 1 (k=0,1,2), r3 for block 2 (k=3,4,5)
        r6 = 2 * r3
        U6 = np.zeros((9, r6))
        V6 = np.zeros((18, r6))
        W6 = np.zeros((18, r6))

        # Block 1: c in 0..r3-1, k in 0,1,2
        U6[:, :r3] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+k, :r3] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W6[k*3+i, :r3] = W3[k*3+i]

        # Block 2: c in r3..2*r3-1, k in 3,4,5
        U6[:, r3:] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+3+k, r3:] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W6[(3+k)*3+i, r3:] = W3[k*3+i]

        if verify_decomp_fast(U6, V6, W6, r6):
            if r6 < best_rank:
                best_rank = r6
                best_result = {
                    "U": U6.tolist(),
                    "V": V6.tolist(),
                    "W": W6.tolist()
                }

    # ============================================================
    # Strategy 2: Direct random search for 3x3x6 with structured init
    # ============================================================

    # Try various rank values from low to high
    for rank in range(35, 54):
        if time.time() > deadline - 15:
            break

        max_attempts = 40
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            rng = np.random.RandomState(attempt * 104729 + rank * 13)

            # Use a structured initialization: start from schoolbook and perturb
            # or use random sparse matrices
            if attempt < 10:
                # Dense random with {-1, 0, 1}
                vals = [-1, 0, 1]
                U = rng.choice(vals, size=(9, rank))
                V = rng.choice(vals, size=(18, rank))
                W = rng.choice(vals, size=(18, rank))
            elif attempt < 20:
                # Sparse random
                vals = [-1, 0, 1]
                density = 0.25
                U = rng.choice(vals, size=(9, rank))
                V = rng.choice(vals, size=(18, rank))
                W = rng.choice(vals, size=(18, rank))
                U = U * (rng.random((9, rank)) < density)
                V = V * (rng.random((18, rank)) < density)
                W = W * (rng.random((18, rank)) < density)
            else:
                # Use half-integer values
                vals = [-0.5, 0, 0.5, 1, 1.5]
                U = rng.choice(vals, size=(9, rank))
                V = rng.choice(vals, size=(18, rank))
                W = rng.choice(vals, size=(18, rank))

            if verify_decomp_fast(U, V, W, rank):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break  # Move to next (lower) rank

    # ============================================================
    # Strategy 3: Try to reduce rank from current best using
    # local search - try removing/merging columns
    # ============================================================
    if best_rank < 54 and time.time() < deadline - 20:
        U_best = np.array(best_result["U"], dtype=np.float64)
        V_best = np.array(best_result["V"], dtype=np.float64)
        W_best = np.array(best_result["W"], dtype=np.float64)
        r_best = best_rank

        # Try removing columns (reducing rank)
        for target_rank in range(r_best - 1, max(30, r_best - 15), -1):
            if time.time() > deadline - 10:
                break
            # Try random subsets of columns
            for attempt in range(30):
                if time.time() > deadline - 5:
                    break
                rng = np.random.RandomState(attempt * 9973 + target_rank)
                cols = rng.choice(r_best, size=target_rank, replace=False)
                cols.sort()
                U_sub = U_best[:, cols]
                V_sub = V_best[:, cols]
                W_sub = W_best[:, cols]
                if verify_decomp_fast(U_sub, V_sub, W_sub, target_rank):
                    best_rank = target_rank
                    best_result = {
                        "U": U_sub.tolist(),
                        "V": V_sub.tolist(),
                        "W": W_sub.tolist()
                    }
                    break

    # ============================================================
    # Strategy 4: Try a completely different approach - use the
    # block structure more aggressively
    # ============================================================

    # View 3x3x6 as two 3x3x3 multiplications sharing A
    # Try to find a decomposition where some terms serve both blocks

    if time.time() < deadline - 25:
        # Try rank = 45, 44, 43, 42 with more sophisticated search
        for target_rank in [45, 44, 43, 42]:
            if target_rank >= best_rank:
                continue
            if time.time() > deadline - 10:
                break

            for attempt in range(50):
                if time.time() > deadline - 8:
                    break

                rng = np.random.RandomState(attempt * 5555 + target_rank)

                # Initialize with a structured pattern
                # U: 9 x rank, V: 18 x rank, W: 18 x rank

                # Try to use a pattern where the first half of columns
                # handles block 1 and the second half handles block 2,
                # but with some overlap
                U = np.zeros((9, target_rank))
                V = np.zeros((18, target_rank))
                W = np.zeros((18, target_rank))

                # Fill with random values from a structured set
                for c in range(target_rank):
                    # Each column gets a random pattern
                    U[:, c] = rng.choice([-1, 0, 1], size=9)
                    V[:, c] = rng.choice([-1, 0, 1], size=18)
                    W[:, c] = rng.choice([-1, 0, 1], size=18)

                # Apply some sparsity
                if rng.random() < 0.5:
                    density = rng.uniform(0.2, 0.4)
                    U = U * (rng.random((9, target_rank)) < density)
                    V = V * (rng.random((18, target_rank)) < density)
                    W = W * (rng.random((18, target_rank)) < density)

                if verify_decomp_fast(U, V, W, target_rank):
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break

    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
