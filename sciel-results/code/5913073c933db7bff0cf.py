"""Combined: structural Strassen construction + refinement machinery + IHT search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute the 60 non-zero tensor entries: (U_idx, V_idx, W_idx)
    entries = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    ENTRIES = entries  # capture for closures

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, r, tol=1e-9):
        """Fast verification: check the 60 non-zero entries only."""
        for (ui, vi, wi) in ENTRIES:
            val = np.dot(U[ui, :r], V[vi, :r] * W[wi, :r])
            if abs(val - 1.0) > tol:
                return False
        # Check all other entries are ~0
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    if (i, j, k) in _entry_set:
                        continue
                    val = np.dot(U[i, :r], V[j, :r] * W[k, :r])
                    if abs(val) > tol:
                        return False
        return True

    # Build set of non-zero entries for fast lookup
    _entry_set = set(ENTRIES)

    def build_rank55():
        """Rank-55 Strassen decomposition: j=0 trivial (20) + 5 Strassen blocks (7 each)."""
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        # Part A: j=0 trivial (2 * 10 = 20 factors)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of (k0, k1)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        assert col == rank
        return U, V, W, rank

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for (ui, vi, wi) in ENTRIES:
            U[ui, col] = 1
            V[vi, col] = 1
            W[wi, col] = 1
            col += 1
        return U, V, W, rank

    def remove_redundant_factors(U, V, W, r, max_passes=3):
        """Iteratively remove factors whose tensor contribution is zero (redundant)."""
        for _ in range(max_passes):
            if elapsed() > DEADLINE:
                break
            changed = False
            for c in range(r - 1, -1, -1):
                if elapsed() > DEADLINE:
                    break
                # Check if factor c contributes zero to all entries
                uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
                redundant = True
                for (ui, vi, wi) in ENTRIES:
                    if abs(uc[ui] * vc[vi] * wc[wi]) > 1e-12:
                        # This factor has a non-zero entry contribution; check if removing it still works
                        # Actually, just check if all contributions are zero
                        pass
                # Better: check if this factor's contribution to the tensor is identically zero
                # A factor contributes u_i * v_j * w_k to entry (i,j,k). If all such products are zero, it's redundant.
                # But we need the SUM to equal the target, so a factor is removable iff its contribution is zero everywhere.
                max_contrib = 0.0
                for (ui, vi, wi) in ENTRIES:
                    contrib = uc[ui] * vc[vi] * wc[wi]
                    if abs(contrib) > max_contrib:
                        max_contrib = abs(contrib)
                if max_contrib < 1e-12:
                    # This factor contributes nothing; remove it
                    mask = [i for i in range(r) if i != c]
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    r = len(mask)
                    changed = True
                    break
            if not changed:
                break
        return U, V, W, r

    def iht_search(U, V, W, r, target_rank, iters=200):
        """IHT on mode-1 flattening to find a lower-rank factorization."""
        # Mode-1 matricization: X of shape (nm, mp*pn)
        # X[i, j*pn + k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # For the true tensor, X[i, j*pn+k] = 1 if (i,j,k) in ENTRIES else 0
        # We want to find U', V', W' with rank r' < r such that sum_c' U'[i,c']*V'[j,c']*W'[k,c'] = X[i, j*pn+k]
        
        # This is hard. Instead, use a simpler approach: 
        # Try to find a factorization by solving a system of equations.
        # For a given rank r', we have r' * (nm + mp + pn) unknowns and nm*mp*pn equations.
        # This is underdetermined for small r'.
        
        # Alternative: use the structure. The tensor is a sum of rank-1 tensors.
        # We can try to find a sparse representation.
        
        # For now, just try to remove factors more aggressively.
        return U, V, W, r

    def polish(U, V, W, r):
        """Round to nearest half-integer and verify."""
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2, r):
            return U2, V2, W2, r
        return U, V, W, r

    def randomized_search(U, V, W, r, iterations=200):
        """Random perturbations + rounding + removal."""
        best = (U, V, W, r)
        for _ in range(iterations):
            if elapsed() > DEADLINE:
                break
            U_p = U + np.random.randn(nm, r) * 0.05
            V_p = V + np.random.randn(mp, r) * 0.05
            W_p = W + np.random.randn(pn, r) * 0.05
            U_r = np.round(U_p * 2) / 2
            V_r = np.round(V_p * 2) / 2
            W_r = np.round(W_p * 2) / 2
            if verify(U_r, V_r, W_r, r):
                U_r, V_r, W_r, r_r = remove_redundant_factors(U_r, V_r, W_r, r)
                if r_r < best[3]:
                    best = (U_r, V_r, W_r, r_r)
        return best

    # Build initial decomposition
    U, V, W, r = build_rank55()
    if not verify(U, V, W, r):
        U, V, W, r = build_schoolbook()

    # Remove redundant factors
    U, V, W, r = remove_redundant_factors(U, V, W, r)

    # Randomized search
    if elapsed() < DEADLINE - 5:
        U, V, W, r = randomized_search(U, V, W, r, iterations=100)

    # Final polish
    U, V, W, r = polish(U, V, W, r)

    # Final verification; fallback to schoolbook if broken
    if not verify(U, V, W, r):
        U, V, W, r = build_schoolbook()

    # Convert to lists of lists for the interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
