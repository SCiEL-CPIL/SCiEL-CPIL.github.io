"""Combined: structural Strassen construction + linear-algebra-based factor removal + polynomial-time verification."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute the 60 non-zero tensor entries: (U_idx, V_idx, W_idx)
    entries = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    ENTRIES = entries
    _entry_set = set(ENTRIES)

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, r, tol=1e-9):
        """Fast verification: check all 6*30*20 = 3600 entries using einsum."""
        T = np.einsum('ic,jc,kc->ijk', U[:, :r], V[:, :r], W[:, :r])
        # Check non-zero entries are 1
        for (ui, vi, wi) in ENTRIES:
            if abs(T[ui, vi, wi] - 1.0) > tol:
                return False
        # Check zero entries are ~0
        # Create target tensor
        T_target = np.zeros((nm, mp, pn))
        for (ui, vi, wi) in ENTRIES:
            T_target[ui, vi, wi] = 1.0
        if not np.allclose(T, T_target, atol=tol):
            return False
        return True

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

    def remove_linearly_dependent_factors(U, V, W, r, tol=1e-8, max_passes=5):
        """
        Remove factors that are linear combinations of others.
        A factor c is redundant if its rank-1 tensor u_c ⊗ v_c ⊗ w_c
        can be expressed as a linear combination of the other rank-1 tensors.
        
        We check this by seeing if the vectorized tensor of factor c
        lies in the span of the vectorized tensors of the other factors.
        """
        for _ in range(max_passes):
            if elapsed() > DEADLINE:
                break
            changed = False
            for c in range(r - 1, -1, -1):
                if elapsed() > DEADLINE:
                    break
                
                # Vectorize factor c's contribution
                # T_c[i,j,k] = U[i,c] * V[j,c] * W[k,c]
                T_c = np.outer(np.outer(U[:, c], V[:, c]), W[:, c]).reshape(-1)
                
                # Build matrix of all other factors' vectorized tensors
                others = [i for i in range(r) if i != c]
                if not others:
                    continue
                    
                # Check if T_c is in the span of {T_others}
                # Use least squares: minimize ||T_c - M @ x||
                M = np.zeros((nm * mp * pn, len(others)))
                for idx, o in enumerate(others):
                    M[:, idx] = np.outer(np.outer(U[:, o], V[:, o]), W[:, o]).reshape(-1)
                
                # Solve least squares
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, T_c, rcond=tol)
                    residual = T_c - M @ x
                    if np.linalg.norm(residual) < tol * max(1.0, np.linalg.norm(T_c)):
                        # Factor c is linearly dependent; remove it
                        mask = [i for i in range(r) if i != c]
                        U = U[:, mask]
                        V = V[:, mask]
                        W = W[:, mask]
                        r = len(mask)
                        changed = True
                        break
                except np.linalg.LinAlgError:
                    continue
            
            if not changed:
                break
        return U, V, W, r

    def try_factor_replacement(U, V, W, r, max_trials=50):
        """
        Try to replace pairs of factors with a single factor.
        If factors c1 and c2 can be combined into one factor c_new
        such that T_c1 + T_c2 = T_c_new, we reduce rank by 1.
        
        This is equivalent to checking if T_c1 + T_c2 is rank-1.
        """
        best_U, best_V, best_W, best_r = U, V, W, r
        
        for trial in range(max_trials):
            if elapsed() > DEADLINE:
                break
            
            # Randomly pick two factors to try combining
            c1, c2 = np.random.choice(r, size=2, replace=False)
            
            # Sum of two rank-1 tensors
            T_sum = np.outer(np.outer(U[:, c1] + U[:, c2], V[:, c1] + V[:, c2]), W[:, c1] + W[:, c2])
            # Wait, that's not right. T_c1 + T_c2 = u1⊗v1⊗w1 + u2⊗v2⊗w2
            # This is NOT generally rank-1.
            
            # Actually, we need to check if u1⊗v1⊗w1 + u2⊗v2⊗w2 is rank-1.
            # A tensor is rank-1 iff it can be written as u⊗v⊗w for some u,v,w.
            # This is hard to check directly.
            
            # Alternative: try to find u,v,w such that u⊗v⊗w = T_c1 + T_c2
            # This is a system of equations. For small dimensions, we can try.
            
            # Actually, a simpler approach: just try to remove one factor at a time
            # using the linear dependence check above.
            
            # Let's instead try a different strategy: perturb and round
            # to find equivalent representations with fewer non-zeros.
            pass
        
        return best_U, best_V, best_W, best_r

    def polish(U, V, W, r):
        """Round to nearest integer and verify."""
        U2 = np.round(U)
        V2 = np.round(V)
        W2 = np.round(W)
        if verify(U2, V2, W2, r):
            return U2, V2, W2, r
        # Try half-integers
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2, r):
            return U2, V2, W2, r
        return U, V, W, r

    # Build initial decomposition
    U, V, W, r = build_rank55()
    if not verify(U, V, W, r):
        U, V, W, r = build_schoolbook()

    # Remove linearly dependent factors
    U, V, W, r = remove_linearly_dependent_factors(U, V, W, r)

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
