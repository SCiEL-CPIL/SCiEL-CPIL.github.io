"""Matrix multiplication 2x3x6: integer rank-1 term search with rank-33 fallback."""
DIMENSIONS = [2, 3, 6]
import time

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build the target multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    def verify_terms(terms):
        """Return True if sum of rank-1 terms equals T exactly."""
        R = np.zeros((nm, mp, pn), dtype=np.int64)
        for u, v, w in terms:
            R += np.einsum('a,b,c->abc', u, v, w)
        return np.array_equal(R, T)

    # --- Fallback: rank-33 block decomposition (Strassen 2x2x2 + scalar 1x1x2 per block) ---
    def build_rank33():
        # 2x3x2 rank-11 terms (Strassen on first 2 cols + 4 scalar terms)
        terms_232 = [
            ({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({3:1, 4:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({4:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({3:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 4:-1}, {2:1, 3:1}, {0:1}),
            ({2:1}, {4:1}, {0:1}),
            ({2:1}, {5:1}, {1:1}),
            ({5:1}, {4:1}, {2:1}),
            ({5:1}, {5:1}, {3:1}),
        ]
        all_terms = []
        for block in range(3):
            for ud, vd, wd in terms_232:
                u = np.zeros(nm, dtype=np.int64)
                v = np.zeros(mp, dtype=np.int64)
                w = np.zeros(pn, dtype=np.int64)
                for idx, val in ud.items():
                    u[idx] = val
                for l_idx, val in vd.items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    v[j_local*p + block*2 + k_local] = val
                for l_idx, val in wd.items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    w[(block*2+k_local)*n + i_local] = val
                all_terms.append((u, v, w))
        return all_terms

    fallback_terms = build_rank33()
    assert verify_terms(fallback_terms), "Rank-33 fallback should verify"

    best_terms = fallback_terms
    best_rank = 33

    # --- Integer rank-1 term search ---
    # Try ranks from 24 to 32 (skip 33 since we already have it)
    for target_rank in range(24, 33):
        if time.time() > deadline - 20:
            break

        # Multiple restarts for randomization
        max_restarts = 12 if target_rank <= 26 else 8
        if time.time() > deadline - 15:
            max_restarts = 2

        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            np.random.seed(42 + restart * 1000 + target_rank)
            terms = []
            R = T.copy()

            for step in range(target_rank):
                if time.time() > deadline - 8:
                    break
                # Compute best rank-1 approximation to residual R
                # u_i = sum_{j,k} R[i,j,k]*v[j]*w[k] etc. Use SVD-like:
                # Flatten R to matrix of shape (nm, mp*pn), get top left singular
                Rmat = R.reshape(nm, mp*pn)
                try:
                    U_s, s_vals, Vt = np.linalg.svd(Rmat, full_matrices=False)
                except:
                    break
                u_cont = U_s[:, 0] * np.sqrt(s_vals[0])
                vw_cont = Vt[0, :] * np.sqrt(s_vals[0])
                v_cont = vw_cont[:mp]
                w_cont = vw_cont[mp:]

                # Normalize: divide by max abs to keep small
                max_u = np.max(np.abs(u_cont))
                max_v = np.max(np.abs(v_cont))
                max_w = np.max(np.abs(w_cont))
                scale = max(max_u, max_v, max_w)
                if scale < 1e-9:
                    break
                u_r = np.round(u_cont / scale)
                v_r = np.round(v_cont / scale)
                w_r = np.round(w_cont / scale)

                # If rounding gives zero vector, try without normalization
                if not (np.any(u_r) and np.any(v_r) and np.any(w_r)):
                    u_r = np.round(u_cont)
                    v_r = np.round(v_cont)
                    w_r = np.round(w_cont)
                    if not (np.any(u_r) and np.any(v_r) and np.any(w_r)):
                        # Try small random perturbation
                        u_r = np.round(u_cont + np.random.uniform(-0.4, 0.4, nm))
                        v_r = np.round(v_cont + np.random.uniform(-0.4, 0.4, mp))
                        w_r = np.round(w_cont + np.random.uniform(-0.4, 0.4, pn))

                if not (np.any(u_r) and np.any(v_r) and np.any(w_r)):
                    break

                # Subtract this term from residual
                term_tensor = np.einsum('a,b,c->abc', u_r, v_r, w_r)
                new_R = R - term_tensor
                # Check that this step doesn't increase the L1 error too much
                # (accept if it reduces or maintains)
                if np.abs(new_R).sum() <= np.abs(R).sum() + 5:
                    R = new_R
                    terms.append((u_r, v_r, w_r))
                else:
                    # Try a few random alternatives
                    accepted = False
                    for attempt in range(3):
                        u_alt = np.round(u_cont + np.random.uniform(-1.5, 1.5, nm))
                        v_alt = np.round(v_cont + np.random.uniform(-1.5, 1.5, mp))
                        w_alt = np.round(w_cont + np.random.uniform(-1.5, 1.5, pn))
                        if not (np.any(u_alt) and np.any(v_alt) and np.any(w_alt)):
                            continue
                        alt_R = R - np.einsum('a,b,c->abc', u_alt, v_alt, w_alt)
                        if np.abs(alt_R).sum() <= np.abs(R).sum() + 3:
                            R = alt_R
                            terms.append((u_alt, v_alt, w_alt))
                            accepted = True
                            break
                    if not accepted:
                        break

            # Check if we've zeroed out the tensor
            if np.all(R == 0):
                if verify_terms(terms):
                    best_terms = terms
                    best_rank = target_rank
                    found = True
                    break

            if found:
                break

    # Build result from best_terms
    rank = best_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    for r, (u, v, w) in enumerate(best_terms):
        for i in range(nm):
            if u[i] != 0:
                U[i][r] = int(u[i])
        for j in range(mp):
            if v[j] != 0:
                V[j][r] = int(v[j])
        for k in range(pn):
            if w[k] != 0:
                W[k][r] = int(w[k])

    result = {"U": U, "V": V, "W": W}

    # Final verification
    if time.time() < deadline - 3:
        try:
            _verify(result, n, m, p)
        except Exception:
            # Fallback to rank-33
            U33 = [[0] * 33 for _ in range(nm)]
            V33 = [[0] * 33 for _ in range(mp)]
            W33 = [[0] * 33 for _ in range(pn)]
            for r, (u, v, w) in enumerate(fallback_terms):
                for i in range(nm):
                    if u[i] != 0:
                        U33[i][r] = int(u[i])
                for j in range(mp):
                    if v[j] != 0:
                        V33[j][r] = int(v[j])
                for k in range(pn):
                    if w[k] != 0:
                        W33[k][r] = int(w[k])
            result = {"U": U33, "V": V33, "W": W33}

    return result

def _verify(result, n, m, p):
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    return True
# EVOLVE-BLOCK-END
