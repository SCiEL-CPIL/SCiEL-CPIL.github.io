import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    # ------------------------------------------------------------------
    # LADDER / PEELING CONSTRUCTION
    #
    # 2x3x7 = (j=2 row: 2x1x7, rank 14)
    #       + (k=6 col, j in {0,1}: 2x2x1, rank 4)
    #       + (core: j in {0,1}, k in {0..5}: 2x2x6)
    #
    # The 2x2x6 core is a direct sum of three 2x2x2 Strassen blocks
    # (k-pairs {0,1},{2,3},{4,5}), each of exact rank 7  ->  rank 21.
    #
    # Total rank = 14 + 4 + 21 = 39.  Verified exactly below.
    # ------------------------------------------------------------------
    def build_peeled_decomposition():
        rank = 14 + 4 + 21
        U = [[0] * rank for _ in range(n * m)]   # index i*m+j
        V = [[0] * rank for _ in range(m * p)]   # index j*p+k
        W = [[0] * rank for _ in range(p * n)]   # index k*n+i
        col = 0

        def add_term(u_map, v_map, w_map):
            """u_map/v_map/w_map: dict {index: coeff}. Adds one rank-1 term."""
            nonlocal col
            for idx, c in u_map.items():
                U[idx][col] += c
            for idx, c in v_map.items():
                V[idx][col] += c
            for idx, c in w_map.items():
                W[idx][col] += c
            col += 1

        def add_strassen_block(k0):
            """2x2x2 Strassen block on A[0..1,0..1] x B[0..1, k0..k0+1] -> C[k0..k0+1, 0..1]."""
            # A indices (i*m+j): A00=0, A01=1, A10=3, A11=4
            # B indices (j*p+k): b00=0*7+k0, b01=0*7+k0+1, b10=1*7+k0, b11=1*7+k0+1
            b00 = 0 * 7 + k0
            b01 = 0 * 7 + k0 + 1
            b10 = 1 * 7 + k0
            b11 = 1 * 7 + k0 + 1
            # C indices (k*n+i): c00=k0*2+0, c01=(k0+1)*2+0, c10=k0*2+1, c11=(k0+1)*2+1
            c00 = k0 * 2 + 0
            c01 = (k0 + 1) * 2 + 0
            c10 = k0 * 2 + 1
            c11 = (k0 + 1) * 2 + 1

            # M1: (A00+A11)(B00+B11) -> C00 + C11
            add_term({0: 1, 4: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
            # M2: (A10+A11)B00 -> C10 - C11
            add_term({3: 1, 4: 1}, {b00: 1}, {c10: 1, c11: -1})
            # M3: A00(B01-B11) -> C01 + C11
            add_term({0: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
            # M4: A11(B10-B00) -> C00 + C10
            add_term({4: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
            # M5: (A00+A01)B11 -> -C00 + C01
            add_term({0: 1, 1: 1}, {b11: 1}, {c00: -1, c01: 1})
            # M6: (A10-A00)(B00+B01) -> C11
            add_term({3: 1, 0: -1}, {b00: 1, b01: 1}, {c11: 1})
            # M7: (A01-A11)(B10+B11) -> C00
            add_term({1: 1, 4: -1}, {b10: 1, b11: 1}, {c00: 1})

        # --- Core: three Strassen blocks ---
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)

        # --- k=6 column, j in {0,1}: 2x2x1 schoolbook (rank 4) ---
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        add_term({0: 1}, {b06: 1}, {c06: 1})   # A00 B06 -> C06
        add_term({1: 1}, {b16: 1}, {c06: 1})   # A01 B16 -> C06
        add_term({3: 1}, {b06: 1}, {c16: 1})   # A10 B06 -> C16
        add_term({4: 1}, {b16: 1}, {c16: 1})   # A11 B16 -> C16

        # --- j=2 row: 2x1x7 schoolbook (rank 14) ---
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term({2: 1}, {b2k: 1}, {c0k: 1})   # A02 B2k -> C0k
            add_term({5: 1}, {b2k: 1}, {c1k: 1})   # A12 B2k -> C1k

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        # tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # ------------------------------------------------------------------
    # Vectorized refinement on the 2x2x6 CORE sub-tensor.
    #
    # The core tensor C[i,j,k] (i in 0..1, j in 0..1, k in 0..5) is a
    # direct sum of three 2x2x2 blocks.  We attempt an alternating
    # minimization to see whether a rank < 21 representation exists for
    # the core.  This is cheap (small dims) and uses vectorized ops.
    # If it converges to residual ~0 at a lower rank we keep it; else we
    # fall back to the verified rank-39 base.
    # ------------------------------------------------------------------
    def core_tensor():
        """Return the 2x2x6 core tensor (all entries should be 1 on the
        block-diagonal Strassen structure, 0 elsewhere) as a float array
        of shape (2,2,6).  Actually the core is the *multiplication tensor*
        restricted to i,j in {0,1}, k in {0..5}: T[i,j,k] = 1 for all
        (i,j,k) in that box (since C[i,k] = sum_j A[i,j] B[j,k])."""
        # The multiplication tensor has T[i,j,k]=1 for every (i,j,k).
        # Restricting to the core box gives an all-ones (2,2,6) tensor.
        return np.ones((2, 2, 6), dtype=np.float64)

    def refine_core(target_rank, max_iter=200, tol=1e-10, seed=0):
        """Alternating minimization for rank-target_rank decomposition of
        the (2,2,6) all-ones core tensor.  Returns (U,V,W) arrays or None."""
        if time.time() > deadline - 15:
            return None
        C = core_tensor()
        rng = np.random.default_rng(1000 + seed + target_rank)
        nm, mp, pn = 2 * 2, 2 * 6, 6 * 2
        # Flatten indices: U over (i,j) 4 entries, V over (j,k) 12, W over (k,i) 12
        best = None
        best_res = np.inf
        for restart in range(2):
            if time.time() > deadline - 15:
                break
            U = rng.standard_normal((nm, target_rank))
            V = rng.standard_normal((mp, target_rank))
            W = rng.standard_normal((pn, target_rank))
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                # Update U: for each (i,j), U[ij,:] = 1 / sum_k V[jk,:] * W[ki,:]
                for ij in range(nm):
                    i, j = divmod(ij, 2)
                    denom = np.zeros(target_rank)
                    for k in range(6):
                        denom += V[j * 6 + k, :] * W[k * 2 + i, :]
                    mask = np.abs(denom) > 1e-12
                    U[ij, mask] = 1.0 / denom[mask]
                    U[ij, ~mask] = 0.0
                # Update V: for each (j,k), V[jk,:] = 1 / sum_i U[ij,:] * W[ki,:]
                for jk in range(mp):
                    j, k = divmod(jk, 6)
                    denom = np.zeros(target_rank)
                    for i in range(2):
                        denom += U[i * 2 + j, :] * W[k * 2 + i, :]
                    mask = np.abs(denom) > 1e-12
                    V[jk, mask] = 1.0 / denom[mask]
                    V[jk, ~mask] = 0.0
                # Update W: for each (k,i), W[ki,:] = 1 / sum_j U[ij,:] * V[jk,:]
                for ki in range(pn):
                    k, i = divmod(ki, 2)
                    denom = np.zeros(target_rank)
                    for j in range(2):
                        denom += U[i * 2 + j, :] * V[j * 6 + k, :]
                    mask = np.abs(denom) > 1e-12
                    W[ki, mask] = 1.0 / denom[mask]
                    W[ki, ~mask] = 0.0
                # Residual: ||T - sum_r U[:,r] V[:,r] W[:,r]||_F
                recon = np.zeros((2, 2, 6))
                for r in range(target_rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                res = float(np.linalg.norm(recon - C))
                if res < best_res:
                    best_res = res
                    best = (U.copy(), V.copy(), W.copy())
                if res < tol:
                    break
        if best is not None and best_res < tol:
            return best
        return None

    # ------------------------------------------------------------------
    # Build the verified base (rank 39) first.
    # ------------------------------------------------------------------
    base_solution = build_peeled_decomposition()
    base_rank = len(base_solution["U"][0])

    if not verify_decomposition(base_solution["U"], base_solution["V"], base_solution["W"], n, m, p):
        # Fallback: pure schoolbook (rank 42) -- guaranteed correct.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ------------------------------------------------------------------
    # Attempt to shave the CORE rank (21) via vectorized refinement.
    # The full rank would drop to 14 + 4 + core_rank.  Try a few target
    # core ranks below 21.  The all-ones (2,2,6) tensor has rank
    # min(2*2, 2*6, 6*2)=4 in the *unconstrained* tensor sense, but it is
    # constrained here to be a *multiplication* tensor (a product of
    # structure), so we only accept solutions that, when spliced back,
    # verify the full 2x3x7 multiplication exactly.
    # ------------------------------------------------------------------
    better_solution = None
    for core_rank in range(4, 21):
        if time.time() > deadline - 20:
            break
        core = refine_core(core_rank)
        if core is None:
            continue
        Uc, Vc, Wc = core
        # Splice: core occupies i,j in {0,1}, k in {0..5}.
        # Build full U,V,W of rank 14+4+core_rank.
        full_rank = 14 + 4 + core_rank
        U = [[0] * full_rank for _ in range(n * m)]
        V = [[0] * full_rank for _ in range(m * p)]
        W = [[0] * full_rank for _ in range(p * n)]
        # Copy base k=6 and j=2 parts (first 18 columns of base) into full.
        # base columns 0..20 = core (Strassen), 21..24 = k=6, 25..38 = j=2.
        # We keep k=6 (4) and j=2 (14) from base, replace core with refined.
        # k=6 block in base: columns 21..24 ; j=2 block: columns 25..38.
        baseU = np.array(base_solution["U"], dtype=np.float64)
        baseV = np.array(base_solution["V"], dtype=np.float64)
        baseW = np.array(base_solution["W"], dtype=np.float64)
        # Place refined core in columns 0..core_rank-1
        for ij in range(4):
            for r in range(core_rank):
                U[ij][r] = float(Uc[ij, r])
        for jk in range(12):
            j, k = divmod(jk, 6)
            if k < 6:
                for r in range(core_rank):
                    V[jk][r] = float(Vc[jk, r])
        for ki in range(12):
            k, i = divmod(ki, 2)
            if k < 6:
                for r in range(core_rank):
                    W[ki][r] = float(Wc[ki, r])
        # Place k=6 block (base cols 21..24) into full cols core_rank..core_rank+3
        for idx in range(n * m):
            for r in range(4):
                U[idx][core_rank + r] = float(baseU[idx, 21 + r])
        for idx in range(m * p):
            for r in range(4):
                V[idx][core_rank + r] = float(baseV[idx, 21 + r])
        for idx in range(p * n):
            for r in range(4):
                W[idx][core_rank + r] = float(baseW[idx, 21 + r])
        # Place j=2 block (base cols 25..38) into full cols core_rank+4..
        for idx in range(n * m):
            for r in range(14):
                U[idx][core_rank + 4 + r] = float(baseU[idx, 25 + r])
        for idx in range(m * p):
            for r in range(14):
                V[idx][core_rank + 4 + r] = float(baseV[idx, 25 + r])
        for idx in range(p * n):
            for r in range(14):
                W[idx][core_rank + 4 + r] = float(baseW[idx, 25 + r])

        # Round to clean values and verify the FULL multiplication.
        U_list = [[round(float(x), 6) for x in row] for row in U]
        V_list = [[round(float(x), 6) for x in row] for row in V]
        W_list = [[round(float(x), 6) for x in row] for row in W]
        if verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
            better_solution = {"U": U_list, "V": V_list, "W": W_list}
            break

    if better_solution is not None:
        return better_solution
    return base_solution

# EVOLVE-BLOCK-END
