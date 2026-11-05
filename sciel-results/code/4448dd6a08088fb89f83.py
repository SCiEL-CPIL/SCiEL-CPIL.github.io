import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    def verify(U, V, W, seed=12345, trials=12):
        """Return True if the decomposition exactly reproduces A@B for random inputs."""
        try:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            rng = np.random.RandomState(seed)
            for _ in range(trials):
                A = rng.rand(n, m)
                B = rng.rand(m, p)
                C_expected = A @ B
                x = (A.reshape(n*m) @ Ua)
                y = (B.reshape(m*p) @ Va)
                prod = x * y
                C_computed = (Wa * prod[None, :]).sum(axis=1).reshape(p, n)
                C_expected = C_expected.T
                if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
                    return False
            return True
        except Exception:
            return False

    def reduce_rank(U, V, W, max_iter=10):
        """Aggressive rank reduction using null-space projection.
        For each term, check if it can be eliminated by expressing it as a
        linear combination of others via the trilinear constraint."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        removed = 0
        
        for iteration in range(max_iter):
            if elapsed() > DEADLINE - 15:
                break
            if rank <= 1:
                break
            
            # Try each column for removal
            found = False
            for c in range(rank):
                if elapsed() > DEADLINE - 15:
                    break
                
                # Remove column c and check if the tensor is still preserved
                Uo = np.delete(Ua, c, axis=1)
                Vo = np.delete(Va, c, axis=1)
                Wo = np.delete(Wa, c, axis=1)
                
                # The removed term contributes:
                # T_removed = sum_{i,j,k} Ua[i*m+j,c] * Va[j*p+k,c] * Wa[k*n+i,c] * e_{i,j,k}
                # We need to check if this can be absorbed by the remaining terms.
                # This is equivalent to checking if the "residual tensor" after removal
                # can be represented by the remaining rank-1 terms.
                
                # Quick check: verify with a few random inputs
                ok = True
                rng = np.random.RandomState(42 + c)
                for _ in range(3):
                    A = rng.rand(n, m)
                    B = rng.rand(m, p)
                    C_expected = A @ B
                    x = (A.reshape(n*m) @ Uo)
                    y = (B.reshape(m*p) @ Vo)
                    prod = x * y
                    C_computed = (Wo * prod[None, :]).sum(axis=1).reshape(p, n)
                    C_exp_T = C_expected.T
                    if not np.allclose(C_computed, C_exp_T, atol=1e-7, rtol=1e-7):
                        ok = False
                        break
                
                if ok:
                    Ua = Uo
                    Va = Vo
                    Wa = Wo
                    rank -= 1
                    removed += 1
                    found = True
                    break
            
            if not found:
                break
        
        def to_list(a):
            return [[(float(v)) for v in row] for row in a]
        return to_list(Ua), to_list(Va), to_list(Wa), removed

    def reduce_rank_v2(U, V, W, max_iter=8):
        """Smarter reduction: use SVD on the combined factor space to find
        linear dependencies among the rank-1 terms."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        removed = 0
        
        for iteration in range(max_iter):
            if elapsed() > DEADLINE - 15:
                break
            if rank <= 1:
                break
            
            # Construct the "signature" of each term in the tensor space.
            # Each term alpha defines a rank-1 tensor T_alpha[i,j,k] = U[i*m+j,alpha]*V[j*p+k,alpha]*W[k*n+i,alpha]
            # Two terms are linearly dependent if their tensor signatures are proportional.
            # More generally, we want to find if any term's signature is in the span of others.
            
            # For efficiency, project onto a random low-dim space
            rng = np.random.RandomState(123 + iteration)
            nproj = min(20, rank - 1)
            
            # Random projections to test linear independence
            proj_U = rng.randn(nproj, n*m)
            proj_V = rng.randn(nproj, m*p)
            proj_W = rng.randn(nproj, p*n)
            
            # Compute projected signatures
            sigs = np.zeros((rank, 3 * nproj))
            for a in range(rank):
                ua = Ua[:, a]
                va = Va[:, a]
                wa = Wa[:, a]
                # Project each factor
                pu = proj_U @ ua
                pv = proj_V @ va
                pw = proj_W @ wa
                sigs[a] = np.concatenate([pu, pv, pw])
            
            # Check rank of signature matrix
            sig_rank = np.linalg.matrix_rank(sigs, tol=1e-8)
            if sig_rank < rank:
                # There's a dependency; find which term to remove
                # Use SVD to find the null space
                _, S, Vt = np.linalg.svd(sigs, full_matrices=False)
                # Find smallest singular value
                min_idx = np.argmin(S)
                if S[min_idx] < 1e-6:
                    # The corresponding right singular vector gives the dependency
                    dep_vec = Vt[min_idx]
                    # Find the largest component
                    c = np.argmax(np.abs(dep_vec))
                    # Try removing term c
                    Uo = np.delete(Ua, c, axis=1)
                    Vo = np.delete(Va, c, axis=1)
                    Wo = np.delete(Wa, c, axis=1)
                    
                    # Verify
                    ok = True
                    rng2 = np.random.RandomState(999 + c)
                    for _ in range(5):
                        A = rng2.rand(n, m)
                        B = rng2.rand(m, p)
                        C_expected = A @ B
                        x = (A.reshape(n*m) @ Uo)
                        y = (B.reshape(m*p) @ Vo)
                        prod = x * y
                        C_computed = (Wo * prod[None, :]).sum(axis=1).reshape(p, n)
                        C_exp_T = C_expected.T
                        if not np.allclose(C_computed, C_exp_T, atol=1e-7, rtol=1e-7):
                            ok = False
                            break
                    
                    if ok:
                        Ua = Uo
                        Va = Vo
                        Wa = Wo
                        rank -= 1
                        removed += 1
                        continue
            
            # Try simple norm-based removal
            norms = [(np.sum(Ua[:, c]**2) + np.sum(Va[:, c]**2) + np.sum(Wa[:, c]**2), c) for c in range(rank)]
            norms.sort()
            found = False
            for _, c in norms[:5]:  # Try only smallest 5
                Uo = np.delete(Ua, c, axis=1)
                Vo = np.delete(Va, c, axis=1)
                Wo = np.delete(Wa, c, axis=1)
                
                ok = True
                rng3 = np.random.RandomState(555 + c)
                for _ in range(3):
                    A = rng3.rand(n, m)
                    B = rng3.rand(m, p)
                    C_expected = A @ B
                    x = (A.reshape(n*m) @ Uo)
                    y = (B.reshape(m*p) @ Vo)
                    prod = x * y
                    C_computed = (Wo * prod[None, :]).sum(axis=1).reshape(p, n)
                    C_exp_T = C_expected.T
                    if not np.allclose(C_computed, C_exp_T, atol=1e-7, rtol=1e-7):
                        ok = False
                        break
                
                if ok:
                    Ua = Uo
                    Va = Vo
                    Wa = Wo
                    rank -= 1
                    removed += 1
                    found = True
                    break
            
            if not found:
                break
        
        def to_list(a):
            return [[(float(v)) for v in row] for row in a]
        return to_list(Ua), to_list(Va), to_list(Wa), removed

    # ---- Build the known-correct double-Strassen (49) + schoolbook (16) = 65 ----
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Inner Strassen 2x2x2
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]

    # Super Strassen 4x4x4
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Verify the baseline
    baseline_ok = verify(U, V, W, seed=99, trials=8)

    # Attempt rank reduction
    best = (U, V, W)
    if baseline_ok and elapsed() < DEADLINE - 30:
        # Try v2 first (smarter)
        Ur, Vr, Wr, removed = reduce_rank_v2(U, V, W, max_iter=8)
        if removed > 0 and verify(Ur, Vr, Wr, seed=777, trials=15):
            best = (Ur, Vr, Wr)
        
        # Try v1 on the result if we have time
        if elapsed() < DEADLINE - 15:
            Ur2, Vr2, Wr2, removed2 = reduce_rank(best[0], best[1], best[2], max_iter=5)
            if removed2 > 0 and verify(Ur2, Vr2, Wr2, seed=888, trials=10):
                best = (Ur2, Vr2, Wr2)

    Uo, Vo, Wo = best
    return {"U": Uo, "V": Vo, "W": Wo}
