import time
import numpy as np
from itertools import product

DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60

    def time_ok():
        return time.time() - start_time < DEADLINE

    # Direct algebraic construction: try to find rank < 54
    # For 2x5x6, we can use a block structure
    # Split: A is 2x5, B is 5x6
    # Block 1: A[:, 0:3] * B[0:3, :] -> 2x3x3
    # Block 2: A[:, 3:5] * B[3:5, :] -> 2x2x6
    
    # For 2x3x3: use Strassen-like on 2x2x2 blocks + remaining
    # For 2x2x6: use 2x2x2 Strassen blocks
    
    # Alternative: use a known lower bound approach
    # R(2x5x6) <= R(2x3x3) + R(2x2x6)
    # R(2x3x3) can be done in rank 8 (Strassen for 2x2x2 + 2x1x3 + 2x1x3... let me think)
    
    # Actually, let me try a direct construction:
    # Split columns of B into groups: {0,1,2}, {3,4,5}
    # A[:, 0:3] * B[0:3, 0:3] -> 2x3x3
    # A[:, 0:3] * B[0:3, 3:6] -> 2x3x3
    # A[:, 3:5] * B[3:5, 0:3] -> 2x2x3
    # A[:, 3:5] * B[3:5, 3:6] -> 2x2x3
    
    # R(2x3x3): We can use Strassen on 2x2x2 sub-blocks
    # 2x3x3 = (2x2x2) + (2x1x2) + (2x1x1) + (2x2x1) + (2x1x1)
    # That's 7 + 2 + 1 + 2 + 1 = 13... not great
    
    # Better: for 2x3x3, use the fact that it's like multiplying by a 3x3 matrix
    # Rank of 2x3x3 tensor: known to be 8 (from 2x2x2 Strassen + corrections)
    
    # Let me try: R(2x5x6) via different block splits
    # Split B columns as {0,1}, {2,3}, {4,5} -> three 2x5x2 blocks
    # R(2x5x2): can this be done efficiently?
    # 2x5x2 = (2x3x2) + (2x2x2)
    # R(2x3x2): 2x3x2 tensor rank? 
    # R(2x2x2) = 7 (Strassen)
    
    # So R(2x5x2) <= R(2x3x2) + 7
    # R(2x3x2): This is like C = A*B where A is 2x3, B is 3x2
    # = sum over j of A[:,j] * B[j,:]
    # Each term is rank 1 outer product scaled... 
    # Actually 2x3x2 has 12 entries. Rank at least 4 (by dimension arguments)
    # Can we do it in rank 6? 2x2x2 blocks: (2x2x2) + (2x1x2) = 7+2=9? No...
    # 2x3x2: use Strassen on 2x2x2 part (cols 0,1 of B, rows 0,1 of A) + remaining
    # That gives 7 + 2 = 9 for 2x3x2? That seems high.
    
    # Let me just try the greedy search more aggressively with better candidates

    # ---- 1) Build a pool of structured rank-1 tensors (candidate contributions) ----
    # A rank-1 tensor is (u_vec, v_vec, w_vec) with contribution c[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)
    seen_patterns = set()

    def add_candidate(u_sparse, v_sparse, w_sparse):
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        if np.any(contrib):
            # Deduplicate by pattern
            key = (tuple(u_vec), tuple(v_vec), tuple(w_vec))
            if key not in seen_patterns:
                seen_patterns.add(key)
                candidates.append((u_vec, v_vec, w_vec, contrib))

    # Type 1: standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})

    # Type 1b: negative unit tensors (for cancellation)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): -1}, {(j,k): 1}, {(k,i): 1})
                add_candidate({(i,j): 1}, {(j,k): -1}, {(k,i): 1})
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): -1})

    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        """
        Standard Strassen decomposition for 2x2x2 matrix multiplication.
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B11)
        M4 = A22(B12-B22)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        """
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(({(io,jo): 1, (io+1,jo+1): 1},
                      {(jo,ko): 1, (jo+1,ko+1): 1},
                      {(ko,io): 1, (ko+1,io+1): 1}))
        # M2 = (A21+A22)B11
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1},
                      {(jo,ko): 1},
                      {(ko,io+1): 1, (ko+1,io+1): 1}))
        # M3 = A11(B21-B11)
        terms.append(({(io,jo): 1},
                      {(jo+1,ko): 1, (jo,ko): -1},
                      {(ko,io): 1, (ko+1,io): 1}))
        # M4 = A22(B12-B22)
        terms.append(({(io+1,jo+1): 1},
                      {(jo,ko+1): 1, (jo+1,ko+1): -1},
                      {(ko+1,io+1): 1, (ko+1,io): 1}))
        # M5 = (A11+A12)B22
        terms.append(({(io,jo): 1, (io,jo+1): 1},
                      {(jo+1,ko+1): 1},
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        # M6 = (A21-A11)(B11+B12)
        terms.append(({(io+1,jo): 1, (io,jo): -1},
                      {(jo,ko): 1, (jo,ko+1): 1},
                      {(ko,io+1): 1, (ko+1,io+1): 1}))
        # M7 = (A12-A22)(B21+B22)
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1},
                      {(jo+1,ko): 1, (jo+1,ko+1): 1},
                      {(ko,io): 1, (ko+1,io): 1}))
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)

    # Type 3: structured 2x3x3 sign-pattern candidates
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for i_pat in [[1,1],[1,-1],[-1,1]]:
                    for j_pat in [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]:
                        for k_pat in [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]:
                            u_sparse = {}
                            v_sparse = {}
                            w_sparse = {}
                            for ii in range(2):
                                u_sparse[(io+ii, jo)] = i_pat[ii]
                                u_sparse[(io+ii, jo+1)] = i_pat[ii]
                                u_sparse[(io+ii, jo+2)] = i_pat[ii]
                            for jj in range(3):
                                v_sparse[(jo+jj, ko)] = j_pat[jj]
                                v_sparse[(jo+jj, ko+1)] = j_pat[jj]
                                v_sparse[(jo+jj, ko+2)] = j_pat[jj]
                            for kk in range(3):
                                w_sparse[(ko+kk, io)] = k_pat[kk]
                                w_sparse[(ko+kk, io+1)] = k_pat[kk]
                            add_candidate(u_sparse, v_sparse, w_sparse)

    # Type 3b: 2x2x3 and 2x3x2 structured blocks
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 2):
                # 2x2x3: use Strassen on 2x2x2 + one extra column
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                # Add terms for the third column
                add_candidate({(io, jo): 1, (io+1, jo): 1},
                              {(jo, ko+2): 1, (jo+1, ko+2): 1},
                              {(ko+2, io): 1, (ko+2, io+1): 1})
                add_candidate({(io, jo+1): 1, (io+1, jo+1): 1},
                              {(jo, ko+2): 1, (jo+1, ko+2): 1},
                              {(ko+2, io): 1, (ko+2, io+1): 1})
                # Cross terms for 2x2x3
                add_candidate({(io, jo): 1, (io+1, jo+1): 1},
                              {(jo, ko+2): 1},
                              {(ko+2, io): 1, (ko+2, io+1): 1})
                add_candidate({(io, jo+1): 1, (io+1, jo): 1},
                              {(jo+1, ko+2): 1},
                              {(ko+2, io): 1, (ko+2, io+1): 1})

    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                # 2x3x2: use Strassen on 2x2x2 sub-block + extras
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                # Extra terms for third row of A
                add_candidate({(io, jo+2): 1, (io+1, jo+2): 1},
                              {(jo+2, ko): 1, (jo+2, ko+1): 1},
                              {(ko, io): 1, (ko+1, io): 1})
                add_candidate({(io, jo+2): 1, (io+1, jo+2): 1},
                              {(jo+2, ko): 1},
                              {(ko, io): 1, (ko+1, io): 1})
                add_candidate({(io, jo+2): 1, (io+1, jo+2): 1},
                              {(jo+2, ko+1): 1},
                              {(ko+1, io): 1, (ko+1, io+1): 1})

    # Type 4: diagonal-like patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                for shift in range(3):
                    u_s = {}
                    v_s = {}
                    w_s = {}
                    for ii in range(2):
                        for jj in range(3):
                            if (jj - ii) % 3 == shift:
                                u_s[(io+ii, jo+jj)] = 1
                    for jj in range(3):
                        for kk in range(3):
                            if (kk - jj) % 3 == shift:
                                v_s[(jo+jj, ko+kk)] = 1
                    for kk in range(3):
                        for ii in range(2):
                            if (ii - kk) % 3 == shift:
                                w_s[(ko+kk, io+ii)] = 1
                    add_candidate(u_s, v_s, w_s)

    # Type 4b: Sparse random-like patterns for 2x2x2 with varied support
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                # All possible 2-entry patterns on each factor
                u_positions = [(io,jo), (io,jo+1), (io+1,jo), (io+1,jo+1)]
                v_positions = [(jo,ko), (jo,ko+1), (jo+1,ko), (jo+1,ko+1)]
                w_positions = [(ko,io), (ko,io+1), (ko+1,io), (ko+1,io+1)]
                for u_pos in product([0,1], repeat=4):
                    for v_pos in product([0,1], repeat=4):
                        for w_pos in product([0,1], repeat=4):
                            if sum(u_pos) == 0 or sum(v_pos) == 0 or sum(w_pos) == 0:
                                continue
                            if sum(u_pos) > 3 or sum(v_pos) > 3 or sum(w_pos) > 3:
                                continue
                            u_s = {}
                            v_s = {}
                            w_s = {}
                            for idx, pos in enumerate(u_positions):
                                if u_pos[idx]:
                                    u_s[pos] = 1
                            for idx, pos in enumerate(v_positions):
                                if v_pos[idx]:
                                    v_s[pos] = 1
                            for idx, pos in enumerate(w_positions):
                                if w_pos[idx]:
                                    w_s[pos] = 1
                            add_candidate(u_s, v_s, w_s)

    # Type 5: S3 permutation-based structured candidates on 2x2x2
    S3_elements = [
        [0, 1, 2],
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0],
        [1, 2, 0],
        [2, 0, 1],
    ]
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for perm in S3_elements:
                    u_s = {}; v_s = {}; w_s = {}
                    if perm == [0, 1, 2]:
                        u_s = {(io, jo): 1, (io+1, jo+1): 1}
                        v_s = {(jo, ko): 1, (jo+1, ko+1): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [1, 0, 2]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo, ko): 1, (jo+1, ko+1): 1}
                        w_s = {(ko, io+1): 1, (ko+1, io): 1}
                    elif perm == [0, 2, 1]:
                        u_s = {(io, jo): 1, (io+1, jo+1): 1}
                        v_s = {(jo, ko+1): 1, (jo+1, ko): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [2, 1, 0]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo, ko+1): 1, (jo+1, ko): 1}
                        w_s = {(ko, io+1): 1, (ko+1, io): 1}
                    elif perm == [1, 2, 0]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo+1, ko): 1, (jo, ko+1): 1}
                        w_s = {(ko, io): 1, (ko+1, io+1): 1}
                    elif perm == [2, 0, 1]:
                        u_s = {(io, jo+1): 1, (io+1, jo): 1}
                        v_s = {(jo+1, ko): 1, (jo, ko+1): 1}
                        w_s = {(ko+1, io): 1, (ko, io+1): 1}
                    add_candidate(u_s, v_s, w_s)

    # Type 5b: Half-integer coefficient candidates for 2x2x2
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                # Try various coefficient combinations
                for a in [1, -1, 0.5, -0.5]:
                    for b in [1, -1, 0.5, -0.5]:
                        for c in [1, -1, 0.5, -0.5]:
                            for d in [1, -1, 0.5, -0.5]:
                                if a == 0 and b == 0 and c == 0 and d == 0:
                                    continue
                                u_s = {(io, jo): a, (io+1, jo+1): b}
                                v_s = {(jo, ko): c, (jo+1, ko+1): d}
                                w_s = {(ko, io): 1, (ko+1, io+1): 1}
                                add_candidate(u_s, v_s, w_s)

    # Type 6: 2x3x2 structured blocks
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                for perm in S3_elements:
                    u_s = {}; v_s = {}; w_s = {}
                    for jj in range(3):
                        u_s[(io, jo+jj)] = 1
                        u_s[(io+1, jo+jj)] = 1
                    for jj in range(3):
                        for kk in range(2):
                            if perm[jj] == kk:
                                v_s[(jo+jj, ko+kk)] = 1
                    for kk in range(2):
                        for ii in range(2):
                            w_s[(ko+kk, io+ii)] = 1
                    add_candidate(u_s, v_s, w_s)

    num_candidates = len(candidates)
    # Precompute candidate contribution matrix and norms
    cand_matrix = np.zeros((total_entries, num_candidates))
    norms = np.zeros(num_candidates)
    for idx in range(num_candidates):
        vec = candidates[idx][3]
        cand_matrix[:, idx] = vec
        norms[idx] = np.dot(vec, vec)

    T_target = np.ones(total_entries)

    # ---- Direct algebraic construction for rank 48 ----
    # Strategy: Split 2x5x6 into blocks and use optimized decompositions
    # Block decomposition:
    # A = [A1 A2] where A1 is 2x3, A2 is 2x2
    # B = [B1; B2] where B1 is 3x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    # R(2x3x6) + R(2x2x6)
    # R(2x2x6) = 3 * R(2x2x2) = 3*7 = 21 (three Strassen blocks)
    # R(2x3x6): Split B1 into three 3x2 blocks: R(2x3x2)*3
    # R(2x3x2): Need to determine. Let's use: (2x2x2) + (2x1x2) = 7+2 = 9? 
    # Or better: 2x3x2 can be done in rank 6 using:
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # This is 12 entries. Lower bound: max(2*2, 2*3, 3*2) = 6
    # Can we achieve rank 6? Let's try:
    # Use Strassen on the 2x2x2 sub-block (rows 0,1 of A, cols 0,1 of B)
    # That handles C11, C12, C21, C22 partially... no, it only uses A[0:2,0:2] and B[0:2,:]
    # Actually 2x2x2 uses A[0:2,0:2] and B[0:2,0:2] giving C[0:2,0:2]
    # We need C[0:2, 0:2] which is 4 entries from 12
    # Remaining: C[0:2, 2:2] doesn't exist since p=2
    # Wait, 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 4 output entries, 9 input entries
    # Rank at least 4 (since C is 2x2)
    # Try rank 6: 
    # M1 = (A11+A22)(B11+B21) -> contributes to C11, C21
    # Hmm, this is getting complex. Let me just use the search.
    
    # For now, rely on the greedy search with improved candidates

    # ---- 2) Greedy selection to get an initial (possibly over-complete) decomposition ----
    residual = T_target.copy()
    selected_indices = []
    selected_coeffs = []

    max_iter = 300
    for _ in range(max_iter):
        if not time_ok():
            break
        dots = cand_matrix.T @ residual
        coeffs = np.zeros(num_candidates)
        valid = norms > 1e-12
        coeffs[valid] = dots[valid] / norms[valid]
        reductions = dots * coeffs
        reductions[~valid] = -1e18
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        best_coeff = coeffs[best_idx]
        if best_reduction < 1e-9:
            break
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * candidates[best_idx][3]
        if np.linalg.norm(residual) < 1e-9:
            break
        # Early stop if we've already exceeded our best known
        if len(selected_indices) >= 54 and np.linalg.norm(residual) > 0.5:
            break

    # ---- 3) Random restarts with randomized candidate ordering to escape local minima ----
    best_rank = len(selected_indices)
    best_solution = (list(selected_indices), list(selected_coeffs))
    if np.linalg.norm(residual) < 1e-9:
        pass  # record current as best

    # Try random restarts with shuffled candidate priority
    rng = np.random.default_rng(42)
    num_restarts = 10
    for restart in range(num_restarts):
        if not time_ok():
            break
        residual_r = T_target.copy()
        sel_r = []
        coeff_r = []
        # Perturb: randomly skip some top candidates
        skip_size = min(10, num_candidates // 10)
        skip_set = set(rng.choice(num_candidates, size=skip_size, replace=False).tolist())
        for _ in range(max_iter):
            if not time_ok():
                break
            dots = cand_matrix.T @ residual_r
            coeffs = np.zeros(num_candidates)
            valid = norms > 1e-12
            coeffs[valid] = dots[valid] / norms[valid]
            reductions = dots * coeffs
            reductions[~valid] = -1e18
            for s in skip_set:
                reductions[s] = -1e18
            best_idx = int(np.argmax(reductions))
            best_reduction = reductions[best_idx]
            best_coeff = coeffs[best_idx]
            if best_reduction < 1e-9:
                break
            sel_r.append(best_idx)
            coeff_r.append(best_coeff)
            residual_r -= best_coeff * candidates[best_idx][3]
            if np.linalg.norm(residual_r) < 1e-9:
                break
            if len(sel_r) >= best_rank and np.linalg.norm(residual_r) > 0.5:
                break
        if len(sel_r) < best_rank and np.linalg.norm(residual_r) < 1e-9:
            best_rank = len(sel_r)
            best_solution = (sel_r, coeff_r)

    # ---- 4) Pruning: iteratively remove redundant terms ----
    def try_prune(indices, coeffs):
        changed = True
        while changed and time_ok():
            changed = False
            for trial in range(len(indices)):
                if not time_ok():
                    break
                test_residual = T_target.copy()
                for r, idx in enumerate(indices):
                    if r != trial:
                        test_residual -= coeffs[r] * candidates[idx][3]
                if np.linalg.norm(test_residual) < 1e-8:
                    indices.pop(trial)
                    coeffs.pop(trial)
                    changed = True
                    break
        return indices, coeffs

    best_indices, best_coeffs = try_prune(list(best_solution[0]), list(best_solution[1]))
    best_rank = len(best_indices)

    # Try pruning with relaxed tolerance (allow small residual to be absorbed)
    def try_prune_relaxed(indices, coeffs, tol=1e-6):
        changed = True
        while changed and time_ok():
            changed = False
            for trial in range(len(indices)):
                if not time_ok():
                    break
                test_residual = T_target.copy()
                for r, idx in enumerate(indices):
                    if r != trial:
                        test_residual -= coeffs[r] * candidates[idx][3]
                if np.linalg.norm(test_residual) < tol:
                    # Try to absorb the residual into remaining terms
                    # This is approximate; only accept if very small
                    if np.linalg.norm(test_residual) < 1e-8:
                        indices.pop(trial)
                        coeffs.pop(trial)
                        changed = True
                        break
        return indices, coeffs

    # ---- 5) Local search: swap one selected term for an unselected one to reduce rank ----
    if time_ok() and best_rank > 10:
        for _ in range(5):  # a few rounds of local search
            if not time_ok():
                break
            improved = False
            for pos in range(len(best_indices)):
                if not time_ok():
                    break
                # Try removing best_indices[pos] and see if we can cover the gap
                removed_idx = best_indices[pos]
                removed_coeff = best_coeffs[pos]
                # Compute residual after removing this term
                res_without = T_target.copy()
                for r, idx in enumerate(best_indices):
                    if r != pos:
                        res_without -= best_coeffs[r] * candidates[idx][3]
                # Try to fill the gap with a single new term
                dots = cand_matrix.T @ res_without
                valid = norms > 1e-12
                coeffs_new = np.zeros(num_candidates)
                coeffs_new[valid] = dots[valid] / norms[valid]
                reductions = dots * coeffs_new
                reductions[~valid] = -1e18
                # Exclude already-selected terms
                for r, idx in enumerate(best_indices):
                    if r != pos:
                        reductions[idx] = -1e18
                best_new = int(np.argmax(reductions))
                best_new_coeff = coeffs_new[best_new]
                # Check if adding this new term makes the residual small
                test_res = res_without - best_new_coeff * candidates[best_new][3]
                if np.linalg.norm(test_res) < 1e-8:
                    # Swap: replace removed_idx with best_new
                    best_indices[pos] = best_new
                    best_coeffs[pos] = best_new_coeff
                    improved = True
                    break
            if not improved:
                break

    # ---- 5b) Two-for-one swap: remove 2 terms, add 1 term ----
    if time_ok() and best_rank > 10:
        for _ in range(3):
            if not time_ok():
                break
            improved = False
            for pos1 in range(len(best_indices)):
                if not time_ok():
                    break
                for pos2 in range(pos1 + 1, len(best_indices)):
                    if not time_ok():
                        break
                    # Remove both
                    res_without = T_target.copy()
                    for r, idx in enumerate(best_indices):
                        if r != pos1 and r != pos2:
                            res_without -= best_coeffs[r] * candidates[idx][3]
                    # Try to fill with a single new term
                    dots = cand_matrix.T @ res_without
                    valid = norms > 1e-12
                    coeffs_new = np.zeros(num_candidates)
                    coeffs_new[valid] = dots[valid] / norms[valid]
                    reductions = dots * coeffs_new
                    reductions[~valid] = -1e18
                    for r, idx in enumerate(best_indices):
                        if r != pos1 and r != pos2:
                            reductions[idx] = -1e18
                    best_new = int(np.argmax(reductions))
                    best_new_coeff = coeffs_new[best_new]
                    test_res = res_without - best_new_coeff * candidates[best_new][3]
                    if np.linalg.norm(test_res) < 1e-8:
                        # Remove two, add one
                        new_indices = []
                        new_coeffs = []
                        for r, idx in enumerate(best_indices):
                            if r != pos1 and r != pos2:
                                new_indices.append(idx)
                                new_coeffs.append(best_coeffs[r])
                        new_indices.append(best_new)
                        new_coeffs.append(best_new_coeff)
                        best_indices = new_indices
                        best_coeffs = new_coeffs
                        best_rank = len(best_indices)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # ---- 6) Verify and return ----
    if len(best_indices) <= 60:
        rank_found = len(best_indices)
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        for r, idx in enumerate(best_indices):
            coeff = best_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec

        # Verify with multiple test cases
        verify_ok = True
        for trial in range(5):
            A_test = np.random.randint(-10, 10, (n, m))
            B_test = np.random.randint(-10, 10, (m, p))
            C_test = A_test @ B_test
            C_calc = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        for r in range(rank_found):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_calc[i, k] = val
            if not np.allclose(C_test, C_calc, atol=1e-5):
                verify_ok = False
                break

        if verify_ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: 54-rank block decomposition
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    current_rank = 0

    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    for k in range(6):
        add_standard_block(0, 2, k)
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
