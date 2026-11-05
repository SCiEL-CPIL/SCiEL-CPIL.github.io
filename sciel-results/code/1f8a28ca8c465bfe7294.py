import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # First, build the rank-66 decomposition as a known-valid baseline
    rank = 66
    U_base = [[0] * rank for _ in range(n * m)]
    V_base = [[0] * rank for _ in range(m * p)]
    W_base = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U_base[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V_base[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W_base[idx][current_rank] += val
        current_rank += 1
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
    
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    # Verify baseline
    if not check_deadline():
        return {"U": U_base, "V": V_base, "W": W_base}
    
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(current_rank):
                    total += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Now attempt GF(2) greedy forward-selection to find a lower-rank binary decomposition
    if not check_deadline():
        return {"U": U_base, "V": V_base, "W": W_base}
    
    n_entries = n * m * p  # 72
    
    # Precompute all possible rank-1 tensors over GF(2)
    # A rank-1 tensor is defined by vectors u in GF(2)^{n*m}, v in GF(2)^{m*p}, w in GF(2)^{p*n}
    # The tensor entry T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    # There are 2^12 * 2^24 * 2^18 = 2^54 possible rank-1 tensors, too many to enumerate
    # Instead, use a restricted search: consider rank-1 tensors from the structure of the problem
    
    # Strategy: Generate candidate rank-1 tensors from structured patterns
    # For each (i,j,k) triple, the target tensor has T[i,j,k] = 1
    # We want to find a set of rank-1 tensors whose XOR sum equals the target
    
    # Generate candidates based on the block structure
    candidates = []
    
    # Type 1: Single entry rank-1 tensors (schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                # Compute tensor
                tensor = np.zeros(n_entries, dtype=np.int8)
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            val = u[ii*m+jj] * v[jj*p+kk] * w[kk*n+ii]
                            if val:
                                idx = ii * m * p + jj * p + kk
                                tensor[idx] = 1
                candidates.append((tensor, u, v, w))
    
    # Type 2: Pairs of entries (like Strassen terms)
    for i1 in range(n):
        for i2 in range(n):
            if i1 >= i2:
                continue
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i1 * m + j] = 1
                    u[i2 * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i1] = 1
                    w[k * n + i2] = 1
                    tensor = np.zeros(n_entries, dtype=np.int8)
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                val = u[ii*m+jj] * v[jj*p+kk] * w[kk*n+ii]
                                if val:
                                    idx = ii * m * p + jj * p + kk
                                    tensor[idx] = 1
                    candidates.append((tensor, u, v, w))
    
    # Type 3: Cross terms (Strassen-style)
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2:
                continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 >= j2:
                        continue
                    for k in range(p):
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        u[i1 * m + j1] = 1
                        u[i2 * m + j2] = 1
                        v[j1 * p + k] = 1
                        v[j2 * p + k] = 1
                        w[k * n + i1] = 1
                        w[k * n + i2] = 1
                        tensor = np.zeros(n_entries, dtype=np.int8)
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    val = u[ii*m+jj] * v[jj*p+kk] * w[kk*n+ii]
                                    if val:
                                        idx = ii * m * p + jj * p + kk
                                        tensor[idx] = 1
                        candidates.append((tensor, u, v, w))
    
    if not check_deadline():
        return {"U": U_base, "V": V_base, "W": W_base}
    
    # Target tensor over GF(2): T[i,j,k] = 1 for all (i,j,k)
    target = np.ones(n_entries, dtype=np.int8)
    
    # Greedy forward selection
    best_rank = current_rank  # Start with baseline rank
    best_U = [row[:] for row in U_base]
    best_V = [row[:] for row in V_base]
    best_W = [row[:] for row in W_base]
    
    # For the greedy search, we work over GF(2)
    # residual = target XOR (sum of selected rank-1 tensors)
    residual = target.copy()
    selected = []
    
    # Precompute candidate tensors as a numpy array for faster operations
    n_cands = len(candidates)
    cand_tensors = np.array([c[0] for c in candidates], dtype=np.int8)
    
    # Greedy: at each step, pick the candidate with highest inner product with residual
    max_iter = min(n_cands, 200)  # Limit iterations for time
    for step in range(max_iter):
        if not check_deadline():
            break
        if step + 1 >= best_rank:
            break
        
        # Compute inner products (over GF(2), this is just XOR sum mod 2)
        # Actually, for greedy, we want to maximize the number of 1s we can fix
        # Inner product over reals: count of positions where both are 1
        # But we need to be careful: we're doing XOR, so we want to minimize residual
        
        # For each candidate, compute how many 1s in residual it can cover
        # This is the Hamming weight of (cand & residual)
        # But we also need to consider that adding a candidate might introduce new 1s
        
        # Simpler approach: just try to find a candidate that reduces the residual weight
        residual_weight = np.sum(residual)
        
        if residual_weight == 0:
            break
        
        # Find best candidate
        best_idx = -1
        best_improvement = 0
        
        # Sample candidates for speed if there are too many
        if n_cands > 1000:
            sample_size = 1000
            sample_indices = np.random.choice(n_cands, size=sample_size, replace=False)
        else:
            sample_indices = np.arange(n_cands)
        
        for idx in sample_indices:
            cand = cand_tensors[idx]
            # New residual = residual XOR cand
            new_residual = residual ^ cand
            new_weight = np.sum(new_residual)
            improvement = residual_weight - new_weight
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = idx
        
        if best_idx == -1 or best_improvement <= 0:
            break
        
        # Add this candidate
        selected.append(best_idx)
        residual = residual ^ cand_tensors[best_idx]
        
        # Build the decomposition
        if len(selected) + 1 < best_rank:
            U_new = [[0] * (len(selected) + 1) for _ in range(n * m)]
            V_new = [[0] * (len(selected) + 1) for _ in range(m * p)]
            W_new = [[0] * (len(selected) + 1) for _ in range(p * n)]
            
            for r, cand_idx in enumerate(selected):
                _, u, v, w = candidates[cand_idx]
                for idx_u, val in enumerate(u):
                    if val:
                        U_new[idx_u][r] = 1
                for idx_v, val in enumerate(v):
                    if val:
                        V_new[idx_v][r] = 1
                for idx_w, val in enumerate(w):
                    if val:
                        W_new[idx_w][r] = 1
            
            best_rank = len(selected) + 1
            best_U = U_new
            best_V = V_new
            best_W = W_new
    
    # Verify the greedy result
    if best_rank < current_rank and check_deadline():
        U_test = np.array(best_U, dtype=float)
        V_test = np.array(best_V, dtype=float)
        W_test = np.array(best_W, dtype=float)
        
        valid_greedy = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(best_rank):
                        total += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        valid_greedy = False
                        break
                if not valid_greedy:
                    break
            if not valid_greedy:
                break
        
        if valid_greedy:
            return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return the baseline
    return {"U": U_base, "V": V_base, "W": W_base}
