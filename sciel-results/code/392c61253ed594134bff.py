"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations, product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_strassen_decomposition(i_groups=None, j_groups=None, k_groups=None):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        if i_groups is None:
            i_groups = [[0, 1], [2]]
        if j_groups is None:
            j_groups = [[0, 1], [2, 3], [4]]
        if k_groups is None:
            k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def find_best_partition():
        """Search over partitions to find the one with minimum rank."""
        if time.time() > deadline:
            return None
        
        def generate_partitions(dim):
            """Generate all partitions of range(dim) into contiguous groups."""
            partitions = []
            for g in range(1, dim + 1):
                for dividers in combinations(range(dim - 1), g - 1):
                    groups = []
                    prev = 0
                    for d in dividers:
                        groups.append(list(range(prev, d + 1)))
                        prev = d + 1
                    groups.append(list(range(prev, dim)))
                    partitions.append(groups)
            return partitions
        
        i_parts = generate_partitions(n)
        j_parts = generate_partitions(m)
        k_parts = generate_partitions(p)
        
        best_rank = float('inf')
        best_partition = None
        
        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    total = 0
                    for ig in ip:
                        for jg in jp:
                            for kg in kp:
                                si, sj, sk = len(ig), len(jg), len(kg)
                                if si == 2 and sj == 2 and sk == 2:
                                    total += 7
                                else:
                                    total += si * sj * sk
                    if total < best_rank:
                        best_rank = total
                        best_partition = (ip, jp, kp)
        
        if best_partition is None:
            return None
        
        ip, jp, kp = best_partition
        result = build_strassen_decomposition(ip, jp, kp)
        rank = result.pop("rank")
        
        # Verify
        if verify(result["U"], result["V"], result["W"], rank):
            return result, rank
        
        return None

    def als_tensor_rank(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-12):
        """
        ALS-based tensor rank minimization.
        Tries to find rank-1 decomposition of the matrix multiplication tensor.
        """
        if time.time() > deadline - 5:
            return None
        
        U = np.array(U_init, dtype=np.float64)
        V = np.array(V_init, dtype=np.float64)
        W = np.array(W_init, dtype=np.float64)
        
        # Ensure correct shapes
        if U.shape != (n*m, target_rank):
            U = np.zeros((n*m, target_rank))
        if V.shape != (m*p, target_rank):
            V = np.zeros((m*p, target_rank))
        if W.shape != (p*n, target_rank):
            W = np.zeros((p*n, target_rank))
        
        # Precompute tensor indices
        tensor_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    tensor_indices.append((u_idx, v_idx, w_idx))
        
        def compute_error():
            err = 0.0
            for (ui, vi, wi) in tensor_indices:
                actual = 0.0
                for c in range(target_rank):
                    actual += U[ui, c] * V[vi, c] * W[wi, c]
                err += (actual - 1.0) ** 2
            return err / len(tensor_indices)
        
        prev_err = compute_error()
        if prev_err < tol:
            return (U, V, W, prev_err)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for ui in range(n * m):
                # For each column c, solve for U[ui, c]
                # Sum over c: U[ui,c] * V[vi,c] * W[wi,c] = 1 for all (vi,wi) pairs
                # This is a linear system in U[ui, :]
                # Build matrix M where M[a,b] = sum over tensor entries involving ui: V[vi,a]*W[wi,b]
                # Actually, we need: for all (vi, wi) in tensor, sum_c U[ui,c] * V[vi,c] * W[wi,c] = 1
                # Fix V and W, solve for U[ui, :]
                
                # Collect all (vi, wi) pairs that involve this ui
                # For a given ui = i*m + j, the valid (vi, wi) are:
                # vi = j*p + k for k in 0..p-1, wi = k*n + i for k in 0..p-1
                # So we have p equations
                
                # Build the system: sum_c U[ui,c] * (V[vi,c] * W[wi,c]) = 1
                # This is: A * U[ui,:] = b where A[row, c] = V[vi_row, c] * W[wi_row, c]
                
                i_val = ui // m
                j_val = ui % m
                
                rows = []
                for k in range(p):
                    vi = j_val * p + k
                    wi = k * n + i_val
                    row = V[vi, :] * W[wi, :]
                    rows.append(row)
                
                A = np.array(rows)  # shape (p, target_rank)
                b = np.ones(p)
                
                # Solve least squares
                if target_rank <= p:
                    # Underdetermined or square, use lstsq
                    U_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[ui, :] = U_new
                else:
                    # Overdetermined, use lstsq
                    U_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[ui, :] = U_new
            
            # Update V
            for vi in range(m * p):
                j_val = vi // p
                k_val = vi % p
                
                rows = []
                for i in range(n):
                    for jj in range(m):
                        # We need all (ui, wi) such that vi is fixed
                        # ui = i*m + jj, wi = k_val*n + i
                        # But we only want valid tensor entries where the middle index matches
                        # Actually, for a given vi = j*p + k, the valid (ui, wi) are:
                        # ui = j*m + i for i in 0..n-1 (since ui = i*m + j, so i ranges and j is fixed)
                        # Wait, ui = i*m + j, so for fixed j, i ranges 0..n-1
                        # wi = k*n + i for i in 0..n-1
                        pass
                
                # For vi = j*p + k:
                # The tensor entries with this vi are: (i, j, k) for i in 0..n-1
                # ui = i*m + j, wi = k*n + i
                rows = []
                for i in range(n):
                    ui = i * m + j_val
                    wi = k_val * n + i
                    row = U[ui, :] * W[wi, :]
                    rows.append(row)
                
                A = np.array(rows)  # shape (n, target_rank)
                b = np.ones(n)
                
                if target_rank <= n:
                    V_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[vi, :] = V_new
                else:
                    V_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[vi, :] = V_new
            
            # Update W
            for wi in range(p * n):
                k_val = wi // n
                i_val = wi % n
                
                # For wi = k*n + i:
                # The tensor entries with this wi are: (i, j, k) for j in 0..m-1
                # ui = i*m + j, vi = j*p + k
                rows = []
                for j in range(m):
                    ui = i_val * m + j
                    vi = j * p + k_val
                    row = U[ui, :] * V[vi, :]
                    rows.append(row)
                
                A = np.array(rows)  # shape (m, target_rank)
                b = np.ones(m)
                
                if target_rank <= m:
                    W_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[wi, :] = W_new
                else:
                    W_new, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[wi, :] = W_new
            
            # Compute error
            curr_err = compute_error()
            
            if abs(prev_err - curr_err) < tol:
                break
            prev_err = curr_err
            
            if curr_err < tol:
                break
        
        final_err = compute_error()
        return (U, V, W, final_err)

    def try_als_from_block(start_rank, max_trials=5):
        """
        Try ALS starting from the block decomposition, attempting to reduce rank.
        """
        if time.time() > deadline - 20:
            return None
        
        best_result = None
        best_rank = float('inf')
        
        # Get the block decomposition as starting point
        block_result, block_rank = None, None
        partition_result = find_best_partition()
        if partition_result is not None:
            block_result, block_rank = partition_result
        else:
            block_result = build_strassen_decomposition()
            block_rank = block_result.pop("rank")
        
        if not verify(block_result["U"], block_result["V"], block_result["W"], block_rank):
            # Fallback to schoolbook
            school = schoolbook()
            block_result = school
            block_rank = n * m * p
        
        U_block = np.array(block_result["U"], dtype=np.float64)
        V_block = np.array(block_result["V"], dtype=np.float64)
        W_block = np.array(block_result["W"], dtype=np.float64)
        
        # Try various target ranks below the block rank
        for target_rank in range(max(90, block_rank - 15), min(block_rank, 100)):
            if time.time() > deadline - 15:
                break
            
            # Initialize with the block factors, truncated/padded to target_rank
            if target_rank <= block_rank:
                U_init = U_block[:, :target_rank]
                V_init = V_block[:, :target_rank]
                W_init = W_block[:, :target_rank]
            else:
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                U_init[:, :block_rank] = U_block
                V_init[:, :block_rank] = V_block
                W_init[:, :block_rank] = W_block
                # Add small random perturbation to new columns
                U_init[:, block_rank:] = np.random.randn(n*m, target_rank - block_rank) * 0.01
                V_init[:, block_rank:] = np.random.randn(m*p, target_rank - block_rank) * 0.01
                W_init[:, block_rank:] = np.random.randn(p*n, target_rank - block_rank) * 0.01
            
            # Run ALS
            result = als_tensor_rank(U_init, V_init, W_init, target_rank, max_iters=150)
            
            if result is not None:
                U_als, V_als, W_als, err = result
                if err < 1e-9:
                    # Verify with exact check
                    U_list = U_als.tolist()
                    V_list = V_als.tolist()
                    W_list = W_als.tolist()
                    
                    # Round to integers if close
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for c in range(len(row)):
                                if abs(row[c] - round(row[c])) < 1e-8:
                                    row[c] = round(row[c])
                    
                    if verify(U_list, V_list, W_list, target_rank):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
        
        return best_result

    def try_random_als(target_rank, max_trials=10):
        """
        Try random initialization ALS for a given target rank.
        """
        if time.time() > deadline - 10:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to help convergence
            U /= np.sqrt(np.sum(U**2))
            V /= np.sqrt(np.sum(V**2))
            W /= np.sqrt(np.sum(W**2))
            
            result = als_tensor_rank(U, V, W, target_rank, max_iters=100)
            
            if result is not None:
                U_als, V_als, W_als, err = result
                if err < 1e-9:
                    U_list = U_als.tolist()
                    V_list = V_als.tolist()
                    W_list = W_als.tolist()
                    
                    # Round to integers if close
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for c in range(len(row)):
                                if abs(row[c] - round(row[c])) < 1e-8:
                                    row[c] = round(row[c])
                    
                    if verify(U_list, V_list, W_list, target_rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try improved decomposition via partition search
    partition_result = find_best_partition()
    if partition_result is not None:
        result, rank = partition_result
        if rank < best_rank:
            best_result = result
            best_rank = rank

    # Try ALS from the block decomposition
    if time.time() < deadline - 30:
        als_result = try_als_from_block(best_rank)
        if als_result is not None:
            als_rank = len(als_result["U"][0])
            if als_rank < best_rank:
                best_result = als_result
                best_rank = als_rank

    # Try random ALS for even lower ranks
    if time.time() < deadline - 15:
        for trial_rank in range(max(90, best_rank - 10), best_rank):
            if time.time() > deadline - 5:
                break
            result = try_random_als(trial_rank, max_trials=3)
            if result is not None:
                if trial_rank < best_rank:
                    best_result = result
                    best_rank = trial_rank
                    break

    return best_result
# EVOLVE-BLOCK-END
