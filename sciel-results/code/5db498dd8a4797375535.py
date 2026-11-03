# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 110.0  # safety margin

    # --- Initial solution via rank matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    # --- Compute initial cost ---
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    cost = compute_cost(p)

    # --- Precompute column sums for b ---
    b_col_sums = b.sum(axis=0)

    # --- Tabu Search ---
    tabu_list = {}  # (i,j) -> iteration when it will no longer be tabu
    tabu_tenure = 30
    best_cost = cost
    best_p = p.copy()
    no_improve_count = 0
    max_no_improve = 2000

    # Precompute a for faster access
    a_arr = a
    b_arr = b

    def get_move_delta(i, j):
        """Compute delta for swapping i and j."""
        li = p[i]
        lj = p[j]
        # Change in cost = a[i,j]*(b[lj,lj] - b[li,li] - b[lj,li] + b[li,lj])
        #                  + a[j,i]*(b[li,li] - b[lj,lj] - b[li,lj] + b[lj,li])
        # Since b[x,x]=0:
        # = a[i,j]*(b[li,lj] - b[lj,li]) + a[j,i]*(b[lj,li] - b[li,lj])
        # = (a[i,j] - a[j,i]) * (b[li,lj] - b[lj,li])
        # But wait, this only accounts for the (i,j) and (j,i) pair.
        # Actually, we need the full change. Let me use the C matrix approach.
        
        # Use the maintained C matrix approach:
        # C[i, L] = sum_k a[k,i] * b[k, L]
        # Delta = C[i, lj] + C[j, li] - C[i, li] - C[j, lj]
        # But we don't maintain C here. Let me compute directly.
        
        # Direct computation:
        # New cost terms involving i or j change.
        # The change is:
        # sum_k a[k,i] * (b[k, lj] - b[k, li]) + a[i,k] * (b[lj, k] - b[li, k])
        # + sum_k a[k,j] * (b[k, li] - b[k, lj]) + a[j,k] * (b[li, k] - b[lj, k])
        # - a[i,j]*(b[li,li] - b[lj,lj] - b[li,lj] + b[lj,li])
        # - a[j,i]*(b[lj,lj] - b[li,li] - b[lj,li] + b[li,lj])
        
        # Since b[x,x] = 0:
        # = sum_k a[k,i]*(b[k,lj]-b[k,li]) + a[i,k]*(b[lj,k]-b[li,k])
        #   + sum_k a[k,j]*(b[k,li]-b[k,lj]) + a[j,k]*(b[li,k]-b[lj,k])
        #   - a[i,j]*(b[li,lj]-b[lj,li]) - a[j,i]*(b[lj,li]-b[li,lj])
        
        # = sum_k a[k,i]*(b[k,lj]-b[k,li]) + sum_k a[i,k]*(b[lj,k]-b[li,k])
        #   + sum_k a[k,j]*(b[k,li]-b[k,lj]) + sum_k a[j,k]*(b[li,k]-b[lj,k])
        #   - (a[i,j]-a[j,i])*(b[li,lj]-b[lj,li])
        
        # Note that sum_k a[k,i]*(b[k,lj]-b[k,li]) = (a[:,i] @ (b[:,lj]-b[:,li]))
        # And sum_k a[i,k]*(b[lj,k]-b[li,k]) = (a[i,:] @ (b[lj,:]-b[li,:]))
        
        diff_lj_li = b[:, lj] - b[:, li]
        diff_lj_li_row = b[lj, :] - b[li, :]
        
        part1 = a[:, i] @ diff_lj_li
        part2 = a[i, :] @ diff_lj_li_row
        part3 = a[:, j] @ (b[:, li] - b[:, lj])
        part4 = a[j, :] @ (b[li, :] - b[lj, :])
        part5 = -(a[i, j] - a[j, i]) * (b[li, lj] - b[lj, li])
        
        return part1 + part2 + part3 + part4 + part5

    # For efficiency, let's maintain the C matrix
    # C[i, L] = sum_k a[k, i] * b[k, L]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        C[i, :] = a[:, i] @ b

    def get_move_delta_fast(i, j):
        li = p[i]
        lj = p[j]
        return C[i, lj] + C[j, li] - C[i, li] - C[j, lj]

    def update_C(i, j, li_old, lj_old):
        """Update C after swapping p[i] and p[j]."""
        # li_old = p[i] before swap, lj_old = p[j] before swap
        # After swap: p[i] = lj_old, p[j] = li_old
        # C[k, L] = sum_m a[m, k] * b[m, L]
        # The change: for k=i, the column index in b changes from li_old to lj_old
        # Actually, C is defined as C[i, L] = sum_k a[k, i] * b[k, L]
        # This doesn't depend on p! So C doesn't change with p.
        # Wait, that means C is constant. Let me re-derive.
        # 
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # 
        # Let's define for each facility i and location L:
        # f(i, L) = sum_j a[i,j] * b[L, p[j]]
        # Then cost = sum_i f(i, p[i])
        # 
        # If we swap p[i] and p[j] (li <-> lj):
        # New f(i, lj) = sum_k a[i,k] * b[lj, p[k]]
        #   = sum_{k!=i,j} a[i,k]*b[lj, p[k]] + a[i,i]*b[lj, p[i]] + a[i,j]*b[lj, p[j]]
        #   = sum_{k!=i,j} a[i,k]*b[lj, p[k]] + a[i,i]*b[lj, li] + a[i,j]*b[lj, lj]
        # Old f(i, li) = sum_k a[i,k] * b[li, p[k]]
        #   = sum_{k!=i,j} a[i,k]*b[li, p[k]] + a[i,i]*b[li, p[i]] + a[i,j]*b[li, p[j]]
        #   = sum_{k!=i,j} a[i,k]*b[li, p[k]] + a[i,i]*b[li, li] + a[i,j]*b[li, lj]
        # 
        # This is getting complicated. Let me just use the direct O(n) formula.
        pass

    # Let me use a simpler approach: maintain cost and use O(n) delta computation
    def compute_delta(i, j):
        li = p[i]
        lj = p[j]
        # Delta = sum_k a[k,i]*(b[k,lj]-b[k,li]) + a[i,k]*(b[lj,k]-b[li,k])
        #         + sum_k a[k,j]*(b[k,li]-b[k,lj]) + a[j,k]*(b[li,k]-b[lj,k])
        #         - (a[i,j]-a[j,i])*(b[li,lj]-b[lj,li])
        
        d1 = np.dot(a[:, i], b[:, lj] - b[:, li])
        d2 = np.dot(a[i, :], b[lj, :] - b[li, :])
        d3 = np.dot(a[:, j], b[:, li] - b[:, lj])
        d4 = np.dot(a[j, :], b[li, :] - b[lj, :])
        d5 = -(a[i, j] - a[j, i]) * (b[li, lj] - b[lj, li])
        return d1 + d2 + d3 + d4 + d5

    # Simpler: just compute the full cost change directly
    def compute_delta_simple(i, j):
        li = p[i]
        lj = p[j]
        # Old contribution of terms involving i or j
        # Terms where first index is i: sum_j a[i,j] * b[li, p[j]]
        # Terms where second index is i: sum_j a[j,i] * b[p[j], li]
        # Similarly for j.
        # When we swap, p[i] becomes lj and p[j] becomes li.
        # 
        # Let's just compute the difference directly using the full sum
        # but that's O(n^2). For n=100, that's 10000 ops per delta, which is fine.
        
        # Old cost contributions involving facility i at location li and j at lj
        old_ii = np.dot(a[i, :], b[li, p])
        old_jj = np.dot(a[j, :], b[lj, p])
        old_ii_rev = np.dot(a[:, i], b[p, li])
        old_jj_rev = np.dot(a[:, j], b[p, lj])
        # Subtract double-counted terms
        old_ii_jj = a[i, j] * b[li, lj]
        old_ji_ij = a[j, i] * b[lj, li]
        old_total = old_ii + old_jj + old_ii_rev + old_jj_rev - old_ii_jj - old_ji_ij
        
        # New: i at lj, j at li
        p_new = p.copy()
        p_new[i] = lj
        p_new[j] = li
        
        new_ii = np.dot(a[i, :], b[lj, p_new])
        new_jj = np.dot(a[j, :], b[li, p_new])
        new_ii_rev = np.dot(a[:, i], b[p_new, lj])
        new_jj_rev = np.dot(a[:, j], b[p_new, li])
        new_ii_jj = a[i, j] * b[lj, li]
        new_ji_ij = a[j, i] * b[li, lj]
        new_total = new_ii + new_jj + new_ii_rev + new_jj_rev - new_ii_jj - new_ji_ij
        
        return new_total - old_total

    # Actually, let's just do 2-opt with full cost recomputation for simplicity
    # since n=100, O(n^2) per evaluation is 10000 ops, and we can do many iterations.
    
    def full_cost(p_arr):
        return np.sum(a * b[np.ix_(p_arr, p_arr)])

    # 2-opt search
    best_p = p.copy()
    best_cost = cost
    improved = True
    iter_count = 0
    
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if (time.time() - start_time) > deadline:
                    break
                # Try swap
                p[j], p[i] = p[i], p[j]
                new_cost = full_cost(p)
                p[j], p[i] = p[i], p[j]
                
                if new_cost < cost:
                    cost = new_cost
                    p[j], p[i] = p[i], p[j]
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    improved = True
                    iter_count += 1
        if not improved:
            # Perturbation
            if (time.time() - start_time) > deadline - 10:
                break
            # Random restart with perturbation
            num_swaps = np.random.randint(5, 15)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            cost = full_cost(p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
