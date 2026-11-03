"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    # 1. Initialization: Rank Matching
    # Facilities with high flow sums should be placed at locations with low distance sums (central)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum (most central first)
    central_order = np.argsort(b_row_sums, kind="stable")
    
    # Create initial permutation p such that p[i] is the location for facility i
    p = np.empty(n, dtype=np.int64)
    for fac, loc in zip(heavy_order, central_order):
        p[fac] = loc

    # Helper to compute cost
    def calc_cost(p):
        # Cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
        # b_perm[i,j] = b[p[i], p[j]]
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Precompute b columns for fast access? 
    # In the vectorized swap delta, we need b[:, p[i]] and b[:, p[j]].
    # Let's keep b as is.

    # Simulated Annealing Parameters
    # Initial temperature should be high enough to accept bad moves
    # Estimate typical delta magnitude
    # Delta for swap (i,j) is roughly related to a[i,:], a[j,:] and b positions.
    # Let's start with a heuristic T0.
    T = 5000.0 
    T_min = 1.0
    cooling_rate = 0.9995  # Slow cooling
    
    # We will use a vectorized approach to find the best swap or sample swaps.
    # For n=150, O(n^2) = 22500 swaps. 
    # Vectorized delta calculation for all pairs is possible but memory heavy if not careful.
    # Delta(i, j) = sum_k,l a[i,k]a[j,l] (b[p[i],p[l]] + b[p[j],p[k]] - b[p[i],p[k]] - b[p[j],p[l]])
    # This is complex to fully vectorize for all i,j at once without huge intermediate arrays.
    # Instead, we can iterate i, and vectorize over j.
    # Or, just use a fast random sampling or best-improvement with early termination.
    
    # Given n=150, a full scan of 22,500 swaps in a Python loop is too slow (~0.1s per scan).
    # We need to vectorize the delta calculation for a fixed i against all j > i.
    
    # Let's implement a hybrid:
    # For each step, pick a random i? No, best-improvement is better for QAP.
    # Let's try to vectorize the delta for a specific i against all j.
    
    # Delta(i, j) = (a[i] - a[j]) @ (b[:, p[j]] - b[:, p[i]])
    # Wait, the formula is:
    # Cost change when swapping p[i] and p[j]:
    # Let u = p[i], v = p[j].
    # New cost - Old cost = sum_{k != i, j} a[i,k] (b[v, p[k]] - b[u, p[k]]) + sum_{k != i, j} a[j,k] (b[u, p[k]] - b[v, p[k]])
    # + a[i,j] (b[v,v] + b[u,u] - b[u,v] - b[v,u])  -- Note: a[i,i] and b[u,u] terms usually 0 in QAP? 
    # QAP usually has 0 diagonal. Let's assume a[i,i]=0, b[i,i]=0.
    # If a[i,i]=0, the terms involving k=i or k=j in the sums above need care.
    # Standard formula:
    # Delta = sum_{k} a[i,k] (b[v, p[k]] - b[u, p[k]]) + sum_{k} a[j,k] (b[u, p[k]] - b[v, p[k]])
    # - a[i,j] (b[v,u] - b[u,v]) - a[j,i] (b[u,v] - b[v,u]) ... this is getting messy.
    
    # Let's use the simpler matrix form:
    # C = sum a[i,j] b[p[i], p[j]]
    # Swap i, j.
    # The change is:
    # (a[i,:] - a[j,:]) @ (b[:, p[j]] - b[:, p[i]])
    # Let's verify this.
    # Term a[i,k] b[p[i], p[k]] becomes a[i,k] b[p[j], p[k]] for k!=j, and a[i,j] b[p[j], p[j]] for k=j.
    # This vectorized formula (a[i] - a[j]) @ (b[:, p[j]] - b[:, p[i]]) actually computes:
    # sum_k (a[i,k] - a[j,k]) (b[p[j], k] - b[p[i], k])
    # = sum_k a[i,k]b[p[j],k] - sum_k a[i,k]b[p[i],k] - sum_k a[j,k]b[p[j],k] + sum_k a[j,k]b[p[i],k]
    # This matches the change in cost if we ignore the specific indices i and j in the sums?
    # Actually, this formula works perfectly for the swap delta in QAP if a and b are symmetric and diagonals are 0.
    # Let's assume this standard property.
    
    # Strategy:
    # Loop i from 0 to n-1.
    # For each i, compute delta for all j > i in a vectorized manner.
    # delta_j = (a[i] - a[j]) @ (b[:, p[j]] - b[:, p[i]])
    # This is a matrix multiplication: (a[i] - A[j>i, :]) @ (B[p[j>i], :] - b[:, p[i]])
    
    # Precompute B_p = b[:, p] ? No, b is static, p changes.
    # b[:, p[j]] is just column p[j] of b.
    
    # Let's optimize the inner loop.
    # For a fixed i:
    # u = p[i]
    # ai = a[i]
    # bu = b[:, u]
    
    # For j > i:
    # v_j = p[j]
    # aj = a[j]
    # bv_j = b[:, v_j]
    
    # delta = (ai - aj) @ (bv_j - bu)
    # = ai @ bv_j - ai @ bu - aj @ bv_j + aj @ bu
    
    # We can precompute:
    # ai_bu = np.dot(ai, bu)  # scalar
    # aj_bu = A[j>i] @ bu     # vector of shape (n-i-1,)
    # ai_bv = ai @ B_p[j>i, :]  # where B_p = b[:, p[j>i]]
    # aj_bv = np.einsum('ij,ij->i', A[j>i], B_p[j>i, :]) # element-wise sum? No, dot product per row.
    # Actually, A[j>i] is (m, n), B_p[j>i, :] is (m, n).
    # We want row-wise dot products: np.sum(A[j>i] * B_p[j>i, :], axis=1)
    
    # So:
    # delta = ai_bv - ai_bu - aj_bv + aj_bu
    
    # This is fully vectorizable for a given i!
    
    # Let's implement this.
    
    current_step = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        best_delta = 0  # We only accept if delta < 0 (improvement) or with prob exp(-delta/T)
        best_i = -1
        best_j = -1
        
        # Vectorized search for best swap
        # To save time, we might not check all i every step?
        # With n=150, checking all i is 150 iterations.
        # Each iteration involves O(n^2) operations (vectorized).
        # 150 * (150*150) = 3.3M ops per full scan.
        # In NumPy, this is very fast (< 10ms).
        # So we can do a full best-improvement scan every step.
        
        found_improvement = False
        
        for i in range(n - 1):
            u = p[i]
            ai = a[i]
            bu = b[:, u]
            
            # Indices j > i
            js = np.arange(i + 1, n)
            vs = p[js]
            ajs = a[js]
            bvs = b[:, vs]
            
            # Compute terms
            # ai @ bu is scalar
            ai_bu = np.dot(ai, bu)
            
            # aj @ bu for each j
            aj_bu = ajs @ bu  # shape (m,)
            
            # ai @ bv for each j
            # ai is (n,), bvs is (n, m). We want ai @ bvs[:, k] for each k.
            # This is ai @ bvs -> shape (m,)
            ai_bv = ai @ bvs
            
            # aj @ bv for each j
            # ajs is (m, n), bvs is (n, m).
            # We want row-wise dot product: sum_k ajs[k, :] * bvs[:, k]
            # This is np.sum(ajs * bvs.T, axis=1)
            aj_bv = np.sum(ajs * bvs.T, axis=1)
            
            deltas = ai_bv - ai_bu - aj_bv + aj_bu
            
            # Find min delta in this block
            min_idx_local = np.argmin(deltas)
            min_delta_local = deltas[min_idx_local]
            
            if min_delta_local < best_delta:
                best_delta = min_delta_local
                best_i = i
                best_j = js[min_idx_local]
        
        if best_i == -1:
            # No improvement found. This shouldn't happen if we start from non-optimal,
            # but if we are stuck in a local optimum, best_delta >= 0.
            # In SA, we might accept a bad move.
            # For simplicity and robustness, if no improvement, let's try a random swap with SA acceptance.
            # Or just break if we've been stuck for too long?
            # Let's implement basic SA acceptance for the best found (even if positive) if we are stuck?
            # Actually, if best_delta >= 0, we can accept it with prob exp(-best_delta/T).
            
            if best_delta < 0:
                # Apply swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # No negative delta found. 
                # This implies we are at a local minimum for 2-opt.
                # To escape, we need to accept a positive delta (SA) or do a larger move (3-opt).
                # Let's accept the best (least bad) move with SA probability.
                if best_delta < float('inf') and np.random.rand() < np.exp(-best_delta / T):
                    p[best_i], p[best_j] = p[best_j], p[best_i]
                    current_cost += int(best_delta)
                    no_improve_count += 1
                else:
                    # Rejected. Cool down.
                    T *= cooling_rate
                    no_improve_count += 1
                    if no_improve_count > 500:
                         # Perturb
                         for _ in range(5):
                             i = np.random.randint(n)
                             j = np.random.randint(n)
                             if i != j:
                                 p[i], p[j] = p[j], p[i]
                         current_cost = calc_cost(p)
                         if current_cost < best_cost:
                             best_cost = current_cost
                             best_p = p.copy()
                         no_improve_count = 0
                         T = max(T_min, T * 0.5) # Reheat slightly
                    continue
        else:
            # We found a negative delta
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += int(best_delta)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Cool down slowly
            T *= cooling_rate
            if T < T_min:
                T = T_min

        current_step += 1
        # Optional: Log progress
        # if current_step % 100 == 0:
        #     print(f"Step {current_step}, Cost {current_cost}, T {T}")

    return best_p.tolist()
# EVOLVE-BLOCK-END
