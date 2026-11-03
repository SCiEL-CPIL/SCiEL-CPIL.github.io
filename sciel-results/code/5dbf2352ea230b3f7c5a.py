"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 118.0  # Safety margin before 120s kill

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        sum_all = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost, max_iterations=50):
        # Best Improvement Local Search
        # We scan all pairs (i, j) with i < j. We find the globally best improving swap.
        # If found, we apply it and restart the scan.
        # This is O(n^3) per iteration (n^2/2 swaps * O(n) cost eval).
        # For n=100, that's 5000 * 100 = 500k ops per iteration.
        # Max 50 iterations -> 25M ops. Should be fast enough.
        
        iterations = 0
        while iterations < max_iterations:
            if time.time() - start_time > timeout:
                break
            
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                i, j = best_i, best_j
                p[i], p[j] = p[j], p[i]
                current_cost += best_delta
                iterations += 1
            else:
                break
        
        return p, current_cost

    def lap_perturb(p, current_cost, noise_scale=0.0):
        """
        Generate a new solution by solving the Linear Assignment Problem
        on the linearized cost matrix, possibly with noise.
        """
        # Pre-computed C is passed or computed here.
        # C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
        # This is a static surrogate.
        # To make it dynamic and guide away from local minima, we could adjust C,
        # but for now we rely on noise.
        
        # C is computed once outside
        C_noisy = C_surrogate + np.random.normal(0, noise_scale, (n, n))
        
        row_ind, col_ind = linear_sum_assignment(C_noisy)
        p_new = np.zeros(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        return p_new

    # 1. Initial Solution: LAP on linear surrogate
    # C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    # Term 1: (a @ b.T)[i, j]
    # Term 2: (a.T @ b)[i, j]
    C_surrogate = (a @ b.T) + (a.T @ b)
    
    row_ind, col_ind = linear_sum_assignment(C_surrogate)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    noise_scale = 0.0
    
    # We will alternate between small random perturbations and LAP perturbations.
    # Also, we can try to "refine" the surrogate C based on the current best solution?
    # A better surrogate for a specific solution p is:
    # C_p[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
    # This is the cost of placing i at j, assuming others stay at p.
    # This is O(n^3) to compute.
    # If we solve LAP on C_p, we get a 2-opt improvement essentially?
    # Actually, solving LAP on C_p finds the best single assignment change (or set of changes)
    # that minimizes cost given the rest are fixed? No, LAP allows multiple changes.
    # This is a powerful move. Let's call it "LAP-Refine".
    
    def lap_refine(p, current_cost):
        """
        Compute C_p and solve LAP.
        C_p[i, j] = cost contribution of assigning i to j, given p for others.
        """
        # C_p[i, j] = sum_{k != i} a[i, k] * b[j, p[k]] + sum_{k != i} a[k, i] * b[p[k], j] + a[i, i] * b[j, j]
        # Wait, the diagonal term a[i,i]b[j,j] is included in the sum if k=i?
        # In the sum sum_k a[i,k] b[j, p[k]], if k=i, term is a[i,i] b[j, p[i]].
        # But if we change p[i] to j, then p[i] is no longer p[i].
        # The standard linearization for "fix others" is:
        # C_p[i, j] = sum_{k != i} a[i, k] b[j, p[k]] + sum_{k != i} a[k, i] b[p[k], j] + a[i, i] b[j, j]
        
        # Let's compute this efficiently.
        # Let B_p = b[np.ix_(p, p)]? No.
        # We need b[j, p[k]] for all j, k.
        # This is b[:, p]. Shape (n, n).
        # Let's call it B_loc. B_loc[j, k] = b[j, p[k]].
        
        B_loc = b[:, p]
        
        # Term 1: sum_{k != i} a[i, k] * B_loc[j, k]
        # This is (a @ B_loc.T)[i, j] - a[i, i] * B_loc[j, i]
        # Because (a @ B_loc.T)[i, j] = sum_k a[i, k] B_loc.T[k, j] = sum_k a[i, k] B_loc[j, k]
        
        # Term 2: sum_{k != i} a[k, i] * b[p[k], j]
        # Note b[p[k], j] is B_loc.T[k, j] ? No.
        # B_loc[j, k] = b[j, p[k]].
        # We need b[p[k], j]. This is B_loc[j, k] if b is symmetric? No.
        # b[p[k], j] is the element at row p[k], col j of b.
        # Let's define B_loc2[k, j] = b[p[k], j].
        # B_loc2 = b[p, :]
        
        B_loc2 = b[p, :]
        
        # Term 2: sum_{k != i} a[k, i] * B_loc2[k, j]
        # This is (a.T @ B_loc2)[i, j] - a[i, i] * B_loc2[i, j]
        # (a.T @ B_loc2)[i, j] = sum_k a.T[i, k] B_loc2[k, j] = sum_k a[k, i] b[p[k], j]
        
        # So C_p = (a @ B_loc.T) + (a.T @ B_loc2)
        # Subtract the k=i terms which are invalid because p[i] changes to j.
        # Term 1 k=i: a[i, i] * B_loc[j, i] = a[i, i] * b[j, p[i]]
        # Term 2 k=i: a[i, i] * B_loc2[i, j] = a[i, i] * b[p[i], j]
        
        # And add the self-term a[i, i] * b[j, j]
        
        C_p = (a @ B_loc.T) + (a.T @ B_loc2)
        
        # Subtract invalid terms
        # For each i, j:
        # C_p[i, j] -= a[i, i] * (b[j, p[i]] + b[p[i], j] - b[j, j])
        
        # Let's do this vectorized.
        # b[j, p[i]] is B_loc[j, i].
        # b[p[i], j] is B_loc2[i, j].
        # b[j, j] is np.diag(b).
        
        diag_b = np.diag(b)
        
        # Matrix of a[i, i] * b[j, p[i]]
        # a_ii = np.diag(a)
        # M1 = a_ii[:, None] * B_loc  -> M1[i, j] = a[i, i] * B_loc[j, i] = a[i, i] * b[j, p[i]]
        # M2 = a_ii[:, None] * B_loc2 -> M2[i, j] = a[i, i] * B_loc2[i, j] = a[i, i] * b[p[i], j]
        # M3 = a_ii[:, None] * diag_b[None, :] -> M3[i, j] = a[i, i] * b[j, j]
        
        a_ii = np.diag(a)
        
        C_p -= a_ii[:, None] * B_loc
        C_p -= a_ii[:, None] * B_loc2
        C_p += a_ii[:, None] * diag_b[None, :]
        
        # Solve LAP on C_p
        row_ind, col_ind = linear_sum_assignment(C_p)
        p_new = np.zeros(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        return p_new

    # Now ILS
    no_improve_count = 0
    # Use a smaller timeout margin to allow more iterations
    inner_timeout = timeout - 10.0
    
    while time.time() - start_time < inner_timeout:
        # Perturbation
        if no_improve_count == 0:
            # Small random perturbation
            num_swaps = 2
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j: continue
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        elif no_improve_count < 5:
            # Medium perturbation: more swaps
            num_swaps = 5
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j: continue
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        else:
            # Global perturbation: LAP with noise
            noise_scale = 1000.0 * (no_improve_count / 5.0)
            p = lap_perturb(p, current_cost, noise_scale)
            current_cost = calc_cost(p)
        
        # Improvement: local_search
        # We removed lap_refine from the main loop to save time, 
        # relying on the better local_search.
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset to best to avoid wandering too far?
            # Or keep current p?
            # If we reset, we are doing ILS from best.
            p = best_p.copy()
            current_cost = best_cost
            
        if time.time() - start_time > timeout - 5.0:
            break
    
    # Final polish: Try lap_refine once more on the best solution if time permits
    if time.time() - start_time < timeout - 2.0:
        p_refined = lap_refine(best_p, best_cost)
        cost_refined = calc_cost(p_refined)
        if cost_refined < best_cost:
            best_cost = cost_refined
            best_p = p_refined.copy()
            # Polish with local search
            p, current_cost = local_search(best_p, best_cost)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
