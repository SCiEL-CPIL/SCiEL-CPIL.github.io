import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Check if B is symmetric
    is_sym = np.allclose(b, b.T)

    # Spectral embedding initialization
    try:
        A_sym_init = a + a.T
        B_sym_init = b + b.T

        k_eig = min(8, n - 1)
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym_init)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym_init)

        # Take eigenvectors with largest absolute eigenvalues
        idx_A = np.argsort(np.abs(eigvals_A))[::-1][:k_eig]
        idx_B = np.argsort(np.abs(eigvals_B))[::-1][:k_eig]

        V_A = eigvecs_A[:, idx_A]  # (n, k_eig)
        V_B = eigvecs_B[:, idx_B]  # (n, k_eig)

        # Build assignment cost: align spectral signatures
        C_spectral = -V_A @ V_B.T  # (n, n)

        row_ind, col_ind = linear_sum_assignment(C_spectral)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind

    except Exception:
        # Fallback to rank-based initialization
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_sums, kind='stable')
        central_order = np.argsort(b_sums, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # Precompute A_sym if B is symmetric for faster C construction
    if is_sym:
        A_sym = a + a.T
    else:
        A_sym = None

    # Helper function for delta cost of swapping i and j
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]

        # Term 1: Change in cost due to rows i and j of A interacting with permuted B
        row_diff_a = a[i, :] - a[j, :]
        row_diff_b = b[v, p] - b[u, p]
        term1 = np.dot(row_diff_a, row_diff_b)

        # Term 2: Change in cost due to columns i and j of A interacting with permuted B
        col_diff_a = a[:, i] - a[:, j]
        col_diff_b = b[:, v][p] - b[:, u][p]
        term2 = np.dot(col_diff_a, col_diff_b)

        return term1 + term2

    # Banded windowed realignment: exploit local sequential structure
    # Treat high-flow facility clusters as needing alignment to low-distance location windows
    def banded_realignment(p, current_cost, window_size=16, num_windows=None):
        """
        For each window of consecutive facilities (in the current assignment order),
        try to reassign them to the best matching window of locations using a
        banded DP approach. This exploits the idea that nearby facilities in the
        flow matrix should be placed at nearby locations in the distance matrix.
        """
        if num_windows is None:
            num_windows = n // window_size + 1
        
        p_work = p.copy()
        cost_work = current_cost
        
        # Identify high-flow facility groups: facilities with high row-sum in A
        a_row_sums = a.sum(axis=1)
        # Sort facilities by flow strength
        facility_order = np.argsort(-a_row_sums, kind='stable')
        
        # For each window of high-flow facilities, try to align to a low-distance location window
        for w in range(num_windows):
            if time.time() > deadline:
                break
            
            # Get a window of facilities (in flow-strength order)
            fac_start = w * window_size
            fac_end = min(fac_start + window_size, n)
            if fac_start >= n:
                break
            
            fac_window = facility_order[fac_start:fac_end]
            w_size = len(fac_window)
            
            # Find the best location window: locations with low average distance to each other
            # Use the current assignment as a starting point for the location window
            # The locations currently assigned to this facility window
            loc_window = p_work[fac_window]
            
            # Try to improve: find a better set of w_size locations for this facility window
            # Use a greedy approach: for each facility in the window, try to find the best
            # unassigned location within a band around its current assignment
            
            # Build a cost matrix for this sub-problem
            # For facilities in fac_window, consider locations within a band
            band_width = max(window_size, 8)
            
            # Get candidate locations: within band of each facility's current location
            candidate_locs = set()
            for fac in fac_window:
                cur_loc = p_work[fac]
                for dl in range(-band_width, band_width + 1):
                    loc = (cur_loc + dl) % n
                    candidate_locs.add(loc)
            
            candidate_locs = list(candidate_locs)
            if len(candidate_locs) < w_size:
                # Add more locations if needed
                all_locs = list(range(n))
                for loc in all_locs:
                    if loc not in candidate_locs:
                        candidate_locs.append(loc)
                    if len(candidate_locs) >= w_size:
                        break
            
            # Build cost matrix: cost of assigning facility fac to location loc
            # considering interactions within the window
            cost_mat = np.zeros((w_size, len(candidate_locs)))
            for ii, fac in enumerate(fac_window):
                for jj, loc in enumerate(candidate_locs):
                    # Cost contribution: flow from fac to other facilities in window * distance from loc to their locations
                    # Simplified: just use the direct cost a[fac, fac] * b[loc, loc] + interactions
                    # For efficiency, use a surrogate: row sum of A for fac * distance from loc to centroid of window
                    cost_mat[ii, jj] = a_row_sums[fac] * b[loc, loc]
            
            # Solve the small assignment problem
            if w_size <= len(candidate_locs):
                # Pad if needed
                if w_size < len(candidate_locs):
                    # Use only the first w_size candidate locations for simplicity
                    cost_mat_sub = cost_mat[:, :w_size]
                    row_ind_sub, col_ind_sub = linear_sum_assignment(cost_mat_sub)
                    # Apply the assignment
                    for ii, jj in zip(row_ind_sub, col_ind_sub):
                        fac = fac_window[ii]
                        new_loc = candidate_locs[jj]
                        # Check if this location is already assigned to another facility in the window
                        # For simplicity, just do the swap if it improves
                        old_loc = p_work[fac]
                        if old_loc != new_loc:
                            # Find who currently has new_loc
                            who_has = np.where(p_work == new_loc)[0]
                            if len(who_has) > 0:
                                other_fac = who_has[0]
                                if other_fac not in fac_window:
                                    # Swap
                                    p_work[fac] = new_loc
                                    p_work[other_fac] = old_loc
                                    # Recalculate cost for this local change
                                    # For efficiency, just track and verify at the end
                            else:
                                p_work[fac] = new_loc
            
            # Verify the cost after all window realignments
            new_cost = calc_cost(p_work)
            if new_cost < cost_work:
                cost_work = new_cost
            else:
                # Revert if worse
                p_work = p.copy()
        
        return p_work, cost_work

    # CTQA-inspired quantum perturbation
    def quantum_perturb(p, current_cost, temperature=1.0, num_steps=50):
        """
        Continuous-Time Quantum Annealing inspired perturbation.
        Uses a transverse field to allow tunneling through energy barriers.
        """
        p_work = p.copy()
        cost_work = current_cost
        
        # Quantum annealing schedule: start with strong transverse field, decay to zero
        for step in range(num_steps):
            if time.time() > deadline:
                break
                
            s = step / max(num_steps - 1, 1)
            gamma = (1.0 - s) * 10.0
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = get_delta(i, j, p_work)
            
            if delta < 0:
                p_work[i], p_work[j] = p_work[j], p_work[i]
                cost_work += delta
            else:
                effective_temp = temperature * (1.0 + gamma)
                if np.random.rand() < np.exp(-delta / max(effective_temp, 1e-10)):
                    p_work[i], p_work[j] = p_work[j], p_work[i]
                    cost_work += delta
        
        return p_work, cost_work

    # Phase 1: Frank-Wolfe on the Birkhoff Polytope (Continuous Relaxation)
    fw_deadline = start_time + 40.0

    while time.time() < fw_deadline:
        # Construct linear cost matrix C for the Frank-Wolfe step
        B_p = b[p]  # Rows of B permuted by p. B_p[i, k] = B[p[i], k]

        if is_sym:
            C = A_sym @ B_p
        else:
            C = a @ B_p.T + a.T @ B_p

        # Solve linear assignment to get the next vertex of the Birkhoff polytope
        row_ind, col_ind = linear_sum_assignment(C)

        # New permutation
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind

        # Evaluate new cost
        new_cost = calc_cost(p_new)

        if new_cost < current_cost:
            current_cost = new_cost
            p = p_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # No improvement from the global linearization step.
            # Try banded realignment first, then quantum perturbation
            improved = False
            
            # Try banded realignment
            p_br, cost_br = banded_realignment(p, current_cost, window_size=16, num_windows=8)
            if cost_br < current_cost:
                p = p_br
                current_cost = cost_br
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
            
            if not improved:
                # Use quantum perturbation to escape local minimum
                for _ in range(5):
                    p_q, cost_q = quantum_perturb(p, current_cost, temperature=1.0, num_steps=30)
                    if cost_q < current_cost:
                        p = p_q
                        current_cost = cost_q
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
                        improved = True
                        break
            if not improved:
                # If even quantum perturbation doesn't help, break to local search
                break

    # Phase 2: 2-Swap Local Search with O(n) delta evaluation
    p = best_p.copy()
    current_cost = best_cost

    no_improve_count = 0
    max_no_improve = 2000
    quantum_counter = 0
    br_counter = 0

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Alternate between quantum perturbation and banded realignment
            if br_counter % 2 == 0:
                # Use quantum perturbation to escape local minimum
                quantum_counter += 1
                temp = 1.0 + 0.1 * min(quantum_counter, 10)
                p_q, cost_q = quantum_perturb(p, current_cost, temperature=temp, num_steps=40)
                
                if cost_q < current_cost:
                    p = p_q
                    current_cost = cost_q
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    # If quantum perturbation didn't help, try a random kick
                    i, j = np.random.choice(n, 2, replace=False)
                    delta = get_delta(i, j, p)
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
            else:
                # Try banded realignment
                br_counter += 1
                p_br, cost_br = banded_realignment(p, current_cost, window_size=12, num_windows=10)
                if cost_br < current_cost:
                    p = p_br
                    current_cost = cost_br
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    # Fallback to random kick
                    i, j = np.random.choice(n, 2, replace=False)
                    delta = get_delta(i, j, p)
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
            
            no_improve_count = 0
            continue

        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue

        delta = get_delta(i, j, p)

        if delta < 0:
            # Accept improvement
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve_count = 0

            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve_count += 1

    # Final polish: run a few more banded realignment passes on the best solution
    if time.time() < deadline - 5.0:
        p_final = best_p.copy()
        cost_final = best_cost
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            p_br, cost_br = banded_realignment(p_final, cost_final, window_size=8, num_windows=12)
            if cost_br < cost_final:
                p_final = p_br
                cost_final = cost_br
        
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
