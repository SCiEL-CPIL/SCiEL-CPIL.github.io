import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 58.0

    # Precompute all APs (length >= 2) and build dense membership matrix
    ap_list = []
    for d in range(1, N + 1):
        for start in range(0, N - d + 1):
            terms = list(range(start, N, d))
            if len(terms) >= 2:
                ap_list.append(terms)
    
    num_aps = len(ap_list)
    
    # Dense membership matrix: M[ap_idx, pos] = 1 if pos in AP ap_idx
    M = np.zeros((num_aps, N), dtype=np.int8)
    for ap_idx, terms in enumerate(ap_list):
        for pos in terms:
            M[ap_idx, pos] = 1

    # Column sums: how many APs contain each position (for greedy init)
    col_sums = M.sum(axis=0)

    def compute_sums(signs):
        return M @ signs.astype(np.int32)

    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))

    def greedy_init():
        signs = np.zeros(N, dtype=np.int8)
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        
        for i in range(N):
            # Try both signs, pick the one that minimizes max abs sum
            best_sign = 1
            best_max = N + 1
            best_sums = None
            
            for s in [1, -1]:
                # Update sums for APs containing position i
                affected = M[:, i]
                if np.any(affected):
                    new_sums = ap_sums + s * affected
                    m = get_max_abs(new_sums)
                    if m < best_max:
                        best_max = m
                        best_sign = s
                        best_sums = new_sums
                else:
                    m = get_max_abs(ap_sums)
                    if m < best_max:
                        best_max = m
                        best_sign = s
                        best_sums = ap_sums.copy()
            
            signs[i] = best_sign
            if best_sums is not None:
                ap_sums = best_sums
        
        return signs, ap_sums

    def try_2flip(signs, ap_sums, M_sub, num_aps_sub, current_max):
        """Try a 2-flip move on a subset of APs. Returns (accepted, new_max)."""
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            return False, current_max
        
        di = -2 * signs[i]
        dj = -2 * signs[j]
        
        # Compute delta for the subset
        delta = di * M_sub[:, i] + dj * M_sub[:, j]
        
        # Check if any AP is affected
        if not np.any(delta):
            return False, current_max
        
        new_sums = ap_sums + delta
        new_max = get_max_abs(new_sums)
        
        if new_max < current_max:
            # Accept
            signs[i] = -signs[i]
            signs[j] = -signs[j]
            return True, new_max
        elif new_max == current_max and random.random() < 0.05:
            # Random acceptance to escape plateaus
            signs[i] = -signs[i]
            signs[j] = -signs[j]
            return True, new_max
        else:
            return False, current_max

    def local_search(signs, ap_sums, M_sub, num_aps_sub, current_max, time_limit):
        while time.time() < time_limit:
            accepted, current_max = try_2flip(signs, ap_sums, M_sub, num_aps_sub, current_max)
            if accepted:
                # Update ap_sums if we're using the full matrix
                if num_aps_sub == num_aps:
                    pass  # ap_sums is already updated via signs change
                # For subset, we don't maintain ap_sums; recompute at end if needed
        return signs, current_max

    # Phase 1: Greedy init
    signs, ap_sums = greedy_init()
    current_max = get_max_abs(ap_sums)
    best_val = current_max
    best_signs = signs.copy()

    # Identify critical APs (those with |sum| >= threshold)
    # Use a subset of APs for fast local search
    threshold = max(1, current_max - 1)
    critical_mask = np.abs(ap_sums) >= threshold
    num_critical = int(np.sum(critical_mask))
    
    if num_critical < 10:
        critical_mask = np.abs(ap_sums) >= max(1, current_max - 3)
        num_critical = int(np.sum(critical_mask))
    
    if num_critical < 10:
        critical_mask = np.ones(num_aps, dtype=bool)
        num_critical = num_aps
    
    M_critical = M[critical_mask, :]
    
    # Phase 2: Local search on critical APs
    # We need to track sums for the critical subset
    crit_sums = ap_sums[critical_mask]
    crit_max = get_max_abs(crit_sums)
    
    time_left_phase2 = min(30.0, deadline - time.time() - 5.0)
    if time_left_phase2 > 1.0:
        signs, crit_max = local_search(signs, crit_sums, M_critical, num_critical, crit_max, time_left_phase2)
    
    # Recompute full sums after critical-AP local search
    ap_sums = compute_sums(signs)
    current_max = get_max_abs(ap_sums)
    
    if current_max < best_val:
        best_val = current_max
        best_signs = signs.copy()

    # Phase 3: Local search on ALL APs (polish)
    time_left_phase3 = deadline - time.time() - 2.0
    if time_left_phase3 > 1.0 and best_val > 2:
        signs, current_max = local_search(signs, ap_sums, M, num_aps, current_max, time_left_phase3)
    
    if current_max < best_val:
        best_val = current_max
        best_signs = signs.copy()

    # Final verification
    final_sums = compute_sums(best_signs)
    final_max = get_max_abs(final_sums)
    
    return best_signs.tolist()
