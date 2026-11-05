import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    start_time = time.time()
    DEADLINE = start_time + 82.0  # Safety margin before 90s kill
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict bitmasks for O(1) conflict checking
    # conflict_mask[i] has bit j set if i conflicts with j
    conflict_mask = [0] * n
    conflicts = []
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflicts.append((i, j))
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Adjacency sets for greedy
    adj = [set() for _ in range(n)]
    for i, j in conflicts:
        adj[i].add(j)
        adj[j].add(i)
    
    # Precompute weights
    weights = [1.0 / nums[i] for i in range(n)]
    
    # Omega function: count total prime factors with multiplicity
    def omega_count(x):
        cnt = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                cnt += 1
                x //= d
            d += 1
        if x > 1:
            cnt += 1
        return cnt
    
    # Partition by Omega parity
    omega_parity = [omega_count(nums[i]) % 2 for i in range(n)]
    even_parity = [i for i in range(n) if omega_parity[i] == 0]
    odd_parity = [i for i in range(n) if omega_parity[i] == 1]
    
    # Helper: check if bitmask is valid (independent set)
    def is_valid_bitmask(mask):
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            if (conflict_mask[i] & (mask ^ lsb)) != 0:
                return False
            m ^= lsb
        return True
    
    # Helper: compute weight sum of bitmask
    def weight_sum(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            total += weights[i]
            m ^= lsb
        return total
    
    best_mask = 0
    best_val = 0.0
    
    # --- MIP solver with time limit ---
    c = np.array([-weights[i] for i in range(n)])
    m_conf = len(conflicts)
    
    if m_conf > 0:
        # Use sparse matrix for efficiency
        row_idx = []
        col_idx = []
        data = []
        for k, (i, j) in enumerate(conflicts):
            row_idx.extend([k, k])
            col_idx.extend([i, j])
            data.extend([1.0, 1.0])
        A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(m_conf, n))
        constraints = LinearConstraint(A, lb=-np.inf, ub=np.ones(m_conf))
    else:
        constraints = LinearConstraint(np.zeros((1, n)), lb=0, ub=1)
    
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    integrality = np.ones(n)
    
    mip_time_limit = 30.0  # Give MIP 30 seconds
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality,
                      bounds=bounds, time_limit=mip_time_limit)
        if result is not None and result.x is not None:
            mask = 0
            for i in range(n):
                if result.x[i] > 0.5:
                    mask |= (1 << i)
            if is_valid_bitmask(mask):
                val = weight_sum(mask)
                if val > best_val:
                    best_val = val
                    best_mask = mask
    except Exception:
        pass
    
    # --- Greedy heuristics ---
    # Greedy 1: by weight descending
    order_w = sorted(range(n), key=lambda i: -weights[i])
    mask = 0
    for i in order_w:
        if (conflict_mask[i] & mask) == 0:
            mask |= (1 << i)
    if is_valid_bitmask(mask):
        val = weight_sum(mask)
        if val > best_val:
            best_val = val
            best_mask = mask
    
    # Greedy 2: by weight/degree ratio (prefer low-conflict high-weight)
    degrees = [len(adj[i]) for i in range(n)]
    order_ratio = sorted(range(n), key=lambda i: -weights[i] / max(1, degrees[i]))
    mask = 0
    for i in order_ratio:
        if (conflict_mask[i] & mask) == 0:
            mask |= (1 << i)
    if is_valid_bitmask(mask):
        val = weight_sum(mask)
        if val > best_val:
            best_val = val
            best_mask = mask
    
    # Greedy 3: by weight ascending (smallest numbers first, highest reciprocals)
    order_asc = sorted(range(n), key=lambda i: nums[i])
    mask = 0
    for i in order_asc:
        if (conflict_mask[i] & mask) == 0:
            mask |= (1 << i)
    if is_valid_bitmask(mask):
        val = weight_sum(mask)
        if val > best_val:
            best_val = val
            best_mask = mask
    
    # --- Omega-parity partitioned local search ---
    # Solve each parity class as a subproblem, then combine
    def local_search_partition(indices, initial_mask, max_time):
        """Local search restricted to a partition."""
        nonlocal best_mask, best_val
        current_mask = initial_mask
        current_val = weight_sum(current_mask)
        improved = True
        max_iters = 5000
        
        while improved and max_iters > 0 and time.time() < max_time:
            improved = False
            max_iters -= 1
            
            # Try adding each non-selected element in partition
            for i in indices:
                if (current_mask >> i) & 1:
                    continue
                if (conflict_mask[i] & current_mask) == 0:
                    new_mask = current_mask | (1 << i)
                    new_val = current_val + weights[i]
                    if new_val > current_val + 1e-12:
                        current_mask = new_mask
                        current_val = new_val
                        improved = True
                        if current_val > best_val:
                            best_val = current_val
                            best_mask = current_mask
            
            # Try 1-swap: remove one, add another
            for remove_i in indices:
                if not ((current_mask >> remove_i) & 1):
                    continue
                base_mask = current_mask ^ (1 << remove_i)
                base_val = current_val - weights[remove_i]
                for add_i in indices:
                    if (base_mask >> add_i) & 1:
                        continue
                    if (conflict_mask[add_i] & base_mask) == 0:
                        new_val = base_val + weights[add_i]
                        if new_val > current_val + 1e-12:
                            current_mask = base_mask | (1 << add_i)
                            current_val = new_val
                            improved = True
                            if current_val > best_val:
                                best_val = current_val
                                best_mask = current_mask
                            break
                if improved:
                    break
        
        return current_mask, current_val
    
    # Run local search on full set first
    if time.time() < DEADLINE - 10:
        remaining = DEADLINE - 10 - time.time()
        if remaining > 5:
            best_mask, best_val = local_search_partition(
                list(range(n)), best_mask, time.time() + remaining)
    
    # Run local search on each parity partition
    for partition, part_name in [(even_parity, "even"), (odd_parity, "odd")]:
        if time.time() >= DEADLINE - 8:
            break
        remaining = DEADLINE - 8 - time.time()
        if remaining < 3:
            break
        
        # Start from best_mask restricted to this partition
        part_mask = 0
        for i in partition:
            if (best_mask >> i) & 1:
                part_mask |= (1 << i)
        
        # Also try greedy start within partition
        part_order = sorted(partition, key=lambda i: -weights[i])
        greedy_part_mask = 0
        for i in part_order:
            if (conflict_mask[i] & greedy_part_mask) == 0:
                greedy_part_mask |= (1 << i)
        
        # Pick better starting point
        start_mask = part_mask if weight_sum(part_mask) >= weight_sum(greedy_part_mask) else greedy_part_mask
        
        part_mask_out, part_val = local_search_partition(partition, start_mask, time.time() + remaining)
        
        # Combine with best from other partition
        other_partition = odd_parity if part_name == "even" else even_parity
        other_mask = 0
        for i in other_partition:
            if (best_mask >> i) & 1:
                other_mask |= (1 << i)
        
        combined = part_mask_out | other_mask
        if is_valid_bitmask(combined):
            combined_val = weight_sum(combined)
            if combined_val > best_val:
                best_val = combined_val
                best_mask = combined
    
    # --- Final local search on full set ---
    if time.time() < DEADLINE - 3:
        remaining = DEADLINE - 3 - time.time()
        if remaining > 3:
            best_mask, best_val = local_search_partition(
                list(range(n)), best_mask, time.time() + remaining)
    
    # --- Extract result ---
    result_list = []
    m = best_mask
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        result_list.append(nums[i])
        m ^= lsb
    
    if not result_list:
        result_list = [2]
    
    # Final validation
    for idx_a in range(len(result_list)):
        for idx_b in range(idx_a + 1, len(result_list)):
            a, b = result_list[idx_a], result_list[idx_b]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                # Invalid - fall back to simple greedy
                mask = 0
                for i in order_asc:
                    if (conflict_mask[i] & mask) == 0:
                        mask |= (1 << i)
                result_list = []
                m = mask
                while m:
                    lsb = m & (-m)
                    i = lsb.bit_length() - 1
                    result_list.append(nums[i])
                    m ^= lsb
                if not result_list:
                    result_list = [2]
                break
    
    return result_list
