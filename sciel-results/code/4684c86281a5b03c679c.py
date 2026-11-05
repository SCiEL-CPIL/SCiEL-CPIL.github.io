"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)  # 127

    # Precompute gcd for all pairs of moduli
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute blocked masks
    # blocked[i][j] is a list of length moduli[i].
    # blocked[i][j][r_i] is a bitmask of residues r_j in moduli[j] that conflict with r_i.
    blocked = [[None] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        mi = moduli[i]
        for j in range(n_mod):
            if i == j:
                continue
            mj = moduli[j]
            g = gcd_mat[i][j]
            masks = [0] * mi
            for r_i in range(mi):
                target = r_i % g
                mask = 0
                for r_j in range(mj):
                    if r_j % g == target:
                        mask |= (1 << r_j)
                masks[r_i] = mask
            blocked[i][j] = masks

    def is_valid_solution(sol):
        """Check if the solution is strictly valid."""
        n = len(sol)
        for i in range(n):
            r_i, m_i = sol[i]
            idx_i = moduli.index(m_i)
            for j in range(i + 1, n):
                r_j, m_j = sol[j]
                idx_j = moduli.index(m_j)
                g = gcd_mat[idx_i][idx_j]
                if r_i % g == r_j % g:
                    return False
        return True

    def greedy_init(order):
        avail = [(1 << moduli[i]) - 1 for i in range(n_mod)]
        chosen = []
        for idx in order:
            if avail[idx] == 0:
                continue
            # Pick the residue that blocks the fewest future residues
            best_r = -1
            best_cost = float('inf')
            mi = moduli[idx]
            for r in range(mi):
                if not (avail[idx] & (1 << r)):
                    continue
                # Count how many residues in future moduli would be blocked
                cost = 0
                # Only consider moduli that appear later in the 'order'
                # To optimize, we can precompute the set of indices that are 'future'
                # But for simplicity, we check if the index is in the remaining part of order.
                # A faster way: iterate through the rest of the order list.
                # However, 'order' is a permutation. We need to know which indices are 'after' idx in the order.
                # Let's create a position map.
                pass # We will optimize this below
                
            # Optimization: Precompute position in order
            # This is done outside or we pass it.
            # For now, simple loop over all j in order that come after idx.
            
            # Re-implementing the cost calculation correctly:
            # We need to know which moduli are 'future' relative to idx in the given order.
            # Let's build a set of indices that are yet to be processed.
            # Actually, the original code iterated `range(idx + 1, n_mod)` which assumed order was 0..n-1.
            # With arbitrary order, we must check if `j` is in the suffix of `order` starting after `idx`.
            
            # To make this fast, let's precompute a `pos` array for the current order.
            # But `greedy_init` is called many times.
            # Let's modify greedy_init to take `order` and use a helper.
            
            # For this specific edit, I will rewrite the inner loop to be correct for arbitrary order.
            # We need a set of indices that are "later" in the order.
            # Let's just iterate through the order list and find the index of idx, then process the rest.
            # This is O(N^2) per step, so O(N^3) total. N=127. 127^3 ~ 2M. This is fine.
            
            # Find position of idx in order
            pos_idx = order.index(idx)
            future_indices = set(order[pos_idx + 1:])
            
            for r in range(mi):
                if not (avail[idx] & (1 << r)):
                    continue
                cost = 0
                for j in future_indices:
                    if avail[j] == 0:
                        continue
                    mask = blocked[idx][j][r]
                    cost += bin(avail[j] & mask).count('1')
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
            
            if best_r >= 0:
                chosen.append((best_r, moduli[idx]))
                # Update available masks
                for j in future_indices:
                    mask = blocked[idx][j][best_r]
                    avail[j] &= ~mask
        return chosen

    best_sol = []
    best_len = 0

    # Try several deterministic orderings
    orderings = [
        list(range(n_mod)),
        list(range(n_mod - 1, -1, -1)),
        sorted(range(n_mod), key=lambda i: moduli[i]),
        sorted(range(n_mod), key=lambda i: -moduli[i]),
    ]
    
    # Add some heuristic orderings
    # Moduli with more factors are more constrained? Or less?
    # Let's try sorting by number of divisors.
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orderings.append(sorted(range(n_mod), key=lambda i: num_divisors(moduli[i])))
    orderings.append(sorted(range(n_mod), key=lambda i: -num_divisors(moduli[i])))

    for order in orderings:
        if time.time() > deadline - 5.0:
            break
        sol = greedy_init(order)
        # Validate
        if is_valid_solution(sol):
            if len(sol) > best_len:
                best_len = len(sol)
                best_sol = sol
        else:
            # If greedy failed validation (shouldn't happen with correct logic),
            # try to repair by removing conflicts
            # For now, just skip if invalid to be safe, or try a simpler repair.
            pass

    # Randomized restarts
    import random
    random.seed(42)
    trials = 0
    while time.time() < deadline - 10.0 and trials < 500:
        trials += 1
        order = list(range(n_mod))
        random.shuffle(order)
        sol = greedy_init(order)
        if is_valid_solution(sol):
            if len(sol) > best_len:
                best_len = len(sol)
                best_sol = sol

    # Final check: ensure the best solution is valid
    if not is_valid_solution(best_sol):
        # Fallback: return a trivially valid solution (e.g., just one class)
        # Or try to find a valid subset of best_sol
        valid_subset = []
        for pair in best_sol:
            # Check if pair can be added to valid_subset
            can_add = True
            for existing in valid_subset:
                r1, m1 = existing
                r2, m2 = pair
                idx1 = moduli.index(m1)
                idx2 = moduli.index(m2)
                g = gcd_mat[idx1][idx2]
                if r1 % g == r2 % g:
                    can_add = False
                    break
            if can_add:
                valid_subset.append(pair)
        best_sol = valid_subset

    return [[r, m] for r, m in best_sol]
# EVOLVE-BLOCK-END
