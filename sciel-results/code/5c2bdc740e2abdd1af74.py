import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute number of non-coprime neighbors for each modulus
    # Higher count means more constraints (harder to place)
    non_coprime_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_tab[i][j] > 1:
                non_coprime_count[i] += 1

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2, m2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_greedy(order, residue_order=None):
        selected = []
        for idx in order:
            m = moduli[idx]
            if residue_order is None:
                # Heuristic: Try residues that are 'hard' to conflict with?
                # Residue 1 is often good. Residue 0 is bad if many even moduli.
                # Let's try 1, then 0, then others.
                residues = [1, 0] + [r for r in range(2, m)]
            else:
                residues = residue_order(m)
            
            # Optimization: If m is large, shuffling might be expensive.
            # But we only do this for random restarts.
            
            found = False
            for r in residues:
                if check_residue(idx, r, selected):
                    selected.append((idx, r, m))
                    found = True
                    break
        return selected

    def residue_order_random(m):
        res = list(range(m))
        random.shuffle(res)
        return res

    best = []
    best_len = 0

    # Deterministic Orderings
    # 1. Most constrained first (highest non_coprime_count)
    orderings = [
        sorted(range(n), key=lambda i: -non_coprime_count[i]),
        sorted(range(n), key=lambda i: non_coprime_count[i]),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        # Primes first (easiest)
        sorted(range(n), key=lambda i: (0 if non_coprime_count[i] == 0 else 1, moduli[i])),
    ]

    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random Restarts
    random.seed(12345)
    # Increase restarts if we are fast enough
    restarts = 5000
    for it in range(restarts):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        # Use random residue order to escape local optima
        res = try_greedy(order, residue_order=residue_order_random)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Local Search 1: Add any missing modulus
    if time.time() < deadline - 15:
        selected_idx = set(idx for (idx, _, _) in best)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                # Try residues in a smart order: 1, 0, then random
                residues_to_try = [1, 0] + list(range(2, m))
                random.shuffle(residues_to_try[2:])
                for r in residues_to_try:
                    if check_residue(idx, r, best):
                        best.append((idx, r, m))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break

    # Local Search 2: Swap residues to enable adding new moduli
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            # Try to add each missing modulus
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                
                # Try changing one existing residue to free up space
                for pos in range(len(best)):
                    old_idx, old_r, old_m = best[pos]
                    
                    # Find all valid residues for old_idx that are compatible with all OTHERS
                    # This allows us to swap old_r to new_r_old
                    valid_old_residues = []
                    for new_r_old in range(old_m):
                        if new_r_old == old_r:
                            continue
                        ok = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r, other_m = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r_old - other_r) % g == 0:
                                ok = False
                                break
                        if ok:
                            valid_old_residues.append(new_r_old)
                    
                    if not valid_old_residues:
                        continue

                    # For each valid new residue for old_idx, see if we can add new_idx
                    for new_r_old in valid_old_residues:
                        # Check if new_idx can be added with new_r_old in place of old_r
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    # Compare with new_r_old
                                    if (r_new - new_r_old) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r, other_m = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        
                        if found_new:
                            # Perform the swap
                            best[pos] = (old_idx, new_r_old, old_m)
                            best.append((new_idx, r_new, m_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

    return [[r, m] for (_, r, m) in best]
