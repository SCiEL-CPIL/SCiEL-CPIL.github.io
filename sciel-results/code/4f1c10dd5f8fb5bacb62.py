import time
import math
import random

def construct():
    start_time = time.time()
    # Safety margin: 90s limit. Use 85s to be safe.
    deadline = start_time + 85.0

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

    # Precompute non-coprime counts for heuristic ordering
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
                # Residue 1 is often good.
                residues = [1, 0] + [r for r in range(2, m)]
            else:
                residues = residue_order(m)
            
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
    orderings = [
        sorted(range(n), key=lambda i: -non_coprime_count[i]),
        sorted(range(n), key=lambda i: non_coprime_count[i]),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        sorted(range(n), key=lambda i: (0 if non_coprime_count[i] == 0 else 1, moduli[i])),
    ]

    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random Restarts
    random.seed(12345)
    restarts = 5000
    for it in range(restarts):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
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

    # Local Search 2: 1-Swap (Change one residue) and 2-Swap (Change two residues)
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            
            # Try to add each missing modulus by modifying existing residues
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                
                # 1. Try 1-Swap: Change one existing residue to allow new_idx
                for pos in range(len(best)):
                    old_idx, old_r, old_m = best[pos]
                    
                    # Find valid residues for old_idx that are compatible with all OTHERS
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

                    for new_r_old in valid_old_residues:
                        # Check if new_idx can be added
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
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
                            best[pos] = (old_idx, new_r_old, old_m)
                            best.append((new_idx, r_new, m_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

                # 2. Try 2-Swap: Change two existing residues to allow new_idx
                if not swap_improved and len(best) >= 2:
                    for pos1 in range(len(best)):
                        if swap_improved: break
                        old_idx1, old_r1, old_m1 = best[pos1]
                        for pos2 in range(pos1 + 1, len(best)):
                            if swap_improved: break
                            old_idx2, old_r2, old_m2 = best[pos2]
                            
                            # We need to find r1', r2' such that:
                            # 1. r1' is compatible with all except old_idx2 (and new_idx)
                            # 2. r2' is compatible with all except old_idx1 (and new_idx)
                            # 3. r1' and r2' are compatible with each other
                            # 4. new_idx is compatible with r1', r2', and all others

                            # This is expensive. We limit the search space for 2-swaps.
                            # Only try a few random valid candidates for r1' and r2'
                            candidates1 = []
                            candidates2 = []
                            
                            # Generate a few random valid residues for pos1
                            for _ in range(10):
                                r1_try = random.randint(0, old_m1 - 1)
                                ok = True
                                for qpos in range(len(best)):
                                    if qpos == pos1 or qpos == pos2:
                                        continue
                                    other_idx, other_r, other_m = best[qpos]
                                    g = gcd_tab[old_idx1][other_idx]
                                    if (r1_try - other_r) % g == 0:
                                        ok = False
                                        break
                                if ok:
                                    candidates1.append(r1_try)
                            
                            if not candidates1:
                                continue

                            for _ in range(10):
                                r2_try = random.randint(0, old_m2 - 1)
                                ok = True
                                for qpos in range(len(best)):
                                    if qpos == pos1 or qpos == pos2:
                                        continue
                                    other_idx, other_r, other_m = best[qpos]
                                    g = gcd_tab[old_idx2][other_idx]
                                    if (r2_try - other_r) % g == 0:
                                        ok = False
                                        break
                                if ok:
                                    candidates2.append(r2_try)
                            
                            if not candidates2:
                                continue

                            for r1_new in candidates1:
                                if swap_improved: break
                                for r2_new in candidates2:
                                    # Check mutual compatibility
                                    if (r1_new - r2_new) % gcd_tab[old_idx1][old_idx2] == 0:
                                        continue
                                    
                                    # Check if new_idx can be added
                                    found_new = False
                                    for r_new in range(m_new):
                                        ok_new = True
                                        for qpos in range(len(best)):
                                            if qpos == pos1:
                                                if (r_new - r1_new) % gcd_tab[new_idx][old_idx1] == 0:
                                                    ok_new = False
                                                    break
                                            elif qpos == pos2:
                                                if (r_new - r2_new) % gcd_tab[new_idx][old_idx2] == 0:
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
                                        best[pos1] = (old_idx1, r1_new, old_m1)
                                        best[pos2] = (old_idx2, r2_new, old_m2)
                                        best.append((new_idx, r_new, m_new))
                                        selected_idx.add(new_idx)
                                        swap_improved = True
                                        break
                            if swap_improved:
                                break

    return [[r, m] for (_, r, m) in best]
