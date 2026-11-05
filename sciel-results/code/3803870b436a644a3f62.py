import time
import math
import random
import sys

sys.setrecursionlimit(2000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute compatibility counts
    compat = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and gcd_tab[i][j] > 1:
                compat[i] += 1

    def is_compatible(idx, r, selected_pairs):
        """selected_pairs: list of (idx, r)"""
        for (idx2, r2) in selected_pairs:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def greedy_solve(order):
        sel = []
        for idx in order:
            m = moduli[idx]
            found = False
            for r in range(m):
                if is_compatible(idx, r, sel):
                    sel.append((idx, r))
                    found = True
                    break
            if not found:
                pass
        return sel

    # 1. Initialization
    best_sel = []
    
    orders = [
        sorted(range(n), key=lambda i: compat[i]),
        sorted(range(n), key=lambda i: -compat[i]),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        list(range(n)),
        list(reversed(range(n))),
    ]
    
    for order in orders:
        res = greedy_solve(order)
        if len(res) > len(best_sel):
            best_sel = res

    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 40:
            break
        order = list(range(n))
        random.shuffle(order)
        res = greedy_solve(order)
        if len(res) > len(best_sel):
            best_sel = res

    current_sel = list(best_sel)
    best_size = len(current_sel)

    # 2. Local Search Loop
    while time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Try to add
        curr_idx_set = set(x[0] for x in current_sel)
        missing = [i for i in range(n) if i not in curr_idx_set]
        
        for idx in missing:
            m = moduli[idx]
            for r in range(m):
                if is_compatible(idx, r, current_sel):
                    current_sel.append((idx, r))
                    best_size = len(current_sel)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue

        # Phase 2: Residue Swap to Add
        curr_idx_set = set(x[0] for x in current_sel)
        missing = [i for i in range(n) if i not in curr_idx_set]
        
        swap_found = False
        for new_idx in missing:
            if swap_found:
                break
            m_new = moduli[new_idx]
            for pos in range(len(current_sel)):
                if swap_found:
                    break
                old_idx, old_r = current_sel[pos]
                # Try changing old_r
                for new_r in range(moduli[old_idx]):
                    if new_r == old_r:
                        continue
                    # Check if new_r is valid for old_idx against others
                    ok_old = True
                    for qpos in range(len(current_sel)):
                        if qpos == pos:
                            continue
                        other_idx, other_r = current_sel[qpos]
                        g = gcd_tab[old_idx][other_idx]
                        if (new_r - other_r) % g == 0:
                            ok_old = False
                            break
                    if not ok_old:
                        continue
                    
                    # Check if new_idx can be added
                    found_new = False
                    for r_new in range(m_new):
                        ok_new = True
                        for qpos in range(len(current_sel)):
                            if qpos == pos:
                                if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                    ok_new = False
                                    break
                            else:
                                other_idx, other_r = current_sel[qpos]
                                if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                    ok_new = False
                                    break
                        if ok_new:
                            found_new = True
                            break
                    if found_new:
                        # Perform swap
                        current_sel[pos] = (old_idx, new_r)
                        current_sel.append((new_idx, r_new))
                        best_size = len(current_sel)
                        swap_found = True
                        break
        
        if swap_found:
            continue

        # Phase 3: Modulus Swap (1-for-1)
        if time.time() > deadline - 10:
            break
            
        curr_idx_set = set(x[0] for x in current_sel)
        missing = [i for i in range(n) if i not in curr_idx_set]
        
        if missing and current_sel:
            # Try a few random swaps
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                idx_remove = random.choice([x[0] for x in current_sel])
                idx_add = random.choice(missing)
                
                # Create new set of indices
                new_indices = [x[0] for x in current_sel if x[0] != idx_remove] + [idx_add]
                
                # Try to find a valid assignment for new_indices
                order = new_indices
                random.shuffle(order)
                res = greedy_solve(order)
                
                if len(res) == len(new_indices):
                    # Valid assignment found for the swapped set
                    current_sel = res
                    # Size is same, but configuration changed
                    break

    return [[r, moduli[idx]] for (idx, r) in best_sel]
