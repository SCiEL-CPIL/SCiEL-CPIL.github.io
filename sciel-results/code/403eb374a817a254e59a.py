import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute forbidden masks
    # forbid[i][j][r_j] is a bitmask of residues r_i in [0, m_i) that are forbidden
    # given that modulus j has residue r_j.
    # forbid[i][j] is a list of length m_j.
    forbid = [[None] * n for _ in range(n)]
    for i in range(n):
        m_i = moduli[i]
        for j in range(n):
            if i == j:
                continue
            m_j = moduli[j]
            g = gcd_tab[i][j]
            forbid[i][j] = [0] * m_j
            for r_j in range(m_j):
                mask = 0
                # r_i is forbidden if (r_i - r_j) % g == 0
                # r_i % g == r_j % g
                rem = r_j % g
                for r_i in range(rem, m_i, g):
                    mask |= (1 << r_i)
                forbid[i][j][r_j] = mask

    def try_greedy_mrv(randomize_residue=False):
        selected = [] # List of (idx, r)
        unselected = set(range(n))
        
        # current_forbidden[i] is the accumulated bitmask of forbidden residues for i
        # due to all selected moduli.
        current_forbidden = [0] * n
        
        while unselected:
            best_idx = -1
            best_valid_mask = 0
            min_count = float('inf')
            
            for idx in unselected:
                m_i = moduli[idx]
                # Full mask for m_i bits
                full_mask = (1 << m_i) - 1
                valid_mask = full_mask & ~current_forbidden[idx]
                count = bin(valid_mask).count('1')
                
                if count == 0:
                    unselected.discard(idx)
                    continue
                
                if count < min_count:
                    min_count = count
                    best_idx = idx
                    best_valid_mask = valid_mask
            
            if best_idx == -1:
                break
            
            # Pick a residue from best_valid_mask
            if randomize_residue:
                # Pick a random set bit
                # Generate random number in range of bits? 
                # Or list of bits and choose?
                # Since m_i <= 128, list of bits is fast.
                bits = [r for r in range(moduli[best_idx]) if (best_valid_mask >> r) & 1]
                r = random.choice(bits)
            else:
                # Pick lowest set bit
                # (mask & -mask).bit_length() - 1
                r = (best_valid_mask & -best_valid_mask).bit_length() - 1
            
            selected.append((best_idx, r))
            unselected.discard(best_idx)
            
            # Update current_forbidden for all unselected
            for j in unselected:
                current_forbidden[j] |= forbid[j][best_idx][r]

        return selected, unselected

    best = []
    best_unselected = set(range(n))
    best_len = 0

    # 1. Deterministic MRV Greedy
    res, unres = try_greedy_mrv(randomize_residue=False)
    if len(res) > best_len:
        best = res
        best_unselected = unres
        best_len = len(res)

    # 2. Randomized MRV Greedy Restarts
    random.seed(42)
    n_restarts = 5000
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        res, unres = try_greedy_mrv(randomize_residue=True)
        if len(res) > best_len:
            best = res
            best_unselected = unres
            best_len = len(res)

    # 3. Local Search: Try to add remaining unselected
    # Re-use the bitmask logic for local search?
    # For simplicity, use the slow check for local search as it's few steps.
    
    def check_residue_slow(idx, r, selected_list):
        for (idx2, r2) in selected_list:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_unselected = list(best_unselected)
            for idx in current_unselected:
                if idx not in best_unselected:
                    continue
                m = moduli[idx]
                for r in range(m):
                    if check_residue_slow(idx, r, best):
                        best.append((idx, r))
                        best_unselected.discard(idx)
                        improved = True
                        break
                if improved:
                    break

    # 4. Local Search: Swap to add
    if time.time() < deadline - 5 and best_unselected:
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            for new_idx in list(best_unselected):
                if new_idx not in best_unselected:
                    continue
                m_new = moduli[new_idx]
                
                found_swap = False
                for pos in range(len(best)):
                    old_idx, old_r = best[pos]
                    old_m = moduli[old_idx]
                    
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
                        
                        ok_old = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        if found_new:
                            best[pos] = (old_idx, new_r)
                            best.append((new_idx, r_new))
                            best_unselected.discard(new_idx)
                            swap_improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

    return [[r, moduli[idx]] for (idx, r) in best]
