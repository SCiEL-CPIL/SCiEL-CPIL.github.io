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

    def check_residue(idx, r, selected):
        """Check if residue r for modulus idx is compatible with all selected."""
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_add(idx, selected):
        """Try to add idx to selected with a valid residue."""
        m = moduli[idx]
        # Shuffle residues for randomness
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if check_residue(idx, r, selected):
                selected.append((idx, r))
                return True
        return False

    def local_search(selected, time_limit):
        """Improve selected set."""
        selected_idx = set(idx for (idx, _) in selected)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add new moduli
            for idx in range(n):
                if idx in selected_idx:
                    continue
                if try_add(idx, selected):
                    selected_idx.add(idx)
                    improved = True
                    break
            
            if improved:
                continue

            # Try to swap residues to enable adding new moduli
            # Pick a random unselected idx
            unselected = [i for i in range(n) if i not in selected_idx]
            if not unselected:
                break
            new_idx = random.choice(unselected)
            m_new = moduli[new_idx]
            
            # Try to change one existing residue to free up space for new_idx
            pos = random.randint(0, len(selected) - 1)
            old_idx, old_r = selected[pos]
            old_m = moduli[old_idx]
            
            found_new = False
            # Try a few random residues for old_idx
            candidates = list(range(old_m))
            random.shuffle(candidates)
            for new_r in candidates:
                if new_r == old_r:
                    continue
                # Check if new_r is compatible with all other selected (excluding old_idx)
                ok_old = True
                for qpos in range(len(selected)):
                    if qpos == pos:
                        continue
                    other_idx, other_r = selected[qpos]
                    g = gcd_tab[old_idx][other_idx]
                    if (new_r - other_r) % g == 0:
                        ok_old = False
                        break
                if not ok_old:
                    continue
                
                # Now check if new_idx can be added with some residue
                # We need to check against all selected except pos, and against new_r at pos
                found_new_r = False
                residues_new = list(range(m_new))
                random.shuffle(residues_new)
                for r_new in residues_new:
                    ok_new = True
                    for qpos in range(len(selected)):
                        if qpos == pos:
                            if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                ok_new = False
                                break
                        else:
                            other_idx, other_r = selected[qpos]
                            if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                ok_new = False
                                break
                    if ok_new:
                        found_new_r = r_new
                        break
                
                if found_new_r is not False:
                    selected[pos] = (old_idx, new_r)
                    selected.append((new_idx, found_new_r))
                    selected_idx.add(new_idx)
                    improved = True
                    break

    best = []
    best_len = 0
    random.seed(42)
    
    # Run multiple restarts
    while time.time() < deadline - 5:
        selected = []
        # Random order initialization
        order = list(range(n))
        random.shuffle(order)
        for idx in order:
            if try_add(idx, selected):
                pass # Added
        
        if len(selected) > best_len:
            best = selected[:]
            best_len = len(selected)
            
        # Local search
        local_search(selected, deadline - 2)
        
        if len(selected) > best_len:
            best = selected[:]
            best_len = len(selected)
            
        # If we found a very good solution, we might stop early?
        # Max possible is 64. If we hit 64, we can stop.
        if best_len >= 64:
            break

    return [[r, moduli[idx]] for (idx, r) in best]
