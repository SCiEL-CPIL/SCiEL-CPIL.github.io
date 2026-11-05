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
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g
    
    # Precompute for each pair (i,j): the set of residues r_i such that 
    # r_i is INCOMPATIBLE with residue r_j for modulus j
    # Actually, for a given pair (i,j) and fixed r_j, the incompatible r_i values
    # are those where (r_i - r_j) % gcd(i,j) == 0
    # So for each pair and each r_j, there are exactly gcd(i,j) incompatible residues out of m_i
    
    # Helper: check if residue r for modulus idx is compatible with all selected
    def check_residue(idx, r, selected):
        """selected is a list of (idx2, r2)"""
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True
    
    # Count conflicts: for a candidate (idx, r), how many selected does it conflict with?
    def conflict_count(idx, r, selected):
        cnt = 0
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                cnt += 1
        return cnt
    
    # Backtracking with MRV heuristic
    def backtrack(selected, remaining, depth=0):
        """Try to extend selected by picking from remaining moduli.
        Returns (success, new_selected)"""
        if time.time() > deadline - 5:
            return False, selected
        
        if not remaining:
            return True, selected
        
        # MRV: pick the modulus with fewest valid residues given current selected
        best_idx = -1
        best_count = float('inf')
        best_residues = None
        
        for idx in remaining:
            m = moduli[idx]
            valid = []
            for r in range(m):
                if check_residue(idx, r, selected):
                    valid.append(r)
            cnt = len(valid)
            if cnt < best_count:
                best_count = cnt
                best_idx = idx
                best_residues = valid
            if cnt == 0:
                break
        
        if best_count == 0 or best_idx == -1:
            return False, selected
        
        # Try residues for best_idx
        # Order: try residues that minimize future conflicts
        remaining2 = [idx for idx in remaining if idx != best_idx]
        
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            success, result = backtrack(new_selected, remaining2, depth + 1)
            if success:
                return True, result
        
        return False, selected
    
    # Greedy with given order
    def try_greedy(order):
        selected = []
        for idx in order:
            m = moduli[idx]
            # Try residues in order of least conflict
            best_r = -1
            best_conflicts = float('inf')
            for r in range(m):
                if check_residue(idx, r, selected):
                    c = conflict_count(idx, r, selected)  # should be 0 if compatible
                    if c < best_conflicts:
                        best_conflicts = c
                        best_r = r
            if best_r >= 0:
                selected.append((idx, best_r))
        return selected
    
    # Local search: try to improve by removing one and adding two
    def local_search(best):
        selected_idx = set(idx for (idx, _) in best)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing one element and adding two new ones
            for pos in range(len(best)):
                if time.time() > deadline - 5:
                    break
                old_idx, old_r = best[pos]
                # Try all pairs of new moduli not in selected
                for i in range(n):
                    if i in selected_idx or i == old_idx:
                        continue
                    for j in range(n):
                        if j in selected_idx or j == old_idx or j == i:
                            continue
                        # Try to find residues for i and j compatible with best minus pos
                        base = best[:pos] + best[pos+1:]
                        found = False
                        for r_i in range(moduli[i]):
                            if not check_residue(i, r_i, base):
                                continue
                            for r_j in range(moduli[j]):
                                if (r_i - r_j) % gcd_tab[i][j] == 0:
                                    continue
                                if not check_residue(j, r_j, base):
                                    continue
                                # Check compatibility between i and j
                                best[pos] = (i, r_i)
                                best.append((j, r_j))
                                selected_idx.discard(old_idx)
                                selected_idx.add(i)
                                selected_idx.add(j)
                                improved = True
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    if improved:
                        break
                if improved:
                    break
        return best
    
    best = []
    best_len = 0
    
    # Deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        # Prefer moduli with many small prime factors first (harder to place)
        sorted(range(n), key=lambda i: (-len([p for p in [2,3,5,7,11,13,17,19,23] if moduli[i] % p == 0]), moduli[i])),
        # Prefer prime moduli first (easier to place)
        sorted(range(n), key=lambda i: (len([p for p in [2,3,5,7,11,13,17,19,23] if moduli[i] % p == 0]), moduli[i])),
        # Prefer moduli with largest gcd with others (most constrained)
        sorted(range(n), key=lambda i: -sum(gcd_tab[i][j] for j in range(n) if j != i)),
    ]
    
    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)
    
    # Random restarts
    random.seed(42)
    n_restarts = 3000
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)
    
    # Local search: try to add more moduli
    if time.time() < deadline - 15:
        selected_idx = set(idx for (idx, _) in best)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    if check_residue(idx, r, best):
                        best.append((idx, r))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break
    
    # Local search: remove one, add two
    if time.time() < deadline - 5:
        best = local_search(best)
        best_len = len(best)
    
    return [[r, moduli[idx]] for (idx, r) in best]
