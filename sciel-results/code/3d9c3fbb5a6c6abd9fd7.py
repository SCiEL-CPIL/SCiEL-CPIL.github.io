# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute forbidden masks: for each m, for each g in [2, m], for each k in [0, g):
    #   mask of r in [0, m) where r % g == k
    precomputed_masks = {}
    for m in moduli:
        precomputed_masks[m] = {}
        for g in range(2, m + 1):
            precomputed_masks[m][g] = {}
            for k in range(g):
                mask = 0
                for r in range(k, m, g):
                    mask |= (1 << r)
                precomputed_masks[m][g][k] = mask
    
    def compute_valid_mask(m, existing_classes):
        all_mask = (1 << m) - 1
        valid = all_mask
        for (m2, r2) in existing_classes:
            i = mod_idx[m]
            j = mod_idx[m2]
            g = gcd_table[i][j]
            if g == 1:
                return 0
            k = r2 % g
            forbidden = precomputed_masks[m][g][k]
            valid &= ~forbidden
        return valid
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def pick_residue(valid_mask, m, existing_classes):
        if valid_mask == 0:
            return None
        candidates = []
        r = 0
        while r < m:
            if (valid_mask >> r) & 1:
                candidates.append(r)
            r += 1
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        # Score by flexibility: count how many other moduli remain feasible
        used_moduli = set(m2 for m2, r2 in existing_classes)
        
        def score_candidate(r):
            total = 0
            for m2 in moduli:
                if m2 in used_moduli or m2 == m:
                    continue
                valid2 = compute_valid_mask(m2, existing_classes + [[m, r]])
                if count_bits(valid2) > 0:
                    total += 1
            return total
        
        if len(candidates) > 15:
            step = max(1, len(candidates) // 15)
            candidates = candidates[::step][:15]
        
        best_r = candidates[0]
        best_s = -1
        for r in candidates:
            s = score_candidate(r)
            if s > best_s:
                best_s = s
                best_r = r
            if time.time() > deadline - 5:
                break
        return best_r
    
    def score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def greedy_construct(order):
        classes = []
        for m in order:
            valid = compute_valid_mask(m, classes)
            if count_bits(valid) == 0:
                continue
            r = pick_residue(valid, m, classes)
            if r is not None:
                classes.append([m, r])
        return classes
    
    def local_search(classes, max_iter=400):
        best = [list(c) for c in classes]
        best_score = score(best)
        used_moduli = set(m for m, r in best)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            for m in sorted(moduli):
                if m in used_moduli:
                    continue
                valid = compute_valid_mask(m, best)
                if count_bits(valid) == 0:
                    continue
                r = pick_residue(valid, m, best)
                if r is None:
                    continue
                new_classes = best + [[m, r]]
                new_score = score(new_classes)
                if new_score > best_score:
                    best = new_classes
                    best_score = new_score
                    used_moduli.add(m)
                    improved = True
                    break
            
            if not improved:
                for i in range(len(best)):
                    if time.time() > deadline - 5:
                        break
                    m_rem = best[i][0]
                    temp = [best[j] for j in range(len(best)) if j != i]
                    temp_used = set(m for m, r in temp)
                    
                    for m in sorted(moduli):
                        if m in temp_used:
                            continue
                        valid = compute_valid_mask(m, temp)
                        if count_bits(valid) == 0:
                            continue
                        r = pick_residue(valid, m, temp)
                        if r is None:
                            continue
                        new_classes = temp + [[m, r]]
                        new_score = score(new_classes)
                        if new_score > best_score:
                            best = new_classes
                            best_score = new_score
                            used_moduli = set(m for m, r in best)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        return best
    
    # Focus on even moduli (all share factor 2, so no coprime conflicts)
    even_moduli = [m for m in moduli if m % 2 == 0]
    
    order1 = sorted(even_moduli)
    classes1 = greedy_construct(order1)
    classes1 = local_search(classes1, max_iter=300)
    
    best_classes = classes1
    best_score = score(classes1)
    
    random.seed(42)
    for start in range(25):
        if time.time() > deadline - 10:
            break
        alpha = 1.5 + random.random()
        weights = [(m, 1.0 / (m ** alpha)) for m in even_moduli]
        
        order = []
        remaining = list(weights)
        while remaining:
            total_w = sum(w for _, w in remaining)
            if total_w <= 0:
                order.append(remaining[0][0])
                remaining.pop(0)
                continue
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, (m, w) in enumerate(remaining):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx][0])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=200)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    if best_classes:
        best_classes = local_search(best_classes, max_iter=300)
    
    return best_classes if best_classes else []
# EVOLVE-BLOCK-END
