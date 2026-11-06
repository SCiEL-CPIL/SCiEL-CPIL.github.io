import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Moduli are even numbers in [20, 200]
    moduli = list(range(20, 201, 2))
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute GCD table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Weights for sorting in B&B (smaller m -> higher weight 1/m)
    # Sort moduli by ascending m to prioritize high weights
    sorted_moduli = sorted(moduli)
    sorted_indices = [idx[m] for m in sorted_moduli]
    weights = [1.0 / m for m in sorted_moduli]
    
    def is_valid(classes):
        nc = len(classes)
        for i in range(nc):
            m1, r1 = classes[i]
            i_idx = idx[m1]
            for j in range(i + 1, nc):
                m2, r2 = classes[j]
                j_idx = idx[m2]
                g = gcd_table[i_idx][j_idx]
                if r1 % g == r2 % g:
                    return False
        return True
        
    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)
        
    def greedy_init(order):
        classes = []
        present = set()
        for m in order:
            mi = idx[m]
            if m in present:
                continue
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in classes:
                    g = gcd_table[mi][idx[m2]]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    present.add(m)
                    found = True
                    break
        return classes
        
    def local_search(classes, max_iters=5000):
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()
        
        rng = random.Random(123)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try to add a missing modulus
            if missing:
                added = False
                # Try smallest missing first
                for m in missing:
                    mi = idx[m]
                    for r in range(m):
                        valid = True
                        for m2, r2 in classes:
                            g = gcd_table[mi][idx[m2]]
                            if r % g == r2 % g:
                                valid = False
                                break
                        if valid:
                            classes.append([m, r])
                            present.add(m)
                            missing.remove(m)
                            score += 1.0 / m
                            added = True
                            break
                    if added:
                        break
            
            if added:
                continue
                
            # If can't add, try to reassign one existing class to allow adding a missing one
            if missing:
                m_target = missing[0]
                mi_target = idx[m_target]
                
                # Find best residue for target with min conflicts
                best_r_target = None
                min_conf = float('inf')
                for r in range(m_target):
                    conf = 0
                    for m2, r2 in classes:
                        g = gcd_table[mi_target][idx[m2]]
                        if r % g == r2 % g:
                            conf += 1
                    if conf < min_conf:
                        min_conf = conf
                        best_r_target = r
                        if conf == 0:
                            break
                
                if min_conf <= 1:
                    # Identify conflicting classes
                    conf_js = []
                    for j, (m2, r2) in enumerate(classes):
                        g = gcd_table[mi_target][idx[m2]]
                        if best_r_target % g == r2 % g:
                            conf_js.append(j)
                    
                    for j in conf_js:
                        mj, rj = classes[j]
                        mj_i = idx[mj]
                        g_jt = gcd_table[mj_i][mi_target]
                        target_bad = best_r_target % g_jt
                        
                        # Try to move class j to a new residue
                        found_move = False
                        for new_r in range(mj):
                            if new_r % g_jt == target_bad:
                                continue
                            ok = True
                            for k, (m2, r2) in enumerate(classes):
                                if k == j:
                                    continue
                                g = gcd_table[mj_i][idx[m2]]
                                if new_r % g == r2 % g:
                                    ok = False
                                    break
                            if ok:
                                classes[j][1] = new_r
                                classes.append([m_target, best_r_target])
                                present.add(m_target)
                                missing.remove(m_target)
                                score += 1.0 / m_target
                                found_move = True
                                improved = True
                                break
                        if found_move:
                            break
            
            if improved:
                continue
                
            # Random Kick: reassign a random class to a random valid residue
            if classes:
                j = rng.randint(0, len(classes) - 1)
                mj, rj = classes[j]
                mj_i = idx[mj]
                for _ in range(10):
                    new_r = rng.randint(0, mj - 1)
                    if new_r == rj:
                        continue
                    ok = True
                    for k, (m2, r2) in enumerate(classes):
                        if k == j:
                            continue
                        g = gcd_table[mj_i][idx[m2]]
                        if new_r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        classes[j][1] = new_r
                        break
                        
        return classes, score

    # 1. Initial Greedy
    best_classes = greedy_init(sorted_moduli)
    best_score = score_of(best_classes)
    
    # 2. Local Search on Greedy
    best_classes, best_score = local_search(best_classes, max_iters=1000)
    
    # 3. Multi-start Local Search
    rng = random.Random(42)
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        shuffled = sorted_moduli[:]
        rng.shuffle(shuffled)
        # Weighted shuffle to still prefer small moduli
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 20))
        cand = greedy_init(weighted)
        cand, s = local_search(cand, max_iters=500)
        if s > best_score:
            best_score = s
            best_classes = cand
            
    # 4. Branch and Bound
    # We use the best_score from LS as the lower bound.
    # B&B tries to beat it.
    
    # Precompute suffix sums of weights for upper bound
    # weights are sorted by ascending m (descending 1/m)
    suffix_weights = [0.0] * (len(sorted_moduli) + 1)
    for i in range(len(sorted_moduli) - 1, -1, -1):
        suffix_weights[i] = suffix_weights[i+1] + weights[i]
        
    def bb_search(index, current_classes, current_score):
        nonlocal best_score, best_classes
        
        if time.time() > deadline - 2:
            return
            
        # Upper bound
        if current_score + suffix_weights[index] <= best_score:
            return
            
        if index == len(sorted_moduli):
            if current_score > best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
            return
            
        m = sorted_moduli[index]
        mi = idx[m]
        rest_indices = range(index + 1, len(sorted_moduli))
        
        # Option 1: Skip m
        bb_search(index + 1, current_classes, current_score)
        
        # Option 2: Include m with some residue r
        # To improve pruning, we can check if adding m is even possible
        # without violating constraints with current_classes.
        # If no r is valid, skip immediately.
        
        valid_residues = []
        for r in range(m):
            ok = True
            for m2, r2 in current_classes:
                g = gcd_table[mi][idx[m2]]
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
                
        if not valid_residues:
            return
            
        # Try each valid residue
        # Heuristic: try residues that might be "less restrictive" to future moduli?
        # Hard to know. Just try all.
        for r in valid_residues:
            new_classes = current_classes + [[m, r]]
            bb_search(index + 1, new_classes, current_score + 1.0 / m)
            
    # Start B&B
    # Note: B&B on 100 items is huge. We rely on the strong lower bound from LS.
    # If LS is near optimal, B&B will prune almost everything.
    # If not, B&B might time out. We have the time check.
    
    # To make B&B faster, we can limit the depth or use a heuristic order for residues.
    # Let's try to run it.
    
    # We start with an empty set of classes for B&B?
    # No, we want to find a set that beats best_score.
    # Starting from scratch in B&B is expensive.
    # However, the bound `current_score + suffix_weights[index]` is strong if we start from index 0.
    # At index 0, bound is sum(1/m).
    # If best_score is close to sum(1/m), we prune early.
    
    # Let's run B&B from scratch.
    bb_search(0, [], 0.0)
    
    # Final check
    if not is_valid(best_classes):
        # Fallback
        best_classes = greedy_init(sorted_moduli)
        best_score = score_of(best_classes)
        
    return best_classes

