import math
import time
import random

def _is_compatible(m1, r1, m2, r2):
    d = math.gcd(m1, m2)
    return (r1 % d) != (r2 % d)

def _find_valid_residues(m, selected):
    valid = []
    for r in range(m):
        ok = True
        for (m_i, r_i) in selected:
            if not _is_compatible(m, r, m_i, r_i):
                ok = False
                break
        if ok:
            valid.append(r)
    return valid

def _greedy_construct(random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = []
    for m in range(20, 201):
        valid_residues = _find_valid_residues(m, selected)
        if valid_residues:
            r = random.choice(valid_residues)
            selected.append((m, r))
    return selected

def _local_improve(classes, time_limit=10):
    start_time = time.time()
    selected = list(classes)
    selected_moduli = {m for m, r in selected}
    unselected = [m for m in range(20, 201) if m not in selected_moduli]
    unselected.sort()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for m_new in list(unselected):
            if (time.time() - start_time) > time_limit:
                break
            
            valids_new = _find_valid_residues(m_new, selected)
            if valids_new:
                r_new = random.choice(valids_new)
                selected.append((m_new, r_new))
                selected_moduli.add(m_new)
                if m_new in unselected:
                    unselected.remove(m_new)
                improved = True
                continue
            
            found_change = False
            for i in range(len(selected)):
                if (time.time() - start_time) > time_limit:
                    break
                m_i, r_i = selected[i]
                others = [s for j, s in enumerate(selected) if j != i]
                valid_for_i = _find_valid_residues(m_i, others)
                
                if not valid_for_i:
                    continue
                
                for r_i_new in valid_for_i:
                    if r_i_new == r_i:
                        continue
                    temp_selected = list(selected)
                    temp_selected[i] = (m_i, r_i_new)
                    
                    if _find_valid_residues(m_new, temp_selected):
                        selected[i] = (m_i, r_i_new)
                        valids = _find_valid_residues(m_new, selected)
                        r_new = random.choice(valids)
                        selected.append((m_new, r_new))
                        selected_moduli.add(m_new)
                        if m_new in unselected:
                            unselected.remove(m_new)
                        improved = True
                        found_change = True
                        break
                if found_change:
                    break
    return selected

def _score(classes):
    return sum(1.0 / m for m, r in classes)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = 0.0
    best_result = []
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = i * 12345 + 67890
        classes = _greedy_construct(seed)
        
        remaining = deadline - time.time()
        if remaining < 1:
            break
        
        time_for_improve = min(2.0, remaining * 0.5)
        improved_classes = _local_improve(classes, time_for_improve)
        
        score = _score(improved_classes)
        if score > best_score:
            best_score = score
            best_result = improved_classes
    
    if not best_result:
        assigned = []
        for m in range(20, 201):
            forbidden = set()
            for (m_i, r_i) in assigned:
                d = math.gcd(m, m_i)
                base = r_i % d
                for k in range(m // d):
                    forbidden.add((base + k * d) % m)
            r = 0
            while r < m and r in forbidden:
                r += 1
            if r < m:
                assigned.append((m, r))
        best_result = [[m, r] for (m, r) in assigned]
        
    return best_result
