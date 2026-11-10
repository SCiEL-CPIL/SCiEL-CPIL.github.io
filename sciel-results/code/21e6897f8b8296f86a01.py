import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def find_valid_residue(m_new, selected):
        """Find a residue r for m_new compatible with all in selected."""
        for r in range(m_new):
            ok = True
            for m_i, r_i in selected:
                g = math.gcd(m_new, m_i)
                if r % g == r_i % g:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def score(classes):
        return sum(1.0 / m for m, r in classes)
    
    # Phase 1: Greedy construction - add smallest moduli first
    best_selected = []
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, best_selected)
        if r is not None:
            best_selected.append((m, r))
    
    best_score = score(best_selected)
    best_result = [[m, r] for m, r in best_selected]
    
    # Phase 2: Local search
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline - 2.0:
            break
        
        if len(best_selected) == 0:
            break
        
        current = best_selected[:]
        i = random.randint(0, len(current) - 1)
        removed = current.pop(i)
        
        found_improvement = False
        for m in range(20, 201):
            if m in [x[0] for x in current] or m == removed[0]:
                continue
            r = find_valid_residue(m, current)
            if r is not None:
                new_selected = current + [(m, r)]
                new_score = score(new_selected)
                if new_score > best_score + 1e-12:
                    best_selected = new_selected
                    best_score = new_score
                    best_result = [[mm, rr] for mm, rr in best_selected]
                    found_improvement = True
                    break
        
        if not found_improvement:
            current.append(removed)
    
    return best_result
