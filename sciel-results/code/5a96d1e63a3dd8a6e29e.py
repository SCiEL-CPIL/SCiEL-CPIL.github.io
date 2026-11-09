import itertools
import random

def construct():
    n = 192
    best_set = []
    best_size = 0
    candidates = list(range(1, n + 1))
    
    # Try a few random permutations
    # Also try sorted descending as a baseline
    orders = [list(range(n, 0, -1))]
    for _ in range(50):
        random.shuffle(candidates)
        orders.append(candidates[:])
        
    for order in orders:
        selected = []
        for x in order:
            valid = True
            
            # Check 1: x must not divide sum of any 3 in selected
            # O(K^2) check using residues
            if len(selected) >= 3:
                residues_x = {}
                for val in selected:
                    r = val % x
                    residues_x[r] = residues_x.get(r, 0) + 1
                
                found = False
                for u, v in itertools.combinations(selected, 2):
                    r_u = u % x
                    r_v = v % x
                    target_r = (-r_u - r_v) % x
                    count = residues_x.get(target_r, 0)
                    
                    if target_r == r_u and target_r == r_v:
                        if count >= 3:
                            found = True
                            break
                    elif target_r == r_u or target_r == r_v:
                        if count >= 2:
                            found = True
                            break
                    else:
                        if count >= 1:
                            found = True
                            break
                if found:
                    valid = False
            if not valid:
                continue
            
            # Check 2: For each a in selected, a must not divide x + u + v
            # O(K^2) check using residues
            for a in selected:
                target = (-x) % a
                residues_a = {}
                for val in selected:
                    if val == a:
                        continue
                    r = val % a
                    residues_a[r] = residues_a.get(r, 0) + 1
                
                violation_found = False
                for u in selected:
                    if u == a:
                        continue
                    r_u = u % a
                    target_r = (target - r_u) % a
                    count = residues_a.get(target_r, 0)
                    if count > 0:
                        if target_r == r_u:
                            if count >= 2:
                                violation_found = True
                                break
                        else:
                            violation_found = True
                            break
                if violation_found:
                    valid = False
                    break
                    
            if valid:
                selected.append(x)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    return best_set
