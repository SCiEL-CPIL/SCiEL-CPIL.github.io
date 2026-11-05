import math
import random
import time

def construct():
    max_mod = 128
    start_time = time.time()
    time_limit = 55 # seconds
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    moduli = list(range(2, max_mod + 1))
    
    # Initial solution: Greedy
    def get_greedy_solution():
        random.shuffle(moduli)
        chosen = []
        for m in moduli:
            placed = False
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    if not are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    placed = True
                    break
        return chosen

    best = get_greedy_solution()
    best_len = len(best)
    
    current = list(best)
    
    # Local Search / Random Walk
    while time.time() - start_time < time_limit:
        # Randomly modify current
        action = random.choice(['add', 'remove', 'change'])
        
        if action == 'add':
            # Pick a random unselected modulus
            unselected = [m for m in moduli if m not in [m2 for _, m2 in current]]
            if not unselected:
                continue
            m = random.choice(unselected)
            # Try to find a valid residue
            valid_res = []
            for r in range(m):
                ok = True
                for (r2, m2) in current:
                    if not are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    valid_res.append(r)
            if valid_res:
                r = random.choice(valid_res)
                new_current = current + [(r, m)]
                if len(new_current) > best_len:
                    best_len = len(new_current)
                    best = list(new_current)
                # Accept with probability 1 if improves, else maybe? 
                # For now, always accept to explore, but keep best.
                current = new_current
                
        elif action == 'remove':
            if current:
                idx = random.randint(0, len(current)-1)
                current.pop(idx)
                
        elif action == 'change':
            if current:
                idx = random.randint(0, len(current)-1)
                r, m = current[idx]
                valid_res = []
                for r_new in range(m):
                    if r_new == r: continue
                    # Check against all others
                    ok = True
                    for j, (r2, m2) in enumerate(current):
                        if j == idx: continue
                        if not are_disjoint(r_new, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        valid_res.append(r_new)
                if valid_res:
                    r_new = random.choice(valid_res)
                    current[idx] = (r_new, m)
        
        # If we hit the max possible, break
        if best_len == len(moduli):
            break
            
    if not best:
        return [[0, 2]]
        
    return [[r, m] for r, m in best]
