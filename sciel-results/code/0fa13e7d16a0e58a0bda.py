import time, random
from math import gcd, exp

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 256
    # candidates 2..256
    cand = list(range(2, N+1))
    idx = {x:i for i,x in enumerate(cand)}
    m = len(cand)

    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of all j such that lcm(cand[i], cand[j]) <= 256
    conflict_mask = [0] * m
    for i in range(m):
        a = cand[i]
        mask = 0
        for j in range(m):
            if i == j: continue
            b = cand[j]
            l = a*b//gcd(a,b)
            if l <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # weight = 1/a
    weight = [1.0/x for x in cand]

    # Greedy using bitmasks
    def greedy(order):
        sel = 0
        for i in order:
            if sel & (1 << i):
                continue
            if (sel & conflict_mask[i]) == 0:
                sel |= (1 << i)
        return sel

    def score(mask):
        s = 0.0
        for i in range(m):
            if mask & (1 << i):
                s += weight[i]
        return s

    # Simulated Annealing
    def simulated_annealing(start_mask, t_start=1.0, t_end=0.001, cooling=0.9999):
        cur_mask = start_mask
        cur_score = score(cur_mask)
        best_mask = cur_mask
        best_score = cur_score
        
        T = t_start
        
        while T > t_end:
            if time.time() > DEADLINE:
                break
            
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                for _ in range(20):
                    i = random.randrange(m)
                    if cur_mask & (1 << i):
                        continue
                    if (cur_mask & conflict_mask[i]) == 0:
                        # Valid add, delta is positive
                        cur_mask |= (1 << i)
                        cur_score += weight[i]
                        if cur_score > best_score:
                            best_score = cur_score
                            best_mask = cur_mask
                        break

            elif move_type < 0.6:
                # Remove
                for _ in range(20):
                    i = random.randrange(m)
                    if cur_mask & (1 << i):
                        delta = -weight[i]
                        if random.random() < exp(delta/T):
                            cur_mask ^= (1 << i)
                            cur_score += delta
                        break

            else:
                # Swap
                for _ in range(20):
                    i = random.randrange(m)
                    if not (cur_mask & (1 << i)):
                        continue
                    j = random.randrange(m)
                    if cur_mask & (1 << j):
                        continue
                    
                    base_mask = cur_mask ^ (1 << i)
                    if (base_mask & conflict_mask[j]) == 0:
                        delta = weight[j] - weight[i]
                        if delta > 0 or random.random() < exp(delta/T):
                            cur_mask = base_mask | (1 << j)
                            cur_score += delta
                            if cur_score > best_score:
                                best_score = cur_score
                                best_mask = cur_mask
                        break
            
            T *= cooling
        
        return best_mask, best_score

    best_mask = 0
    best_score = -1.0
    
    # Initial Greedy
    base_orders = [
        list(range(m)),
        list(range(m-1,-1,-1)),
        sorted(range(m), key=lambda i: -weight[i]),
        sorted(range(m), key=lambda i: weight[i]),
    ]
    
    for o in base_orders:
        sel = greedy(o)
        v = score(sel)
        if v > best_score:
            best_score = v
            best_mask = sel
        if time.time() > DEADLINE:
            break

    # Run SA from best
    if time.time() < DEADLINE:
        # Run multiple SA instances
        for _ in range(10):
            if time.time() > DEADLINE - 2:
                break
            # Random restarts or perturbations?
            # Let's just run SA on the current best.
            # If it gets stuck, the randomness in moves might help, 
            # but starting from a local optimum is hard.
            # Let's try starting from a random greedy solution occasionally.
            if _ % 3 == 0 and _ > 0:
                # Random greedy
                order = list(range(m))
                random.shuffle(order)
                start_sel = greedy(order)
            else:
                start_sel = best_mask
            
            bm, bs = simulated_annealing(start_sel, t_start=0.5, t_end=0.001, cooling=0.9999)
            if bs > best_score:
                best_score = bs
                best_mask = bm

    # Final extraction
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(cand[i])
    
    if not result:
        result = [2]
        
    return result
