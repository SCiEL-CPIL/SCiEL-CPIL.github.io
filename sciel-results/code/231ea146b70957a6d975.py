import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def count_nonzero_square(exps, coefs):
        square_coefs = {}
        for i in range(12):
            ci = coefs[i]
            ei = exps[i]
            for j in range(12):
                cj = coefs[j]
                ej = exps[j]
                power = ei + ej
                val = ci * cj
                if power in square_coefs:
                    square_coefs[power] += val
                else:
                    square_coefs[power] = val
        return sum(1 for v in square_coefs.values() if v != 0)

    def get_polynomial_list(exps, coefs):
        max_deg = max(exps)
        if max_deg > 40:
            return None
        for c in coefs:
            if c < -64 or c > 64 or c == 0:
                return None
        poly = [0] * (max_deg + 1)
        for e, c in zip(exps, coefs):
            poly[e] = c
        # Ensure exactly 12 non-zero terms
        if sum(1 for c in poly if c != 0) != 12:
            return None
        return poly

    best_score = float('inf')
    best_exps = None
    best_coefs = None
    best_poly = None

    def update_best(exps, coefs, score):
        nonlocal best_score, best_exps, best_coefs, best_poly
        if score < best_score:
            best_score = score
            best_exps = exps[:]
            best_coefs = coefs[:]
            best_poly = get_polynomial_list(exps, coefs)

    # Initial candidates
    # 1. Consecutive 1s
    exps0 = list(range(12))
    coefs0 = [1] * 12
    update_best(exps0, coefs0, count_nonzero_square(exps0, coefs0))
    
    # 2. Consecutive alternating
    exps1 = list(range(12))
    coefs1 = [1 if i % 2 == 0 else -1 for i in range(12)]
    update_best(exps1, coefs1, count_nonzero_square(exps1, coefs1))

    # 3. Random starts
    for _ in range(10):
        window_size = 25
        offset = random.randint(0, 15)
        while True:
            exps = random.sample(range(offset, offset + window_size), 12)
            if len(set(exps)) == 12:
                break
        coefs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        score = count_nonzero_square(exps, coefs)
        update_best(exps, coefs, score)

    # Simulated Annealing
    current_exps = best_exps[:]
    current_coefs = best_coefs[:]
    current_score = best_score

    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        r = random.random()
        if r < 0.5:
            # Change coefficient
            idx = random.randint(0, 11)
            old_c = current_coefs[idx]
            choices = [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]
            new_c = random.choice(choices)
            if new_c == old_c:
                continue
            current_coefs[idx] = new_c
            new_score = count_nonzero_square(current_exps, current_coefs)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    update_best(current_exps, current_coefs, new_score)
            else:
                current_coefs[idx] = old_c
        else:
            # Change exponent
            idx = random.randint(0, 11)
            old_e = current_exps[idx]
            new_e = random.randint(0, 40)
            if new_e in current_exps and new_e != old_e:
                continue
            current_exps[idx] = new_e
            new_score = count_nonzero_square(current_exps, current_coefs)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    update_best(current_exps, current_coefs, new_score)
            else:
                current_exps[idx] = old_e
                
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Final Hill Climb
    # Try to improve the best found
    h_exps = best_exps[:]
    h_coefs = best_coefs[:]
    h_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single moves
        for i in range(12):
            # Coef changes
            old_c = h_coefs[i]
            for nc in [-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64]:
                if nc == old_c: continue
                h_coefs[i] = nc
                ns = count_nonzero_square(h_exps, h_coefs)
                if ns < h_score:
                    h_score = ns
                    improved = True
                    break
                h_coefs[i] = old_c
            
            # Exp changes
            old_e = h_exps[i]
            for ne in range(0, 41):
                if ne == old_e: continue
                if ne in h_exps: continue
                h_exps[i] = ne
                ns = count_nonzero_square(h_exps, h_coefs)
                if ns < h_score:
                    h_score = ns
                    improved = True
                    break
                h_exps[i] = old_e
            
            if improved:
                break

    if h_score < best_score:
        update_best(h_exps, h_coefs, h_score)

    if best_poly is None:
        return [1] * 12

    return best_poly
