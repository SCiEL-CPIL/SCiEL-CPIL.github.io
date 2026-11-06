# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
FRAC_DENOMS = [Fraction(1, d) for d in DENOMINATORS]
ONE = Fraction(1, 1)

def _try_improve(best_idx, best_deficit, deadline):
    """Local search: 1-swap, 1-add moves to reduce deficit while keeping sum < 1."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Try 1-swap: remove i, add j
        for i in best_idx[:]:
            if time.time() >= deadline:
                break
            f_out = FRAC_DENOMS[i]
            sum_without = best_sum - f_out
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                new_sum = sum_without + f_in
                if new_sum >= ONE:
                    continue
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.remove(i)
                    best_idx.append(j)
                    in_set.remove(i)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
            if changed:
                break
        
        # Try 1-add: add j if it fits
        if not changed:
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                new_sum = best_sum + f_in
                if new_sum >= ONE:
                    continue
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.append(j)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
    
    return best_idx, best_deficit

def _triple_swap_improve(best_idx, best_deficit, deadline):
    """Try triple swaps (remove three, add three) for larger jumps."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    k = len(best_idx)
    
    if k < 3:
        return best_idx, best_deficit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_remove = len(best_idx)
        for a in range(n_remove):
            if time.time() >= deadline:
                break
            for b in range(a+1, n_remove):
                if time.time() >= deadline:
                    break
                for c in range(b+1, n_remove):
                    if time.time() >= deadline:
                        break
                    i, j, l = best_idx[a], best_idx[b], best_idx[c]
                    sum_without_three = best_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j] - FRAC_DENOMS[l]
                    for x in range(N):
                        if x in in_set:
                            continue
                        f_x = FRAC_DENOMS[x]
                        sum_after_x = sum_without_three + f_x
                        if sum_after_x >= ONE:
                            continue
                        for y in range(x+1, N):
                            if y in in_set:
                                continue
                            f_y = FRAC_DENOMS[y]
                            sum_after_xy = sum_after_x + f_y
                            if sum_after_xy >= ONE:
                                continue
                            for z in range(y+1, N):
                                if z in in_set:
                                    continue
                                f_z = FRAC_DENOMS[z]
                                new_sum = sum_after_xy + f_z
                                if new_sum >= ONE:
                                    continue
                                new_def = ONE - new_sum
                                if new_def < best_deficit:
                                    best_idx.remove(i)
                                    best_idx.remove(j)
                                    best_idx.remove(l)
                                    best_idx.append(x)
                                    best_idx.append(y)
                                    best_idx.append(z)
                                    in_set.remove(i)
                                    in_set.remove(j)
                                    in_set.remove(l)
                                    in_set.add(x)
                                    in_set.add(y)
                                    in_set.add(z)
                                    best_sum = new_sum
                                    best_deficit = new_def
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_idx, best_deficit

def _simulated_annealing(best_idx, best_deficit, deadline, rng):
    """Simulated annealing to escape local optima."""
    import math
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    T = Fraction(1, 100)  # Initial temperature
    T_min = Fraction(1, 10000000)
    cooling = 0.995
    
    current_sum = best_sum
    current_deficit = best_deficit
    current_idx = list(best_idx)
    current_in_set = set(best_idx)
    
    while time.time() < deadline and T > T_min:
        # Generate a random move
        move_type = rng.choice(['swap', 'add', 'remove', 'double_swap'])
        
        new_idx = list(current_idx)
        new_in_set = set(current_in_set)
        new_sum = current_sum
        
        if move_type == 'swap' and len(current_idx) > 0:
            i = rng.choice(current_idx)
            candidates = [j for j in range(N) if j not in current_in_set]
            if candidates:
                j = rng.choice(candidates)
                new_sum = current_sum - FRAC_DENOMS[i] + FRAC_DENOMS[j]
                if new_sum < ONE:
                    new_idx.remove(i)
                    new_idx.append(j)
                    new_in_set.remove(i)
                    new_in_set.add(j)
                else:
                    continue
        elif move_type == 'add':
            candidates = [j for j in range(N) if j not in current_in_set]
            if candidates:
                j = rng.choice(candidates)
                new_sum = current_sum + FRAC_DENOMS[j]
                if new_sum < ONE:
                    new_idx.append(j)
                    new_in_set.add(j)
                else:
                    continue
        elif move_type == 'remove' and len(current_idx) > 1:
            i = rng.choice(current_idx)
            new_sum = current_sum - FRAC_DENOMS[i]
            new_idx.remove(i)
            new_in_set.remove(i)
        elif move_type == 'double_swap' and len(current_idx) >= 2:
            i, j = rng.sample(current_idx, 2)
            candidates = [k for k in range(N) if k not in current_in_set]
            if len(candidates) >= 2:
                x, y = rng.sample(candidates, 2)
                new_sum = current_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j] + FRAC_DENOMS[x] + FRAC_DENOMS[y]
                if new_sum < ONE:
                    new_idx.remove(i)
                    new_idx.remove(j)
                    new_idx.append(x)
                    new_idx.append(y)
                    new_in_set.remove(i)
                    new_in_set.remove(j)
                    new_in_set.add(x)
                    new_in_set.add(y)
                else:
                    continue
        else:
            continue
        
        new_deficit = ONE - new_sum
        delta = new_deficit - current_deficit
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 0 and rng.random() < math.exp(-float(delta) / float(T))):
            current_idx = new_idx
            current_in_set = new_in_set
            current_sum = new_sum
            current_deficit = new_deficit
            
            if new_deficit < best_deficit:
                best_idx = list(current_idx)
                best_deficit = new_deficit
                best_sum = new_sum
        
        T *= cooling
    
    return best_idx, best_deficit

def _greedy_seed():
    """Greedy: pick fractions in order until sum would exceed 1."""
    selected, total = [], Fraction()
    for i, d in enumerate(DENOMINATORS):
        if total + Fraction(1, d) < ONE:
            selected.append(i)
            total += Fraction(1, d)
    return selected, total

def _random_seed(rng, target_size=None):
    """Random subset that stays below 1."""
    if target_size is None:
        target_size = rng.randint(3, 8)
    selected = []
    total = Fraction()
    candidates = list(range(N))
    rng.shuffle(candidates)
    for idx in candidates:
        if len(selected) >= target_size:
            break
        f = FRAC_DENOMS[idx]
        if total + f < ONE:
            selected.append(idx)
            total += f
    if not selected:
        selected = [0]
        total = Fraction(1, 2)
    return selected, total

def _known_good_seeds():
    """Return a list of known good starting points."""
    seeds = []
    
    # 1/2 + 1/3 + 1/7 + 1/43 = 1805/1806, deficit = 1/1806
    s1 = [0, 1, 5, 41]
    sum1 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    if sum1 < ONE:
        seeds.append((s1, sum1))
    
    # 1/2 + 1/3 + 1/8 + 1/24 + 1/360... let's try various
    # 1/2 + 1/4 + 1/6 + 1/12 + 1/20 + 1/60 = 1 exactly, so need to be below
    # 1/2 + 1/4 + 1/6 + 1/12 + 1/20 = 59/60, deficit = 1/60
    s2 = [0, 2, 4, 10, 18]
    sum2 = Fraction(1,2) + Fraction(1,4) + Fraction(1,6) + Fraction(1,12) + Fraction(1,20)
    if sum2 < ONE:
        seeds.append((s2, sum2))
    
    # 1/2 + 1/3 + 1/10 + 1/15 = 0.5+0.333+0.1+0.0667 = 0.9997, deficit ~ 1/300
    s3 = [0, 1, 8, 13]
    sum3 = Fraction(1,2) + Fraction(1,3) + Fraction(1,10) + Fraction(1,15)
    if sum3 < ONE:
        seeds.append((s3, sum3))
    
    # 1/2 + 1/3 + 1/7 + 1/42 = 1 exactly, so use 1/43 instead
    # Already have that
    
    # 1/2 + 1/3 + 1/9 + 1/18 + 1/36 = 0.5+0.333+0.111+0.0556+0.0278 = 1.0278 > 1, no
    # 1/2 + 1/3 + 1/9 + 1/18 = 0.9722, deficit = 1/36
    s4 = [0, 1, 7, 16]
    sum4 = Fraction(1,2) + Fraction(1,3) + Fraction(1,9) + Fraction(1,18)
    if sum4 < ONE:
        seeds.append((s4, sum4))
    
    return seeds

def construct():
    deadline = time.time() + 50  # Leave 10s safety margin
    rng = random.Random(42)
    
    # Start with greedy seed
    best_idx, best_sum = _greedy_seed()
    best_deficit = ONE - best_sum
    
    # Try known good starting points
    for seed_idx, seed_sum in _known_good_seeds():
        if seed_sum < ONE:
            seed_def = ONE - seed_sum
            if seed_def < best_deficit:
                best_idx = list(seed_idx)
                best_sum = seed_sum
                best_deficit = seed_def
    
    # Local search on initial seed
    best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _triple_swap_improve(best_idx, best_deficit, deadline)
    
    # Simulated annealing
    best_idx, best_deficit = _simulated_annealing(best_idx, best_deficit, deadline, rng)
    
    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        # Generate random seed with varying sizes
        target_size = rng.randint(3, 10)
        seed_idx, seed_sum = _random_seed(rng, target_size)
        
        # Local search
        idx, deficit = _try_improve(seed_idx, ONE - seed_sum, deadline)
        idx, deficit = _triple_swap_improve(idx, deficit, deadline)
        
        # Simulated annealing on this result
        idx, deficit = _simulated_annealing(idx, deficit, deadline, rng)
        
        if deficit < best_deficit:
            best_idx = idx
            best_deficit = deficit
    
    # Final refinement
    best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _triple_swap_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _simulated_annealing(best_idx, best_deficit, deadline, rng)
    
    return best_idx
# EVOLVE-BLOCK-END
