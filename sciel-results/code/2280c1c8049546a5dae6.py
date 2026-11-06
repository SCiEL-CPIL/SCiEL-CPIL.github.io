# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _square_nonzero_count(coeffs):
    sq = defaultdict(int)
    n = len(coeffs)
    for i in range(n):
        ei, ci = coeffs[i]
        for j in range(i, n):
            ej, cj = coeffs[j]
            if i == j:
                sq[ei + ej] += ci * ci
            else:
                sq[ei + ej] += 2 * ci * cj
    return sum(1 for v in sq.values() if v != 0)

def _coeffs_to_list(coeffs):
    max_deg = max(e for e, c in coeffs)
    lst = [0] * (max_deg + 1)
    for e, c in coeffs:
        lst[e] = c
    return lst

def _random_polynomial():
    exponents = random.sample(range(0, 41), 12)
    coefficients = [random.choice([-1, 1]) * random.randint(1, 64) for _ in range(12)]
    return list(zip(exponents, coefficients))

def _local_search(coeffs, time_budget):
    best = list(coeffs)
    best_cnt = _square_nonzero_count(best)
    while _time_left() > time_budget:
        idx = random.randint(0, 11)
        e, c = best[idx]
        new_e = e + random.choice([-2, -1, 1, 2])
        if new_e < 0 or new_e > 40:
            continue
        if any(new_e == be for i, (be, bc) in enumerate(best) if i != idx):
            continue
        new_c = c + random.choice([-1, 1, -2, 2, -3, 3])
        if new_c == 0 or abs(new_c) > 64:
            continue
        new_coeffs = list(best)
        new_coeffs[idx] = (new_e, new_c)
        new_cnt = _square_nonzero_count(new_coeffs)
        if new_cnt < best_cnt:
            best = new_coeffs
            best_cnt = new_cnt
            if best_cnt <= 12:
                break
    return best, best_cnt

def _candidate_patterns():
    candidates = []
    # Alternating consecutive
    candidates.append([(i, 1 if i % 2 == 0 else -1) for i in range(12)])
    # AP with alternating signs
    for step in [2, 3, 4]:
        exponents = [i * step for i in range(12)]
        if max(exponents) <= 40:
            candidates.append([(exponents[i], 1 if i % 2 == 0 else -1) for i in range(12)])
    # Difference structure P = Q - x^m Q
    for m in [12, 15, 18, 20]:
        for q_step in [1, 2, 3]:
            q_exponents = [i * q_step for i in range(6)]
            if max(q_exponents) + m <= 40:
                coeffs = []
                for e in q_exponents:
                    coeffs.append((e, 1))
                for e in q_exponents:
                    coeffs.append((e + m, -1))
                if len(set(e for e, c in coeffs)) == 12:
                    candidates.append(coeffs)
    # All ones consecutive
    candidates.append([(i, 1) for i in range(12)])
    return candidates

def construct():
    random.seed(12345)
    best_coeffs = None
    best_cnt = float('inf')
    
    for coeffs in _candidate_patterns():
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_cnt:
            best_cnt = cnt
            best_coeffs = coeffs
        if best_cnt <= 12:
            break
    
    while _time_left() > 5.0:
        coeffs = _random_polynomial()
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_cnt:
            best_cnt = cnt
            best_coeffs = coeffs
            if best_cnt <= 12:
                break
        if cnt <= best_cnt + 4:
            improved, improved_cnt = _local_search(coeffs, time_budget=2.0)
            if improved_cnt < best_cnt:
                best_cnt = improved_cnt
                best_coeffs = improved
                if best_cnt <= 12:
                    break
    
    if best_coeffs is None:
        return [1] * 12
    return _coeffs_to_list(best_coeffs)
# EVOLVE-BLOCK-END
