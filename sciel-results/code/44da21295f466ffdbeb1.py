import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def eval_poly(exponents, coeffs):
        if len(set(exponents)) != 12:
            return float('inf')
        if any(e < 0 or e > 20 for e in exponents):
            return float('inf')
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return float('inf')
        sq = {}
        for i in range(12):
            for j in range(i, 12):
                e = exponents[i] + exponents[j]
                c = coeffs[i] * coeffs[j]
                if i != j:
                    c *= 2
                sq[e] = sq.get(e, 0) + c
        return sum(1 for v in sq.values() if v != 0)
    
    candidates = [
        (list(range(12)), [1]*12),
        (list(range(0, 24, 2)), [1]*12),
        (list(range(0, 24, 2)), [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]),
    ]
    for _ in range(300):
        exps = sorted(random.sample(range(0, 21), 12))
        coefs = [random.choice([-1, 1]) for _ in range(12)]
        candidates.append((exps, coefs))
    
    best_score = float('inf')
    best_exponents = None
    best_coeffs = None
    
    for exps, coefs in candidates:
        s = eval_poly(exps, coefs)
        if s < best_score:
            best_score = s
            best_exponents = exps[:]
            best_coeffs = coefs[:]
    
    cur_exps = best_exponents[:]
    cur_coefs = best_coeffs[:]
    cur_score = best_score
    
    T = 10.0
    T_min = 0.005
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        move_type = random.random()
        
        if move_type < 0.4:
            i = random.randint(0, 11)
            old_e = cur_exps[i]
            new_e = random.randint(0, 20)
            if new_e != old_e and new_e not in cur_exps:
                cur_exps[i] = new_e
                cur_exps.sort()
                new_score = eval_poly(cur_exps, cur_coefs)
                delta = new_score - cur_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    if new_score < best_score:
                        best_score = new_score
                        best_exponents = cur_exps[:]
                        best_coeffs = cur_coefs[:]
                    cur_score = new_score
                else:
                    cur_exps[i] = old_e
                    cur_exps.sort()
        elif move_type < 0.6:
            i = random.randint(0, 11)
            cur_coefs[i] *= -1
            new_score = eval_poly(cur_exps, cur_coefs)
            delta = new_score - cur_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_score < best_score:
                    best_score = new_score
                    best_exponents = cur_exps[:]
                    best_coeffs = cur_coefs[:]
                cur_score = new_score
            else:
                cur_coefs[i] *= -1
        elif move_type < 0.75:
            i = random.randint(0, 11)
            old_c = cur_coefs[i]
            sign = 1 if old_c > 0 else -1
            new_mag = random.choice([1, 1, 1, 2, 2, 3, 4, 5, 8, 16, 32])
            new_c = sign * new_mag
            if abs(new_c) <= 64:
                cur_coefs[i] = new_c
                new_score = eval_poly(cur_exps, cur_coefs)
                delta = new_score - cur_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    if new_score < best_score:
                        best_score = new_score
                        best_exponents = cur_exps[:]
                        best_coeffs = cur_coefs[:]
                    cur_score = new_score
                else:
                    cur_coefs[i] = old_c
        else:
            i, j = random.sample(range(12), 2)
            saved = cur_exps[:]
            cur_exps[i], cur_exps[j] = cur_exps[j], cur_exps[i]
            cur_exps.sort()
            new_score = eval_poly(cur_exps, cur_coefs)
            delta = new_score - cur_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_score < best_score:
                    best_score = new_score
                    best_exponents = cur_exps[:]
                    best_coeffs = cur_coefs[:]
                cur_score = new_score
            else:
                cur_exps = saved[:]
        
        T *= cooling
    
    exps = best_exponents
    coefs = best_coeffs
    
    max_deg = max(exps)
    result = [0] * (max_deg + 1)
    for e, c in zip(exps, coefs):
        result[e] = c
    
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    assert len(result) <= 41
    assert result[-1] != 0
    assert all(-64 <= x <= 64 for x in result)
    assert sum(1 for x in result if x != 0) == 12
    
    return result
