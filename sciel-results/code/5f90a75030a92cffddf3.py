import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_score = float('inf')
    best_exps = None
    best_coefs = None
    
    max_deg = 40
    
    def _score(exps, coefs):
        # Fast check for distinct exponents
        if len(set(exps)) != 12:
            return 1000
        if any(e < 0 or e > max_deg for e in exps):
            return 1000
        if any(c == 0 or c < -64 or c > 64 for c in coefs):
            return 1000
            
        p = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coefs):
            p[e] += c
        p_sq = np.convolve(p, p)
        return np.count_nonzero(p_sq)

    def _init():
        exps = sorted(random.sample(range(max_deg + 1), 12))
        # Try small coefficients first
        coefs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        return exps, coefs

    # Initial seed
    exps = list(range(12))
    coefs = [1] * 12
    s = _score(exps, coefs)
    if s < best_score:
        best_score = s
        best_exps = exps.copy()
        best_coefs = coefs.copy()

    # Random restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        exps, coefs = _init()
        current_score = _score(exps, coefs)
        
        # Local search
        # Use a simple hill climbing with random restarts within the loop
        steps = 0
        max_steps = 5000
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if random.random() < 0.5:
                # Coeff
                idx = random.randint(0, 11)
                old_c = coefs[idx]
                new_c = random.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
                if new_c == old_c:
                    continue
                coefs[idx] = new_c
                ns = _score(exps, coefs)
                if ns < current_score:
                    current_score = ns
                    if current_score < best_score:
                        best_score = current_score
                        best_exps = exps.copy()
                        best_coefs = coefs.copy()
                else:
                    coefs[idx] = old_c
            else:
                # Exp
                idx = random.randint(0, 11)
                old_e = exps[idx]
                new_e = random.randint(0, max_deg)
                # Check distinct
                if new_e != old_e:
                    if new_e in exps:
                        # Find index of new_e
                        # If it's not idx, reject
                        # Since exps is a list, we can check
                        if exps.index(new_e) != idx:
                            continue
                exps[idx] = new_e
                ns = _score(exps, coefs)
                if ns < current_score:
                    current_score = ns
                    if current_score < best_score:
                        best_score = current_score
                        best_exps = exps.copy()
                        best_coefs = coefs.copy()
                else:
                    exps[idx] = old_e

    # Build result
    if best_exps is None:
        return [1] * 12
        
    max_exp = max(best_exps)
    result = [0] * (max_exp + 1)
    for e, c in zip(best_exps, best_coefs):
        result[e] = c
        
    if result[-1] == 0:
        result[-1] = 1
        
    return result
