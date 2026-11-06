import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 1000
    best_result = None
    
    def evaluate(exps, coefs):
        poly = np.zeros(41, dtype=int)
        for e, c in zip(exps, coefs):
            if 0 <= e <= 40:
                poly[e] = c
            else:
                return 1000
        sq = np.convolve(poly, poly)
        return np.count_nonzero(sq)

    def make_result(exps, coefs):
        max_deg = max(exps)
        result = [0] * (max_deg + 1)
        for e, c in zip(exps, coefs):
            result[e] = c
        # Ensure no trailing zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    def random_init():
        exps = random.sample(range(41), 12)
        coefs = []
        for _ in range(12):
            c = random.randint(-64, 64)
            while c == 0:
                c = random.randint(-64, 64)
            coefs.append(c)
        return exps, coefs

    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        exps, coefs = random_init()
        current_score = evaluate(exps, coefs)
        
        if current_score < best_score:
            best_score = current_score
            best_result = make_result(exps, coefs)
            
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.005
        
        steps = 0
        max_steps = 100000
        
        while steps < max_steps and time.time() < deadline - 5.0:
            idx = random.randint(0, 11)
            
            if random.random() < 0.5:
                old_e = exps[idx]
                new_e = random.randint(0, 40)
                # Check collision
                if new_e != old_e and new_e in exps:
                    # If collision, we can swap or reject. Reject is simpler.
                    continue
                
                exps[idx] = new_e
                new_score = evaluate(exps, coefs)
                
                delta = new_score - current_score
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_result = make_result(exps, coefs)
                else:
                    exps[idx] = old_e
            else:
                old_c = coefs[idx]
                new_c = random.randint(-64, 64)
                if new_c == 0:
                    continue
                
                coefs[idx] = new_c
                new_score = evaluate(exps, coefs)
                
                delta = new_score - current_score
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_result = make_result(exps, coefs)
                else:
                    coefs[idx] = old_c
            
            T *= cooling_rate
            steps += 1
            
            # Optional: If we hit a very low score, we can stop early?
            # Theoretical min for 12 terms? 
            # If we find 6, it's very good.
            if best_score <= 6:
                break

    if best_result is None:
        # Fallback: 12 terms, dense
        return [1] * 12
        
    return best_result
