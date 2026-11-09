import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    MAX_DEG = 40
    SQ_SIZE = 2 * MAX_DEG + 1  # 81
    
    valid_coeffs = [i for i in range(-64, 65) if i != 0]
    
    def score_poly(degrees, coeffs):
        sq = [0] * SQ_SIZE
        n = 12
        for i in range(n):
            di = degrees[i]
            ci = coeffs[i]
            for j in range(i, n):
                dj = degrees[j]
                cj = coeffs[j]
                deg = di + dj
                val = ci * cj
                if i == j:
                    sq[deg] += val
                else:
                    sq[deg] += 2 * val
        count = 0
        for v in sq:
            if v != 0:
                count += 1
        return count

    def init_poly():
        degrees = [0]
        coeffs = [1]
        used = {0}
        for _ in range(11):
            while True:
                d = random.randint(1, MAX_DEG)
                if d not in used:
                    used.add(d)
                    degrees.append(d)
                    break
            c = random.choice(valid_coeffs)
            coeffs.append(c)
        return degrees, coeffs

    best_score = float('inf')
    best_degrees = None
    best_coeffs = None
    
    while time.time() < deadline:
        degrees, coeffs = init_poly()
        current_score = score_poly(degrees, coeffs)
        
        if current_score < best_score:
            best_score = current_score
            best_degrees = degrees[:]
            best_coeffs = coeffs[:]
            
        # Local Search (Simulated Annealing)
        temp = 5.0
        min_temp = 0.01
        cooling = 0.995
        no_improve_counter = 0
        local_best = current_score
        local_best_degrees = degrees[:]
        local_best_coeffs = coeffs[:]
        
        while time.time() < deadline and temp > min_temp:
            idx = random.randint(1, 11)
            old_d = degrees[idx]
            old_c = coeffs[idx]
            
            # Generate move
            if random.random() < 0.5:
                new_d = old_d
                new_c = random.choice(valid_coeffs)
            else:
                new_c = old_c
                # Pick new degree
                attempts = 0
                while attempts < 10:
                    new_d = random.randint(1, MAX_DEG)
                    # Check if new_d is already used by another index
                    conflict = False
                    for k in range(12):
                        if k != idx and degrees[k] == new_d:
                            conflict = True
                            break
                    if not conflict:
                        break
                    attempts += 1
                if attempts >= 10:
                    new_d = old_d # No change
            
            degrees[idx] = new_d
            coeffs[idx] = new_c
            
            new_score = score_poly(degrees, coeffs)
            
            if new_score < local_best:
                local_best = new_score
                local_best_degrees = degrees[:]
                local_best_coeffs = coeffs[:]
                no_improve_counter = 0
                if local_best < best_score:
                    best_score = local_best
                    best_degrees = local_best_degrees[:]
                    best_coeffs = local_best_coeffs[:]
            else:
                delta = new_score - local_best
                if random.random() < math.exp(-delta / temp):
                    local_best = new_score
                    local_best_degrees = degrees[:]
                    local_best_coeffs = coeffs[:]
                    no_improve_counter = 0
                else:
                    degrees[idx] = old_d
                    coeffs[idx] = old_c
                    no_improve_counter += 1
            
            temp *= cooling
            
            if no_improve_counter > 1000:
                break

    if best_degrees is None:
        return [1] * 12
        
    pairs = sorted(zip(best_degrees, best_coeffs))
    max_deg = pairs[-1][0]
    result = [0] * (max_deg + 1)
    for deg, coeff in pairs:
        result[deg] = coeff
        
    return result
