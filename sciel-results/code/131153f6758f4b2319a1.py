import time
import random
import copy

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    best_poly = None
    best_count = float('inf')
    
    # Helper: compute square and count nonzero terms
    def square_nonzero_count(coeffs, exponents):
        # coeffs and exponents are lists of length 12
        # Build a dict of degree -> coefficient
        sq = {}
        for i in range(12):
            for j in range(12):
                deg = exponents[i] + exponents[j]
                sq[deg] = sq.get(deg, 0) + coeffs[i] * coeffs[j]
        # Count nonzero
        count = sum(1 for v in sq.values() if v != 0)
        return count, sq
    
    # Helper: validate and build result
    def build_result(coeffs, exponents):
        # Check constraints
        if len(coeffs) != 12 or len(exponents) != 12:
            return None
        if any(c == 0 for c in coeffs):
            return None
        if any(c < -64 or c > 64 for c in coeffs):
            return None
        if any(e < 0 or e > 40 for e in exponents):
            return None
        if len(set(exponents)) != 12:
            return None
        # Build coefficient list
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for c, e in zip(coeffs, exponents):
            result[e] = c
        if result[-1] == 0:
            return None
        if len(result) > 41:
            return None
        return result
    
    # Initial candidate: try a few structured starts
    # Start 1: alternating signs, consecutive exponents
    exponents = list(range(12))
    coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
    cnt, _ = square_nonzero_count(coeffs, exponents)
    res = build_result(coeffs, exponents)
    if res:
        best_count = cnt
        best_poly = res
    
    # Start 2: all +1, consecutive
    coeffs2 = [1] * 12
    cnt2, _ = square_nonzero_count(coeffs2, exponents)
    res2 = build_result(coeffs2, exponents)
    if res2 and cnt2 < best_count:
        best_count = cnt2
        best_poly = res2
    
    # Local search
    def local_search(coeffs, exponents, iterations):
        nonlocal best_count, best_poly
        cur_coeffs = list(coeffs)
        cur_exponents = list(exponents)
        cur_cnt, _ = square_nonzero_count(cur_coeffs, cur_exponents)
        cur_res = build_result(cur_coeffs, cur_exponents)
        if cur_res and cur_cnt < best_count:
            best_count = cur_cnt
            best_poly = cur_res
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick a random move
            move_type = random.random()
            if move_type < 0.5:
                # Change a coefficient
                idx = random.randint(0, 11)
                old_c = cur_coeffs[idx]
                new_c = random.choice([-64,-32,-16,-8,-4,-2,-1,1,2,4,8,16,32,64])
                if new_c == old_c:
                    continue
                cur_coeffs[idx] = new_c
                new_cnt, _ = square_nonzero_count(cur_coeffs, cur_exponents)
                new_res = build_result(cur_coeffs, cur_exponents)
                if new_res and new_cnt < cur_cnt:
                    cur_cnt = new_cnt
                    if new_cnt < best_count:
                        best_count = new_cnt
                        best_poly = new_res
                else:
                    cur_coeffs[idx] = old_c
            elif move_type < 0.8:
                # Change an exponent
                idx = random.randint(0, 11)
                old_e = cur_exponents[idx]
                new_e = random.randint(0, 40)
                if new_e == old_e or new_e in cur_exponents:
                    continue
                cur_exponents[idx] = new_e
                new_cnt, _ = square_nonzero_count(cur_coeffs, cur_exponents)
                new_res = build_result(cur_coeffs, cur_exponents)
                if new_res and new_cnt < cur_cnt:
                    cur_cnt = new_cnt
                    if new_cnt < best_count:
                        best_count = new_cnt
                        best_poly = new_res
                else:
                    cur_exponents[idx] = old_e
            else:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                cur_exponents[i], cur_exponents[j] = cur_exponents[j], cur_exponents[i]
                new_cnt, _ = square_nonzero_count(cur_coeffs, cur_exponents)
                new_res = build_result(cur_coeffs, cur_exponents)
                if new_res and new_cnt < cur_cnt:
                    cur_cnt = new_cnt
                    if new_cnt < best_count:
                        best_count = new_cnt
                        best_poly = new_res
                else:
                    cur_exponents[i], cur_exponents[j] = cur_exponents[j], cur_exponents[i]
        
        return cur_coeffs, cur_exponents, cur_cnt
    
    # Run multiple restarts
    random.seed(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random starting point
        if r == 0:
            coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
            exponents = list(range(12))
        elif r == 1:
            coeffs = [1] * 12
            exponents = list(range(12))
        else:
            # Random exponents with some structure
            base = random.randint(0, 10)
            step = random.choice([1, 2, 3, 4, 5])
            exponents = sorted(set(base + step * i for i in range(12)))
            if len(exponents) < 12:
                # Fill in randomly
                while len(exponents) < 12:
                    e = random.randint(0, 40)
                    if e not in exponents:
                        exponents.append(e)
                exponents = sorted(exponents)
            exponents = exponents[:12]
            coeffs = [random.choice([-1, 1, 2, -2, 3, -3]) for _ in range(12)]
        
        iters = 200
        _, _, _ = local_search(coeffs, exponents, iters)
    
    if best_poly is None:
        # Fallback: return a valid simple polynomial
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    return best_poly
