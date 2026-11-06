import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    best_score = 21  # Initial worst case (max possible nonzeros is 23, but we start high)
    best_candidate = [1] * 12  # Fallback: 12 terms, square has 23 nonzeros? No, [1]*12 is x^0..x^11. Square has 23 terms.
    
    # Actually, [1]*12 is not valid if we need exactly 12 nonzero terms and length <= 41.
    # [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # That's 12 terms.
    
    def compute_square_nonzeros(exps, coeffs):
        square = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            square[2 * ei] += ci * ci
            for j in range(i + 1, n):
                square[ei + exps[j]] += 2 * ci * coeffs[j]
        return sum(1 for v in square.values() if v != 0)

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            dense = [0] * (max(exps) + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense
        return score

    def local_search(exps, coeffs, max_iters=2000, deadline_t=None):
        exps = list(exps)
        coeffs = list(coeffs)
        current_score = compute_square_nonzeros(exps, coeffs)
        
        # Coefficients to try
        coeff_choices = [-3, -2, -1, 1, 2, 3]
        
        for it in range(max_iters):
            if deadline_t and time.time() > deadline_t:
                break
            improved = False
            
            # 1. Try coefficient changes
            for i in range(12):
                old = coeffs[i]
                for new_c in coeff_choices:
                    if new_c == old:
                        continue
                    coeffs[i] = new_c
                    s = compute_square_nonzeros(exps, coeffs)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        coeffs[i] = old
                if improved:
                    break
            
            if improved:
                continue

            # 2. Try exponent changes
            for i in range(12):
                old_e = exps[i]
                # Try a random subset of new exponents to save time
                candidates = [e for e in range(0, 41) if e != old_e and e not in exps]
                random.shuffle(candidates)
                for new_e in candidates[:20]:  # Limit candidates
                    exps[i] = new_e
                    s = compute_square_nonzeros(exps, coeffs)
                    if s < current_score:
                        current_score = s
                        improved = True
                        break
                    else:
                        exps[i] = old_e
                
                if improved:
                    break
            
            if not improved:
                # Stagnation: random kick
                if random.random() < 0.1:
                    # Change 2 random exponents
                    idx1, idx2 = random.sample(range(12), 2)
                    old_e1, old_e2 = exps[idx1], exps[idx2]
                    new_e1 = random.randint(0, 40)
                    new_e2 = random.randint(0, 40)
                    if new_e1 not in exps and new_e2 not in exps and new_e1 != new_e2:
                        exps[idx1] = new_e1
                        exps[idx2] = new_e2
                        s = compute_square_nonzeros(exps, coeffs)
                        if s < current_score + 2:  # Allow slight worsening to escape
                            current_score = s
                        else:
                            exps[idx1] = old_e1
                            exps[idx2] = old_e2
                else:
                    break
                    
        return exps, coeffs, current_score

    def run_search():
        nonlocal best_score, best_candidate
        random.seed(12345)
        
        # Strategy 1: Start with dense arithmetic progression
        for start in range(0, 10):
            for step in [1, 2]:
                exps = [start + i * step for i in range(12)]
                if max(exps) > 40:
                    continue
                coeffs = [1] * 12
                # Randomize some coefficients
                for _ in range(20):
                    if time.time() > deadline - 10:
                        break
                    c = list(coeffs)
                    for i in range(12):
                        if random.random() < 0.5:
                            c[i] = random.choice([-1, 1])
                    le, lc, ls = local_search(exps, c, max_iters=1000, deadline_t=deadline - 5)
                    try_candidate(le, lc)
        
        # Strategy 2: Random starts
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            le, lc, ls = local_search(exps, coeffs, max_iters=1500, deadline_t=deadline - 2)
            try_candidate(le, lc)
            
        # Strategy 3: Refine best candidate
        if best_score < 21 and time.time() < deadline - 2:
            best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
            best_coeffs = [best_candidate[i] for i in best_exps]
            le, lc, ls = local_search(best_exps, best_coeffs, max_iters=3000, deadline_t=deadline - 1)
            try_candidate(le, lc)

    run_search()

    return best_candidate

# EVOLVE-BLOCK-END
