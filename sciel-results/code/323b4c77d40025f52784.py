"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _square_nonzero_count(coeffs):
    """Count nonzero terms in the square of a sparse polynomial given as (pos, coeff) list."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Build dense square
    max_deg = 2 * max(p for p, c in coeffs) if coeffs else 0
    if max_deg > 80:
        return 999  # too large, penalize
    square = [0] * (max_deg + 1)
    for i in range(n):
        pi, ci = coeffs[i]
        for j in range(n):
            pj, cj = coeffs[j]
            square[pi + pj] += ci * cj
    return sum(1 for v in square if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    best_count = 999
    best_coeffs = None
    
    # Initial random candidate
    random.seed(42)
    
    def random_candidate():
        positions = sorted(random.sample(range(0, 41), 12))
        coeffs = [(p, random.choice([-2,-1,1,2])) for p in positions]
        return coeffs
    
    # Evaluate initial
    cand = random_candidate()
    cnt = _square_nonzero_count(cand)
    if cnt < best_count:
        best_count = cnt
        best_coeffs = cand
    
    # Local search
    while time.time() < deadline:
        # Generate neighbor
        cur = list(best_coeffs)
        mutation_type = random.random()
        
        if mutation_type < 0.3:
            # Change coefficient value
            idx = random.randint(0, 11)
            new_val = random.choice([-4,-3,-2,-1,1,2,3,4])
            if new_val != cur[idx][1]:
                cur[idx] = (cur[idx][0], new_val)
        elif mutation_type < 0.5:
            # Move a term to a new position
            idx = random.randint(0, 11)
            new_pos = random.randint(0, 40)
            # Check no collision
            if all(p != new_pos for p, c in cur if (p,c) != cur[idx]):
                cur[idx] = (new_pos, cur[idx][1])
                cur.sort(key=lambda x: x[0])
        elif mutation_type < 0.7:
            # Swap two positions
            i, j = random.sample(range(12), 2)
            pi, ci = cur[i]
            pj, cj = cur[j]
            cur[i] = (pj, ci)
            cur[j] = (pi, cj)
            cur.sort(key=lambda x: x[0])
        elif mutation_type < 0.85:
            # Change one coefficient and move one term
            idx1 = random.randint(0, 11)
            idx2 = random.randint(0, 11)
            if idx1 != idx2:
                new_val = random.choice([-4,-3,-2,-1,1,2,3,4])
                cur[idx1] = (cur[idx1][0], new_val)
                new_pos = random.randint(0, 40)
                if all(p != new_pos for p, c in cur if (p,c) != cur[idx2]):
                    cur[idx2] = (new_pos, cur[idx2][1])
                    cur.sort(key=lambda x: x[0])
        else:
            # Random restart
            cur = random_candidate()
        
        cnt = _square_nonzero_count(cur)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = cur
            if best_count <= 12:
                break
    
    # Convert to coefficient list
    if best_coeffs is None:
        return [1] * 12
    
    max_pos = max(p for p, c in best_coeffs)
    result = [0] * (max_pos + 1)
    for p, c in best_coeffs:
        result[p] = c
    
    # Ensure exactly 12 nonzero terms, length <= 41, nonzero final coefficient
    nonzero = [c for c in result if c != 0]
    if len(nonzero) != 12:
        return [1] * 12
    if len(result) > 41:
        return [1] * 12
    if result[-1] == 0:
        return [1] * 12
    if any(abs(c) > 64 for c in result):
        return [1] * 12
    
    return result
# EVOLVE-BLOCK-END
