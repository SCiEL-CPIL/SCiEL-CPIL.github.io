import time
import random
from collections import Counter

def construct():
    deadline = time.time() + 80
    
    def count_nonzero_square(exponents, signs):
        n = len(exponents)
        coeff = {}
        for i in range(n):
            for j in range(i, n):
                deg = exponents[i] + exponents[j]
                if i == j:
                    val = 1
                else:
                    val = 2 * signs[i] * signs[j]
                coeff[deg] = coeff.get(deg, 0) + val
        return sum(1 for v in coeff.values() if v != 0)
    
    def try_exponent_set(exponents):
        n = len(exponents)
        best_count = n * n + 1
        best_signs = None
        for mask in range(1 << n):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(n)]
            cnt = count_nonzero_square(exponents, signs)
            if cnt < best_count:
                best_count = cnt
                best_signs = signs[:]
            if time.time() > deadline:
                return best_count, best_signs
        return best_count, best_signs
    
    candidate_sets = []
    
    # Consecutive blocks
    for start in range(0, 29):
        candidate_sets.append(list(range(start, start + 12)))
    
    # Arithmetic progressions with various step sizes
    for d in range(2, 5):
        for start in range(0, 41 - 11 * d):
            candidate_sets.append([start + k * d for k in range(12)])
    
    # Two-block structures
    for gap in range(6, 30):
        for offset in range(0, 12):
            block1 = list(range(offset, offset + 6))
            block2 = list(range(gap + offset, gap + offset + 6))
            if max(block2) <= 40:
                candidate_sets.append(block1 + block2)
    
    # Three-block structures
    for g1 in range(5, 20):
        for g2 in range(5, 25):
            for offset in range(0, 8):
                b1 = list(range(offset, offset + 4))
                b2 = list(range(g1 + offset, g1 + offset + 4))
                b3 = list(range(g1 + g2 + offset, g1 + g2 + offset + 4))
                if max(b3) <= 40:
                    candidate_sets.append(b1 + b2 + b3)
    
    # Power-of-2 based
    candidate_sets.append([0,1,2,4,8,16,32,33,34,35,36,37])
    candidate_sets.append([0,1,2,4,8,16,17,18,19,20,21,22])
    candidate_sets.append([0,1,2,4,8,16,32,33,34,35,36,37])
    candidate_sets.append([0,1,2,4,8,16,17,18,20,24,32,40])
    candidate_sets.append([0,1,3,7,15,31,32,33,34,35,36,37])
    candidate_sets.append([0,1,2,3,4,5,35,36,37,38,39,40])
    candidate_sets.append([0,1,2,3,4,5,30,31,32,33,34,35])
    candidate_sets.append([0,1,2,3,4,5,25,26,27,28,29,30])
    candidate_sets.append([0,1,2,3,4,5,10,11,12,13,14,15])
    candidate_sets.append([0,1,2,3,4,5,15,16,17,18,19,20])
    candidate_sets.append([0,1,2,3,4,5,20,21,22,23,24,25])
    
    # Deduplicate
    seen = set()
    unique_sets = []
    for s in candidate_sets:
        if len(s) != 12 or max(s) > 40 or min(s) < 0:
            continue
        t = tuple(sorted(s))
        if t in seen:
            continue
        seen.add(t)
        unique_sets.append(list(t))
    
    # Score by collision potential
    def collision_score(exps):
        sums = Counter()
        for i in range(12):
            for j in range(i, 12):
                sums[exps[i] + exps[j]] += 1
        return sum(v * (v - 1) for v in sums.values())
    
    unique_sets.sort(key=collision_score, reverse=True)
    
    best_overall_count = 145
    best_overall_exponents = None
    best_overall_signs = None
    
    for exponents in unique_sets:
        if time.time() > deadline:
            break
        cnt, signs = try_exponent_set(exponents)
        if cnt < best_overall_count:
            best_overall_count = cnt
            best_overall_exponents = exponents[:]
            best_overall_signs = signs[:]
    
    # Use remaining time for random search
    while time.time() < deadline - 2:
        exps = sorted(random.sample(range(41), 12))
        cnt, signs = try_exponent_set(exps)
        if cnt < best_overall_count:
            best_overall_count = cnt
            best_overall_exponents = exps[:]
            best_overall_signs = signs[:]
    
    if best_overall_exponents is not None and best_overall_signs is not None:
        max_exp = max(best_overall_exponents)
        result = [0] * (max_exp + 1)
        for e, s in zip(best_overall_exponents, best_overall_signs):
            result[e] = s
        assert len(result) <= 41
        assert result[-1] != 0
        assert all(-64 <= x <= 64 for x in result)
        assert sum(1 for x in result if x != 0) == 12
        return result
    
    return [1] * 12
