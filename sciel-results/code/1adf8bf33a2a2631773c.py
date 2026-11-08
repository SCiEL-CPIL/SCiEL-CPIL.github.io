import time
from math import log, floor
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_set(U):
    """Faster scoring for small sets using list-based approach."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    n = len(U)
    
    for i in range(n):
        a = U[i]
        for j in range(n):
            b = U[j]
            s = a + b
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            d = abs(a - b)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            c = 2*a - b
            if 0 < c <= limit_val and c not in U_set:
                candidates.add(c)
            for k in range(n):
                c_val = U[k]
                if c_val == a or c_val == b:
                    continue
                t = a + b - c_val
                if 0 < t <= limit_val and t not in U_set:
                    candidates.add(t)
            
    for a in U:
        for mult in [2, 3, 4, 5]:
            val = a * mult
            if 0 < val <= limit_val and val not in U_set:
                candidates.add(val)
    
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = fast_score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 1.5) + 10)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.45:
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = fast_score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.65:
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = fast_score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = fast_score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def spectral_seed(bound, k):
    """
    Use spectral clustering on a graph where nodes are integers 1..bound.
    Edge weight between i and j is higher if |i-j| is small (clustering nearby numbers)
    but we want to separate additive structures. 
    Actually, for this problem, we want to find a subset that maximizes |U-U|/|U+U|.
    Large differences come from elements far apart. Small sums come from elements close together.
    So we want elements spread out but not too far (to keep max U small relative to the spread).
    
    Let's try a different spectral approach:
    We want to select k elements from 1..bound.
    Let's define a graph where nodes are 1..bound.
    We want to find a subset that is "well-spread" but "compact" in terms of sumset.
    
    Actually, let's just use the spectral gap of a specific graph.
    Consider the graph G on vertices {1, ..., bound}.
    Connect i and j if i+j is a square? No, that's too specific.
    
    Let's stick to the heuristic:
    The score is 1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
    We want to maximize the ratio |U-U|/|U+U|.
    |U-U| is at most n(n-1)+1. |U+U| is at least n(n+1)/2.
    The ratio is maximized when U is "Sidon-like" for sums (so |U+U| is large? No, we want |U+U| small relative to |U-U|).
    Wait, |U+U| >= |U-U| is not always true?
    For U = {0, 1, 4, 9, 16}, |U+U| = 15, |U-U| = 21. Ratio > 1.
    We want |U+U| to be as small as possible and |U-U| to be as large as possible.
    |U-U| is large if elements are spread out.
    |U+U| is small if elements are "additively dependent" or clustered?
    No, |U+U| is minimized when U is an arithmetic progression?
    For AP {0, 1, ..., n-1}, |U+U| = 2n-1, |U-U| = 2n-1. Ratio = 1. Score = 1.
    For powers of 2, |U+U| = n(n+1)/2, |U-U| = n(n-1)+1. Ratio ~ 1.
    
    Actually, the best sets found so far have ratio > 1.
    Let's look at the structure of the best set found by the seed: [0, 1, 4, 9, 11].
    Sums: {0, 1, 2, 4, 5, 8, 9, 10, 11, 13, 15, 18, 20, 22} -> 14 sums.
    Diffs: {0, ±1, ±3, ±4, ±5, ±7, ±8, ±10, ±11} -> 19 diffs.
    Ratio = 19/14 = 1.357.
    Max U = 11. D = 23.
    Score = 1 + log(1.357)/log(23) = 1 + 0.305/3.135 = 1.097.
    
    The seed program found a score of 1.028. My manual calc for [0,1,4,9,11] gives 1.097.
    Why is the seed score lower?
    Ah, the seed program returns the best set found. The score 1.028 is the score of the *returned* set.
    If I hardcode [0, 1, 4, 9, 11], I should get ~1.097.
    Let's check the seed's best_set initialization: [0, 1, 4, 9, 11].
    It runs exhaustive search for n=3,4,5,6.
    It might find a better set.
    
    Let's try to find a better set manually or via better search.
    The goal is to maximize ratio / log(2*max(U)+1).
    Increasing max(U) decreases the score (log in denominator).
    So we want small max(U) and large ratio.
    Small max(U) means small n or small values.
    Small n means small |U-U| and |U+U|.
    
    Let's try n=6.
    If we can get ratio 2 with max(U)=20, D=41, log(41)=3.7. Score = 1 + log(2)/3.7 = 1.18.
    If we can get ratio 10 with max(U)=100, D=201, log(201)=5.3. Score = 1 + log(10)/5.3 = 1.46.
    
    Can we get ratio 10?
    Max |U-U| for n elements is n(n-1)+1.
    Min |U+U| for n elements is n(n+1)/2.
    Ratio max ~ (n^2) / (n^2/2) = 2.
    So ratio is bounded by ~2 for large n?
    Wait, |U+U| can be much larger than n(n+1)/2 if U is not an AP.
    For powers of 2, |U+U| = n(n+1)/2. |U-U| = n(n-1)+1. Ratio ~ 1.
    For "random" sets, |U+U| ~ n^2/2, |U-U| ~ n^2/2. Ratio ~ 1.
    To get ratio > 1, we need |U-U| > |U+U|.
    This happens when U is "sum-free" like? No.
    It happens when U has many differences but few sums.
    This is related to "Sidon sets" but for differences vs sums.
    
    Actually, if U is a set of powers of 2, sums are distinct, diffs are distinct.
    |U+U| = n(n+1)/2. |U-U| = n(n-1)+1.
    Ratio = (n^2-n+1) / (n^2/2 + n/2) ~ 2.
    So powers of 2 give ratio ~ 2.
    Score = 1 + log(2) / log(2*2^(n-1)+1) = 1 + log(2) / (n log(2)) = 1 + 1/n.
    For n=10, score = 1.1.
    For n=20, score = 1.05.
    
    The seed found 1.028. This is lower than 1.1 for powers of 2 with n=10.
    Why?
    Maybe the seed didn't find powers of 2?
    Or maybe my score formula interpretation is wrong?
    "score is the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)"
    
    Let's check powers of 2: U = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256}. n=10.
    |U+U| = 55. |U-U| = 91. Ratio = 91/55 = 1.65.
    Max U = 256. D = 513.
    Score = 1 + log(1.65)/log(513) = 1 + 0.49/6.24 = 1.078.
    
    What if we use a different base?
    U = {0, 1, 3, 9, 27, 81}. n=6.
    |U+U| = 21. |U-U| = 31. Ratio = 31/21 = 1.47.
    Max U = 81. D = 163.
    Score = 1 + log(1.47)/log(163) = 1 + 0.38/5.09 = 1.075.
    
    It seems hard to beat 1.1 with geometric sets.
    The seed found 1.028. This suggests the seed's search was not very effective or the best set is actually low.
    But inspiration 1 got 1.0391.
    
    Let's try to find a set with higher ratio and smaller max U.
    We want |U-U|/|U+U| to be large.
    This requires many differences and few sums.
    This is achieved if U is "sum-free" in the sense that sums collide? No, we want sums to be few, so sums should collide.
    But differences should be many, so differences should be distinct.
    This is a "difference set" vs "sumset".
    
    Consider U = {0, 1, 4, 10, 18, 28, 40, 54}. (Quadratic-ish)
    Differences: 1, 3, 6, 8, 10, 14, 18, 20, 24, 28, 34, 40, 54...
    Sums: 1, 4, 10, 18, 28, 40, 54, 5, 11, 19, 29, 41, 55...
    
    Let's just run a better local search.
    The key is to explore the space of sets more effectively.
    The current local search uses "generate_candidates" which is based on sums/diffs of existing elements.
    This might be too local.
    
    Let's try a "simulated annealing" style search with a better cooling schedule.
    And let's allow "jumps" to random sets to escape local minima.
    
    Also, let's try to optimize the ratio directly.
    The score is 1 + log(ratio) / log(2*max(U)+1).
    We can treat this as an objective function for local search.
    
    Let's rewrite the local search to be more aggressive.
    1. Start with a geometric set.
    2. Try to replace elements with random values in a range.
    3. Accept if score improves, or with probability exp(-delta/T) if it worsens.
    4. Cool down T over time.
    
    This should find better sets.
"""

import time
from math import log, floor, exp
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_set(U):
    """Faster scoring for small sets using list-based approach."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def simulated_annealing(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Simulated annealing to optimize the score.
    """
    U = sorted(set(U))
    current_score = fast_score_set(U)
    best_score = current_score
    best_set = list(U)
    
    T = 0.5  # Initial temperature
    cooling_rate = 0.995
    
    while time.time() < deadline - 2 and T > 0.01:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.5:
            # Replace an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Choose a new value
                # Try values near the old value or random in range
                if random.random() < 0.5:
                    new_val = max(1, min(max_val_limit, old_val + random.randint(-10, 10)))
                else:
                    new_val = random.randint(1, max_val_limit)
                
                if new_val not in U:
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) == len(U):
                        new_score = fast_score_set(new_U)
                        delta = new_score - current_score
                        if delta > 0 or random.random() < exp(delta / T):
                            U = new_U
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
        elif move_type < 0.75:
            # Add an element
            if len(U) < max_n_limit:
                new_val = random.randint(1, max_val_limit)
                if new_val not in U:
                    new_U = sorted(U + [new_val])
                    new_score = fast_score_set(new_U)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < exp(delta / T):
                        U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
        else:
            # Remove an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_U = [x for i, x in enumerate(U) if i != idx]
                new_score = fast_score_set(new_U)
                delta = new_score - current_score
                if delta > 0 or random.random() < exp(delta / T):
                    U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U)
        
        T *= cooling_rate
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 5, 25, 125, 625], # Powers of 5
        [0, 1, 10, 100, 1000], # Powers of 10
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    if time.time() < deadline - 50:
        s, c = exhaustive_search(3, 50)
        if s > best_score:
            best_score = s
            best_set = c
        
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
        s, c = exhaustive_search(5, 30)
        if s > best_score:
            best_score = s
            best_set = c
            
        if time.time() < deadline - 30:
            s, c = exhaustive_search(6, 20)
            if s > best_score:
                best_score = s
                best_set = c

    # 3. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Simulated Annealing
    seeds = [best_set]
    for cand in structured_candidates[:4]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        if restarts % 3 == 0:
            base = seeds[0]
        else:
            base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = simulated_annealing(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            seeds[0] = local_set
            if len(seeds) < 6:
                seeds.append(local_set)
            elif len(seeds) > 1:
                seeds[-1] = local_set

    # 5. Final Polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        max_iterations = 2000
        while improved and time.time() < deadline - 1 and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-30, 31):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and n < 35 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max(1, max_val - 30), max_val + 31):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = fast_score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
