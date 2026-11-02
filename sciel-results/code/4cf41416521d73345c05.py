import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute balls as lists of integers
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    for w in range(total):
        ball = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                ball.append(new_w)
        balls.append(ball)
        
    # Precompute ball sets for fast intersection
    ball_sets = [set(b) for b in balls]
    
    # Precompute reverse mapping: for each word, which codewords cover it?
    # This is expensive to build fully, but we can use it for finding candidates.
    # Actually, for 2-exchange, we need to find c3 covering U_12.
    # U_12 is a set of words. c3 must be in intersection of balls of those words?
    # No, c3 is a codeword. Its ball must contain U_12.
    # So c3 must be in the set of all codewords whose ball intersects U_12?
    # No, c3's ball must *contain* U_12.
    # So for each w in U_12, c3 must be in {c | w in balls[c]}.
    # Let's precompute `coverers[w]` = list of codewords c such that w is in balls[c].
    coverers = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            coverers[w].append(c)
            
    best_size = total
    best_set = None
    
    # Run multiple restarts
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # To speed up greedy, we can sample candidates if the pool is large
        # But 6561 is small.
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break
            
            # Count coverage for each candidate among uncovered words
            # We can optimize this by only checking candidates that cover at least one uncovered word
            # But initially all do.
            
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid due to timeout
        
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: 1-Swap (Remove 1, Add 1)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find c_prime that covers all critical_words
                # c_prime must be in intersection of coverers[w] for w in critical_words
                candidates = set(coverers[critical_words[0]])
                for cw in critical_words[1:]:
                    candidates.intersection_update(coverers[cw])
                    if not candidates:
                        break
                
                candidates.discard(c)
                
                if candidates:
                    c_prime = random.choice(list(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        # Exhaustive search for pairs
        improved_2ex = True
        while improved_2ex and time.time() < deadline - 5.0:
            improved_2ex = False
            N = len(selected)
            # To avoid O(N^2) becoming too slow if N is large, we can limit iterations
            # But N is ~500, N^2 is 250k. Each check is fast.
            
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                c1 = selected[i]
                for j in range(i + 1, N):
                    c2 = selected[j]
                    
                    # Calculate U_12: words that become uncovered if c1 and c2 are removed
                    # w becomes uncovered if count[w] becomes 0.
                    # count[w] decreases by 1 for w in B_c1, by 1 for w in B_c2.
                    # If w in B_c1 ∩ B_c2, count drops by 2. Uncovered if count was 2.
                    # If w in (B_c1 ∪ B_c2) \ (B_c1 ∩ B_c2), count drops by 1. Uncovered if count was 1.
                    
                    b1 = balls[c1]
                    b2 = balls[c2]
                    s1 = set(b1)
                    s2 = set(b2)
                    
                    inter = s1.intersection(s2)
                    sym_diff = s1.symmetric_difference(s2)
                    
                    u12 = []
                    for w in inter:
                        if cover_count[w] == 2:
                            u12.append(w)
                    for w in sym_diff:
                        if cover_count[w] == 1:
                            u12.append(w)
                            
                    if not u12:
                        # Both can be removed
                        cover_count[b1] -= 1
                        cover_count[b2] -= 1
                        selected = [c for c in selected if c != c1 and c != c2]
                        improved_2ex = True
                        break
                    else:
                        # Find c3 that covers all words in u12
                        candidates = set(coverers[u12[0]])
                        for uw in u12[1:]:
                            candidates.intersection_update(coverers[uw])
                            if not candidates:
                                break
                        
                        candidates.discard(c1)
                        candidates.discard(c2)
                        
                        if candidates:
                            c3 = random.choice(list(candidates))
                            # Check if c3 is already in selected?
                            # If c3 is in selected, we are just moving c1, c2 to c3?
                            # No, we remove c1, c2 and add c3.
                            # If c3 is already there, we just remove c1, c2.
                            # But if c3 is there, its count increases.
                            # Is it valid? Yes.
                            
                            cover_count[b1] -= 1
                            cover_count[b2] -= 1
                            cover_count[balls[c3]] += 1
                            
                            # Remove c1, c2 from selected
                            # If c3 is in selected, it stays.
                            # We just remove c1 and c2.
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            if c3 not in new_selected:
                                new_selected.append(c3)
                            selected = new_selected
                            improved_2ex = True
                            break
                if improved_2ex:
                    break
                    
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486:
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
