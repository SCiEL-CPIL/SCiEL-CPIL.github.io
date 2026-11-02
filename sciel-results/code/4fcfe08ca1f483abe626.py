import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback for other parameters
    # Simple greedy for general case
    return _general_greedy(q, n, radius, deadline)

def _general_greedy(q, n, radius, deadline):
    total = q ** n
    weights = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls
    # This is expensive for large n, but n=8 is small
    # For general q, n, we might not be able to precompute all balls if q^n is huge.
    # But the task is specific to 3^8.
    # For fallback, we just return a trivial solution or simple greedy.
    
    # Trivial solution
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls as indices
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # 1. Greedy Initialization - vectorized
    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered = np.ones(total, dtype=bool)
    
    while np.any(uncovered) and time.time() < deadline - 30.0:
        # Count how many uncovered words each candidate covers
        counts = np.sum(uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        # Pick the first one, or randomize to avoid bias?
        # Randomization can help escape local minima in greedy
        best_idx = candidates[random.randint(0, len(candidates)-1)]
        
        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered = (cov == 0)
    
    # 2. Local Search
    in_code = np.zeros(total, dtype=bool)
    for c in selected:
        in_code[c] = True
        
    def add_code(c):
        cov[balls[c]] += 1
        in_code[c] = True
        selected.append(c)
        
    def remove_code(c):
        cov[balls[c]] -= 1
        in_code[c] = False
        selected.remove(c)
        
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # Phase 1: Remove redundant codewords
        # A codeword c is redundant if all words in ball[c] are covered by other codewords
        # i.e., min(cov[ball[c]]) > 1
        for c in list(selected):
            if not in_code[c]:
                continue
            if np.min(cov[balls[c]]) > 1:
                remove_code(c)
                improved = True
                
        # Phase 2: Single Swap
        # Try to replace one codeword with another to reduce size or improve structure
        # We look for a codeword c such that its unique neighbors can be covered by a single new codeword w
        for c in list(selected):
            if not in_code[c]:
                continue
            
            # Find words uniquely covered by c
            unique_neighbors = balls[c][cov[balls[c]] == 1]
            if len(unique_neighbors) == 0:
                continue  # This would have been removed in Phase 1
            
            # Find candidates w that cover all unique_neighbors
            # w must be in the intersection of balls[unique_neighbors]
            valid_w_set = set(balls[unique_neighbors[0]].tolist())
            for v in unique_neighbors[1:]:
                valid_w_set.intersection_update(balls[v].tolist())
                
            # Filter out words already in code
            w_list = [w for w in valid_w_set if not in_code[w]]
            if not w_list:
                continue
                
            # Evaluate candidates
            best_w = -1
            best_score = -1
            for w in w_list:
                # Score: How many words does w cover that are already covered >1 times?
                # Maximizing this helps with future redundancy removal
                score = np.sum(cov[balls[w]] > 1)
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1:
                remove_code(c)
                add_code(best_w)
                improved = True
                break  # Restart loop to re-evaluate redundancy
                
        # Phase 3: Pair Swap (limited)
        if not improved and len(selected) >= 2 and time.time() < deadline - 5.0:
            # Try to swap two codewords for two others
            # This is expensive, so we limit tries
            n_selected = len(selected)
            tries = 0
            max_tries = 20
            while tries < max_tries and time.time() < deadline - 5.0:
                tries += 1
                i = random.randint(0, n_selected - 1)
                j = random.randint(0, n_selected - 1)
                if i == j:
                    continue
                c1 = selected[i]
                c2 = selected[j]
                if not in_code[c1] or not in_code[c2]:
                    continue
                
                # Temporarily remove c1, c2
                cov[balls[c1]] -= 1
                cov[balls[c2]] -= 1
                in_code[c1] = False
                in_code[c2] = False
                selected.remove(c1)
                selected.remove(c2)
                
                # Find uncovered words
                uncovered_words = np.where(cov == 0)[0]
                if len(uncovered_words) == 0:
                    # Just re-add
                    add_code(c1)
                    add_code(c2)
                    continue
                
                # Try to find w1, w2 that cover uncovered_words
                # w1 must cover some, w2 the rest
                # Simple heuristic: pick w1 to cover max uncovered, then w2 for rest
                best_pair = None
                best_score = -1
                
                # Candidates for w1: words in ball[uncovered_words]
                cand_w1 = set()
                for v in uncovered_words:
                    cand_w1.update(balls[v].tolist())
                cand_w1 = [w for w in cand_w1 if not in_code[w]]
                
                if not cand_w1:
                    add_code(c1)
                    add_code(c2)
                    continue
                
                for w1 in cand_w1[:10]: # Limit
                    covered1 = set(balls[w1].tolist())
                    remaining = [v for v in uncovered_words if v not in covered1]
                    if not remaining:
                        # w1 covers all. Need w2 to be any valid non-code word?
                        # No, we need to add 2 words to replace 2.
                        # If w1 covers all uncovered, we still need to add w2?
                        # The size remains the same (remove 2, add 2).
                        # We want w2 to help with redundancy.
                        # Pick w2 that maximizes redundancy.
                        all_w = [w for w in range(total) if not in_code[w] and w != w1]
                        if not all_w: continue
                        # Pick best w2
                        max_red = -1
                        best_w2 = all_w[0]
                        for w2 in all_w[:10]:
                            red = np.sum(cov[balls[w2]] > 1)
                            if red > max_red:
                                max_red = red
                                best_w2 = w2
                        score = np.sum(cov[balls[w1]] > 1) + max_red
                        if score > best_score:
                            best_score = score
                            best_pair = (w1, best_w2)
                    else:
                        # Find w2 in intersection of balls[remaining]
                        valid_w2 = set(balls[remaining[0]].tolist())
                        for v in remaining[1:]:
                            valid_w2.intersection_update(balls[v].tolist())
                        valid_w2 = [w for w in valid_w2 if not in_code[w] and w != w1]
                        if not valid_w2: continue
                        for w2 in valid_w2[:5]:
                            score = np.sum(cov[balls[w1]] > 1) + np.sum(cov[balls[w2]] > 1)
                            if score > best_score:
                                best_score = score
                                best_pair = (w1, w2)
                
                if best_pair:
                    w1, w2 = best_pair
                    add_code(w1)
                    add_code(w2)
                    improved = True
                    break
                else:
                    add_code(c1)
                    add_code(c2)

    # Final cleanup
    for c in list(selected):
        if in_code[c] and np.min(cov[balls[c]]) > 1:
            remove_code(c)
            
    # Verify
    final_cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cov[balls[c]] += 1
        
    if np.any(final_cov == 0):
        missing = np.where(final_cov == 0)[0]
        for m in missing:
            add_code(m)
            final_cov[balls[m]] += 1
            
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
