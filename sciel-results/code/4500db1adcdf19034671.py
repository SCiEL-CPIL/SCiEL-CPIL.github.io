"""Maximize pairwise-noncoprime subset of [1,255] containing 255.

255 = 3 * 5 * 17. The trivial clique is all multiples of 3 (85 elements).
We can do better by taking a union of multiples of different primes, then
removing elements that are coprime to some other element in the set.

Strategy:
1. Start with all multiples of 3 (guaranteed valid baseline, 85 elements).
2. Try adding multiples of 5 and/or 17, then greedily remove elements that
   violate the pairwise gcd>1 condition.
3. Also try local search: swap elements to increase size.
"""
import math
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Prime factors of 255
    # 255 = 3 * 5 * 17
    
    # Helper: check if a set is pairwise noncoprime
    def is_valid(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if math.gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    # Helper: find elements in s that are coprime to at least one other element
    def find_bad_elements(s):
        """Return set of elements that have gcd=1 with at least one other element in s."""
        bad = set()
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if math.gcd(lst[i], lst[j]) == 1:
                    bad.add(lst[i])
                    bad.add(lst[j])
        return bad
    
    # Helper: try to fix a set by removing bad elements iteratively
    def fix_set(s):
        s = set(s)
        changed = True
        while changed:
            changed = False
            bad = find_bad_elements(s)
            if bad:
                # Remove the element with the most conflicts first
                s -= bad
                changed = True
        return s
    
    # Baseline: multiples of 3
    base = set(range(3, 256, 3))
    best = set(base)
    best_size = len(best)
    
    # Strategy 1: Union of multiples of 3 and 5, then fix
    s = set(range(3, 256, 3)) | set(range(5, 256, 5))
    s_fixed = fix_set(s)
    if len(s_fixed) > best_size:
        best = s_fixed
        best_size = len(best)
    
    # Strategy 2: Union of multiples of 3 and 17, then fix
    s = set(range(3, 256, 3)) | set(range(17, 256, 17))
    s_fixed = fix_set(s)
    if len(s_fixed) > best_size:
        best = s_fixed
        best_size = len(best)
    
    # Strategy 3: Union of multiples of 5 and 17, then fix (must include 255)
    s = set(range(5, 256, 5)) | set(range(17, 256, 17))
    if 255 not in s:
        s.add(255)
    s_fixed = fix_set(s)
    if len(s_fixed) > best_size:
        best = s_fixed
        best_size = len(best)
    
    # Strategy 4: All numbers with gcd(x,255)>1, then fix
    s = set(x for x in range(1, 256) if math.gcd(x, 255) > 1)
    s_fixed = fix_set(s)
    if len(s_fixed) > best_size:
        best = s_fixed
        best_size = len(best)
    
    # Strategy 5: Local search - try adding one element at a time
    all_candidates = set(range(1, 256)) - best
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(all_candidates):
            trial = best | {x}
            bad = find_bad_elements(trial)
            if not bad:
                # Can add x without removal
                best = trial
                all_candidates.discard(x)
                best_size += 1
                improved = True
            else:
                # Try adding x and removing bad elements
                trial2 = trial - bad
                if len(trial2) > best_size:
                    best = trial2
                    all_candidates.discard(x)
                    best_size = len(best)
                    improved = True
                # Also try removing just the specific conflicting elements
                # (already handled by fix_set above)
    
    # Strategy 6: More aggressive local search with swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing one element and adding two
            for x in list(best):
                trial = best - {x}
                # Try to add two elements
                candidates = [y for y in range(1, 256) if y not in trial]
                for y in candidates:
                    trial_y = trial | {y}
                    bad_y = find_bad_elements(trial_y)
                    if bad_y:
                        continue
                    for z in candidates:
                        if z == y:
                            continue
                        trial_zy = trial_y | {z}
                        bad_zy = find_bad_elements(trial_zy)
                        if not bad_zy:
                            if len(trial_zy) > best_size:
                                best = trial_zy
                                best_size = len(best)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Ensure 255 is in the result
    if 255 not in best:
        best.add(255)
    
    return list(best)
