import random
import itertools
from collections import Counter

def construct():
    def count_collisions(s):
        """Count the number of duplicate unordered pair sums (including diagonals) in set s."""
        s = sorted(s)
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        c = Counter(sums)
        return sum(v - 1 for v in c.values() if v > 1)
    
    def largest_sidon_size(s):
        """Find the largest Sidon subset size in s."""
        s = sorted(s)
        n = len(s)
        best = 1
        # Try all subsets from largest to smallest
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(s, size):
                subset = list(subset)
                sums = set()
                valid = True
                for i in range(len(subset)):
                    for j in range(i, len(subset)):
                        sm = subset[i] + subset[j]
                        if sm in sums:
                            valid = False
                            break
                        sums.add(sm)
                    if not valid:
                        break
                if valid:
                    best = size
                    found = True
                    break
            if found:
                break
        return best
    
    def optimize(s):
        """Greedily swap elements to maximize collisions using steepest descent."""
        s = list(s)
        current_score = count_collisions(s)
        
        improved = True
        # Limit iterations to prevent infinite loops in case of ties (though score is integer, ties are possible)
        # But steepest descent on integer landscape usually converges.
        max_iters = 50
        for _ in range(max_iters):
            if not improved:
                break
            improved = False
            for i in range(12):
                best_swap_val = s[i]
                best_score = current_score
                original_val = s[i]
                
                # Try swapping s[i] with each possible value in [0, 128]
                for new_val in range(129):
                    if new_val == original_val:
                        continue
                    # Check if new_val is already in the set (at a different index)
                    # If so, we can't simply replace s[i] with new_val without handling the duplicate.
                    # The original code skipped these. We maintain that logic.
                    if new_val in s and new_val != s[i]:
                        continue
                    
                    s[i] = new_val
                    score = count_collisions(s)
                    if score > best_score:
                        best_score = score
                        best_swap_val = new_val
                
                s[i] = best_swap_val
                if best_swap_val != original_val:
                    current_score = best_score
                    improved = True
        
        return s

    # Start with several random initializations and pick the best
    random.seed(42)
    candidates = []
    
    # 1. Random restarts
    for trial in range(100):
        s = random.sample(range(129), 12)
        s_opt = optimize(s)
        candidates.append(s_opt)
        
    # 2. Structured starts
    structured_starts = [
        list(range(12)),
        list(range(0, 129, 11))[:12],
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 124],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 7, 15, 31, 63, 127, 2, 5, 11, 23],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ]
    for start in structured_starts:
        if len(set(start)) == 12 and all(0 <= x <= 128 for x in start):
            s_opt = optimize(list(start))
            candidates.append(s_opt)

    # Evaluate actual largest Sidon size for all candidates
    # We want to MINIMIZE this size
    best_candidate = None
    min_sidon_size = 13 # Max possible is 12
    
    # Remove duplicates in candidates to save time
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    for c in unique_candidates:
        sidon_size = largest_sidon_size(c)
        if sidon_size < min_sidon_size:
            min_sidon_size = sidon_size
            best_candidate = c
            
    # If we didn't find any (shouldn't happen), fallback to first candidate
    if best_candidate is None:
        best_candidate = candidates[0]
        
    # Verify
    assert len(best_candidate) == 12
    assert len(set(best_candidate)) == 12
    assert all(0 <= x <= 128 for x in best_candidate)
    
    return list(best_candidate)
