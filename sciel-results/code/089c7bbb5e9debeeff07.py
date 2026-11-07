import time
import random
import math

_DEADLINE = 85.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _is_valid_set(elements):
    """
    Validates if the set of elements satisfies:
    1. Distinct elements in 1..256.
    2. All non-empty subset sums are distinct.
    3. No two distinct subset sums divide one another.
    """
    if not elements:
        return True
    n = len(elements)
    if n > 10: # Sanity check, max possible is small
        return False
        
    # Check distinct elements and range
    if len(set(elements)) != n:
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
            
    # Compute all subset sums
    sums = set()
    for x in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        # Check distinctness: new_sums should be disjoint from sums
        # Actually, if we just union, we lose info on count.
        # Better:
        sums = sums | new_sums
        
    # Check distinctness count
    if len(sums) != (1 << n) - 1:
        return False
        
    # Check divisibility
    # Convert to list for sorted iteration? Or just use set for lookup.
    # We need to check if any s divides t for s != t.
    # Iterate s in sums:
    #   for multiple in range(2*s, max_sum+1, s):
    #       if multiple in sums: return False
    max_sum = max(sums)
    sums_set = sums # already a set
    
    for s in sums:
        # Check multiples
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
            
    return True

def _try_construct_backtrack(candidates, target_size, current_set, current_sums, deadline_check):
    """
    Recursive backtracking to find a set of size target_size.
    candidates: list of available numbers.
    current_set: list of chosen numbers.
    current_sums: set of subset sums of current_set.
    """
    if deadline_check() < 0.5:
        return None
        
    if len(current_set) == target_size:
        return current_set[:]
        
    # Pruning: if not enough candidates left
    if len(candidates) < target_size - len(current_set):
        return None
        
    # Heuristic: Try candidates in a specific order
    # We pass candidates sorted or shuffled externally.
    
    for i, x in enumerate(candidates):
        if deadline_check() < 0.5:
            return None
            
        # Check if x is valid to add
        # 1. Distinctness: x not in current_set (guaranteed by candidates list if we remove used)
        # 2. Subset sum distinctness:
        #    new_sums = current_sums | {s + x for s in current_sums} | {x}
        #    We need len(new_sums) == len(current_sums) * 2 + 1
        #    This is equivalent to: current_sums and (current_sums + x) are disjoint, and x not in current_sums.
        
        # Quick check: x not in current_sums
        if x in current_sums:
            continue
            
        # Check disjointness of current_sums and current_sums + x
        # s1 = s2 + x  => s1 - s2 = x
        # We can check if any s in current_sums has s-x in current_sums.
        # Or just generate the set and check size.
        # Generating set is O(2^k). For k=7, 128 elements. Fast.
        
        new_sums = set()
        # Add x
        new_sums.add(x)
        # Add s+x
        disjoint = True
        for s in current_sums:
            ns = s + x
            # If ns is already in new_sums, collision
            if ns in new_sums:
                disjoint = False
                break
            new_sums.add(ns)
        
        if not disjoint:
            continue
            
        # Also need to ensure new_sums doesn't intersect current_sums?
        # No, new_sums is the set of ALL sums.
        # Wait, my logic above:
        # Old sums: S.
        # New sums: S' = S \cup (S+x) \cup {x}.
        # Distinctness requires S \cap (S+x) = \emptyset, x \notin S, x \notin S+x.
        # x \notin S is checked.
        # x \notin S+x is true since x>0 and S has non-negative sums?
        # S contains 0? No, non-empty sums.
        # If S has 1, and x=1, then 1 is in S+x? No, 1+1=2.
        # But if x=1, and 1 in S, then x in S. Checked.
        # So the loop above checking `ns in new_sums` handles collisions within (S+x) \cup {x}.
        # But we also need to check if any `ns` is in `S`?
        # No, S is a subset of S'.
        # The condition for distinct sums is that all 2^(k+1)-1 sums are distinct.
        # This is equivalent to S \cap (S+x) = \emptyset and x \notin S \cup (S+x).
        # My loop added x and s+x to new_sums.
        # If any s+x == x, then s=0. But S has no 0.
        # If any s+x == x' (another new sum), caught.
        # If any s+x is in S?
        # If s+x = s_old, then x = s_old - s.
        # This means two sums in S differ by x.
        # This is exactly S \cap (S+x) != \emptyset.
        # So we need to check if `ns` is in `current_sums`.
        
        # Let's re-verify the disjointness check.
        # We need S and S+x to be disjoint.
        # So for any s in S, s+x should NOT be in S.
        # And x should NOT be in S.
        # And elements in S+x should be distinct from each other (guaranteed if S is distinct).
        # And x should be distinct from S+x (guaranteed if x>0 and S has no 0? 
        #   If s+x = x => s=0. S has no 0. So yes.)
        
        # So the check is:
        # 1. x not in current_sums
        # 2. For all s in current_sums, (s+x) not in current_sums
        
        valid_addition = True
        if x in current_sums:
            valid_addition = False
        else:
            for s in current_sums:
                if (s + x) in current_sums:
                    valid_addition = False
                    break
        
        if not valid_addition:
            continue
            
        # Construct full new sums set for divisibility check
        # new_sums_set = current_sums | {s+x for s in current_sums} | {x}
        # We can compute this.
        
        new_sums_set = set(current_sums)
        new_sums_set.add(x)
        for s in current_sums:
            new_sums_set.add(s + x)
            
        # Check divisibility on new_sums_set
        # Optimization: Only check if necessary?
        # For k < 7, it's fast.
        
        max_sum = max(new_sums_set)
        is_div_free = True
        for s in new_sums_set:
            multiple = s * 2
            while multiple <= max_sum:
                if multiple in new_sums_set:
                    is_div_free = False
                    break
                multiple += s
            if not is_div_free:
                break
                
        if not is_div_free:
            continue
            
        # Recursive step
        # Remove x from candidates
        remaining_candidates = candidates[:i] + candidates[i+1:]
        result = _try_construct_backtrack(remaining_candidates, target_size, current_set + [x], new_sums_set, deadline_check)
        if result:
            return result
            
    return None

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    
    # Try to find a set of size 8
    # Use a few random orderings
    best_result = None
    best_size = 0
    
    # Pre-compute a "good" order: maybe primes first?
    # Or just random.
    
    rng = random.Random(42)
    
    # Try target size 8
    # To make backtracking faster, we can limit the depth or use heuristics.
    # The current backtracking is exhaustive for the given candidate list.
    # If the candidate list is large (256), it will be slow.
    # We should reduce the candidate pool.
    # Which numbers are "good"?
    # Large numbers are less likely to divide others.
    # Primes are good.
    
    # Let's try a pool of top 50 largest numbers, then top 50 primes, etc.
    
    # Strategy 1: Backtracking on a reduced pool of "promising" candidates
    # Promising: Large values, Primes.
    
    candidates_large = [x for x in pool if x > 100]
    candidates_prime = [x for x in pool if all(x % i != 0 for i in range(2, int(math.sqrt(x)) + 1))]
    
    # Merge and unique
    # Try a few random subsets of the pool
    
    attempts = 0
    while _remaining() > 5.0 and attempts < 20:
        attempts += 1
        
        # Create a random candidate list
        # Bias towards larger numbers
        random_pool = pool[:]
        rng.shuffle(random_pool)
        
        # Sort by a heuristic: larger is better?
        # Let's try sorting descending
        random_pool.sort(reverse=True)
        
        # Limit pool size to speed up backtracking
        # If we take 30 candidates, backtracking depth 8 is manageable?
        # 30 choose 8 is large, but pruning is strong.
        
        pool_subset = random_pool[:40]
        
        result = _try_construct_backtrack(pool_subset, 8, [], set(), _remaining)
        
        if result:
            # Validate fully just in case
            if _is_valid_set(result):
                best_result = result
                best_size = len(result)
                break # Found size 8, stop? Or try for 9?
                # Let's try to see if we can get 9 later, but 8 is a big jump from 7.
                
    # If size 8 not found, try size 7 to ensure we have a valid 7 (in case random failed)
    # The prompt says current best is 7 (score 2.33).
    # We should ensure we return at least a valid set of size 7 if we fail 8.
    # But the greedy in the previous program found 7.
    # Let's do a quick greedy for 7 if 8 failed.
    
    if not best_result or best_size < 7:
        # Greedy fallback
        chosen = []
        current_sums = set()
        
        # Use a deterministic order: descending
        for x in sorted(pool, reverse=True):
            if _remaining() < 1.0:
                break
            if x in chosen:
                continue
                
            # Check valid addition
            valid = True
            if x in current_sums:
                valid = False
            else:
                for s in current_sums:
                    if (s + x) in current_sums:
                        valid = False
                        break
            
            if not valid:
                continue
                
            new_sums = set(current_sums)
            new_sums.add(x)
            for s in current_sums:
                new_sums.add(s + x)
                
            # Divisibility check
            max_sum = max(new_sums)
            div_free = True
            for s in new_sums:
                multiple = s * 2
                while multiple <= max_sum:
                    if multiple in new_sums:
                        div_free = False
                        break
                    multiple += s
                if not div_free:
                    break
                    
            if div_free:
                chosen.append(x)
                current_sums = new_sums
                
        if len(chosen) > best_size:
            best_result = chosen
            best_size = len(chosen)
            
    # Final Fallback: Hardcoded valid set of size 3 or 4 if all else fails
    # [129, 255] is size 2.
    # Let's try to find a small valid set programmatically if needed.
    if not best_result:
        best_result = [129, 255]
        
    # Ensure types
    return [int(x) for x in best_result]

