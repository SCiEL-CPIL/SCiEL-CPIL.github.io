import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    def is_valid_addition(chosen, x, sums_by_card):
        """Check if adding x to chosen keeps cardinality sums disjoint."""
        new_sums = [set(s) for s in sums_by_card]
        for k in range(len(chosen), -1, -1):
            for s in sums_by_card[k]:
                new_sums[k+1].add(s + x)
        # Check disjointness across all cardinalities
        for k in range(len(new_sums)):
            for j in range(k+1, len(new_sums)):
                if new_sums[k] & new_sums[j]:
                    return False, None
        return True, new_sums
    
    def try_greedy(order, deadline):
        """Try greedy addition in given order, return best valid set found."""
        sums_by_card = [set() for _ in range(61)]
        sums_by_card[0].add(0)
        chosen = []
        
        for x in order:
            if time.time() > deadline:
                break
            if x in chosen:
                continue
            valid, new_sums = is_valid_addition(chosen, x, sums_by_card)
            if valid:
                chosen.append(x)
                sums_by_card = new_sums
                if len(chosen) >= 60:
                    break
        return chosen
    
    def try_greedy_with_backtrack(order, deadline):
        """Greedy with occasional backtracking to try alternatives."""
        sums_by_card = [set() for _ in range(61)]
        sums_by_card[0].add(0)
        chosen = []
        best_size = 0
        
        for x in order:
            if time.time() > deadline:
                break
            if x in chosen:
                continue
            valid, new_sums = is_valid_addition(chosen, x, sums_by_card)
            if valid:
                chosen.append(x)
                sums_by_card = new_sums
                best_size = max(best_size, len(chosen))
                if len(chosen) >= 60:
                    break
        return chosen
    
    # Strategy 1: Descending order (largest first)
    result1 = try_greedy(range(256, 0, -1), deadline)
    
    # Strategy 2: Ascending order (smallest first)
    result2 = try_greedy(range(1, 257), deadline)
    
    # Strategy 3: Descending with step (skip some to allow more elements)
    result3 = try_greedy(range(256, 0, -2), deadline)
    
    # Strategy 4: Randomized attempts
    best_result = max([result1, result2, result3], key=len)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        random.seed(attempt)
        # Mix of large and medium values
        pool = list(range(128, 257)) + list(range(1, 64))
        random.shuffle(pool)
        result = try_greedy(pool, deadline)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 5: Try to extend best_result by trying remaining values
    if len(best_result) < 60:
        used = set(best_result)
        candidates = [x for x in range(256, 0, -1) if x not in used]
        sums_by_card = [set() for _ in range(61)]
        sums_by_card[0].add(0)
        # Rebuild sums
        for x in best_result:
            for k in range(len(best_result), -1, -1):
                for s in list(sums_by_card[k]):
                    sums_by_card[k+1].add(s + x)
        # Actually rebuild properly
        sums_by_card = [set() for _ in range(61)]
        sums_by_card[0].add(0)
        for x in best_result:
            for k in range(len(best_result), -1, -1):
                for s in sums_by_card[k]:
                    sums_by_card[k+1].add(s + x)
        
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_sums = is_valid_addition(best_result, x, sums_by_card)
            if valid:
                best_result.append(x)
                sums_by_card = new_sums
                if len(best_result) >= 60:
                    break
    
    # Ensure distinct and in range
    best_result = list(set(best_result))
    best_result = [x for x in best_result if 1 <= x <= 256]
    best_result.sort()
    
    # Final validation
    if len(best_result) == 0:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Verify
    sums_by_card = [set() for _ in range(len(best_result)+1)]
    sums_by_card[0].add(0)
    for x in best_result:
        for k in range(len(best_result), -1, -1):
            for s in sums_by_card[k]:
                sums_by_card[k+1].add(s + x)
    for k in range(len(sums_by_card)):
        for j in range(k+1, len(sums_by_card)):
            if sums_by_card[k] & sums_by_card[j]:
                # Invalid, fall back to known good
                return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best_result
