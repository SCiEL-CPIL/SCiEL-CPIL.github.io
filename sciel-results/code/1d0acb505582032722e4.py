"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def greedy_sum_free(S, order):
        """Greedy: add elements in given order if it keeps the set sum-free."""
        chosen = []
        chosen_set = set()
        for x in order:
            # Check if adding x violates sum-free property
            valid = True
            if 2 * x in chosen_set:
                valid = False
            else:
                for y in chosen_set:
                    if x + y in chosen_set:
                        valid = False
                        break
                if valid:
                    for y in chosen_set:
                        if x - y in chosen_set:
                            valid = False
                            break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
        return chosen
    
    def heuristic_order(S):
        """Order elements to try first: larger absolute values tend to be harder to form as sums."""
        return sorted(S, key=lambda x: -abs(x))
    
    def try_partition(S):
        """Try to find a large sum-free subset using various heuristics."""
        best = []
        
        # Strategy 1: All odd numbers
        odds = [x for x in S if x % 2 != 0]
        if is_sum_free(odds):
            best = odds
        
        # Strategy 2: Greedy with different orderings
        # Try descending by absolute value
        order1 = heuristic_order(S)
        result1 = greedy_sum_free(S, order1)
        if len(result1) > len(best):
            best = result1
        
        # Try ascending by absolute value
        order2 = sorted(S, key=lambda x: abs(x))
        result2 = greedy_sum_free(S, order2)
        if len(result2) > len(best):
            best = result2
        
        # Try positive numbers first
        order3 = sorted([x for x in S if x > 0] + [x for x in S if x < 0], key=lambda x: -x)
        result3 = greedy_sum_free(S, order3)
        if len(result3) > len(best):
            best = result3
        
        # Try negative numbers first
        order4 = sorted([x for x in S if x < 0] + [x for x in S if x > 0], key=lambda x: -x)
        result4 = greedy_sum_free(S, order4)
        if len(result4) > len(best):
            best = result4
        
        # Strategy 3: Try to maximize by taking elements in upper half
        # For positive numbers, take those > n/2
        # For negative numbers, take those < -n/2 (in magnitude)
        if S:
            max_val = max(S)
            min_val = min(S)
            
            # Upper half positive
            upper_pos = [x for x in S if x > 0 and x > max_val / 2]
            if is_sum_free(upper_pos) and len(upper_pos) > len(best):
                best = upper_pos
            
            # Lower half negative (large magnitude)
            lower_neg = [x for x in S if x < 0 and x < min_val / 2]
            if is_sum_free(lower_neg) and len(lower_neg) > len(best):
                best = lower_neg
        
        return best
    
    def local_search(S, initial, time_limit):
        """Local search: try to improve by swapping elements."""
        current = set(initial)
        current_list = list(initial)
        S_set = set(S)
        
        # Precompute conflicts for efficiency
        def can_add(x, S_current):
            """Check if x can be added to S_current while maintaining sum-free."""
            if 2 * x in S_current:
                return False
            for y in S_current:
                if x + y in S_current:
                    return False
                if x - y in S_current:
                    return False
            return True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(current_list)):
                if time.time() >= time_limit:
                    break
                x = current_list[i]
                # Temporarily remove x
                current.discard(x)
                # Try adding each element not in current
                for y in S:
                    if y in current:
                        continue
                    if can_add(y, current):
                        # Add y, remove x
                        current.add(y)
                        current_list[i] = y
                        improved = True
                        break
                if improved:
                    break
                else:
                    # Put x back
                    current.add(x)
        
        return list(current)
    
    def spectral_bisection_init(S):
        """Use spectral bisection idea to get a good initial partition."""
        n = len(S)
        if n == 0:
            return []
        
        # Build a simple conflict graph
        # Edge between i,j if S[i]+S[j] is in S or S[i]+S[j]=S[k] for some k
        # Actually, let's use a simpler heuristic:
        # Group by sign and magnitude
        
        positives = [x for x in S if x > 0]
        negatives = [x for x in S if x < 0]
        
        # For positives, take upper half
        if positives:
            positives_sorted = sorted(positives)
            mid = len(positives_sorted) // 2
            upper_pos = positives_sorted[mid:]
        else:
            upper_pos = []
        
        # For negatives, take lower half (large magnitude)
        if negatives:
            negatives_sorted = sorted(negatives)
            mid = len(negatives_sorted) // 2
            lower_neg = negatives_sorted[:mid]
        else:
            lower_neg = []
        
        candidate = upper_pos + lower_neg
        
        # Check if sum-free
        if is_sum_free(candidate):
            return candidate
        
        # Fallback to greedy
        return greedy_sum_free(S, sorted(S, key=lambda x: -abs(x)))
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Start with multiple initializations
        candidates = []
        
        # 1. Odd numbers
        odds = [x for x in S if x % 2 != 0]
        if is_sum_free(odds):
            candidates.append(odds)
        
        # 2. Spectral bisection
        spectral = spectral_bisection_init(S)
        if is_sum_free(spectral):
            candidates.append(spectral)
        
        # 3. Try partition heuristics
        partition = try_partition(S)
        if is_sum_free(partition):
            candidates.append(partition)
        
        # Pick the best initial candidate
        best = max(candidates, key=len) if candidates else []
        
        # Local search to improve
        if time.time() < deadline:
            time_per_instance = (deadline - time.time()) / max(1, 3 - len(results))
            improved = local_search(S, best, min(time.time() + time_per_instance, deadline))
            if is_sum_free(improved) and len(improved) > len(best):
                best = improved
        
        results.append(best)
        
        if time.time() > deadline:
            # Return what we have for remaining instances using just odds
            for s, sd in ((60, 801), (120, 802), (240, 803))[len(results):]:
                S_rem = instance(s, sd)
                odds_rem = [x for x in S_rem if x % 2 != 0]
                results.append(odds_rem)
            break
    
    return results
# EVOLVE-BLOCK-END
