"""Program for the suite task: maximize cardinality of S ⊆ [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
def construct():
    import itertools
    
    # We will search for a large subset of [1, 100] with no subset summing to 200.
    # Strategy: Start with a large candidate set (e.g., {50, ..., 100}) and remove elements
    # that participate in many subset sums of 200 (Greedy Hitting Set).
    
    # Let's try different starting intervals to see which yields the largest valid set.
    # The smaller the starting interval, the more elements we have, but the more constraints (smaller subsets can sum to 200).
    # However, if we start too low (e.g., 1..100), the number of constraints is huge.
    # {50..100} allows only 3 and 4 element subsets to sum to 200 (min 5-sum is 260).
    # {40..100} allows 3, 4, and 5 element subsets (min 6-sum is 40+..+45 = 285 > 200? No. 40+41+42+43+44+45 = 255 > 200).
    # So for {40..100}, we need to check subsets of size 3, 4, 5.
    
    # Let's define a function to solve the hitting set problem for a given universe.
    def solve_hitting_set(universe, target=200):
        universe = list(universe)
        if not universe:
            return set()
        
        # Generate all forbidden subsets (subsets of universe that sum to target)
        # To keep it fast, we limit the size of subsets to check.
        # Max size k such that sum of k smallest in universe > target is not needed, 
        # but sum of k smallest < target means we must check up to some K.
        # For universe {50..100}, min 5-sum is 260 > 200. So max size is 4.
        # For universe {40..100}, min 5-sum is 40+41+42+43+44 = 210 > 200. So max size is 4? 
        # Wait. 40+41+42+43+44 = 210. Yes. So max size is 4.
        # For universe {30..100}, min 5-sum is 30+31+32+33+34 = 160 < 200.
        # Min 6-sum is 30+..+35 = 225 > 200. So max size is 5.
        
        # We will try universes {k..100} for k in [30, 50].
        
        forbidden = []
        u_size = len(universe)
        
        # Determine max subset size to check
        # Sort universe
        u_sorted = sorted(universe)
        max_k = 1
        current_sum = 0
        for i in range(u_size):
            current_sum += u_sorted[i]
            if current_sum > target:
                max_k = i
                break
            max_k = i + 1
            
        # Generate combinations
        # To avoid generating too many, we prune.
        # But for k=4 and n=50, C(50,4) is ~230k. Manageable.
        # For k=5 and n=70, C(70,5) is ~12M. Too slow in Python if done naively.
        # We will stick to k=4 or 5 with pruning.
        
        # Pruning: 
        # If sum of first k elements > target, no combination of k elements can sum to target.
        # If sum of last k elements < target, no combination of k elements can sum to target.
        
        forbidden_sets = []
        
        for k in range(3, max_k + 1): # Start from 3, since 1 and 2 are unlikely for high universes
            # Check if any k-combination can sum to target
            min_sum = sum(u_sorted[:k])
            max_sum = sum(u_sorted[-k:])
            if min_sum > target or max_sum < target:
                continue
                
            # Generate combinations with pruning
            # We can use recursive generation or itertools with early termination if possible.
            # For simplicity and speed in Python, we'll use itertools.combinations 
            # but only if the count is reasonable.
            
            # Estimate count
            # If u_size > 60 and k >= 4, it might be slow.
            # Let's just try it. If it's too slow, we'll restrict.
            
            try:
                for comb in itertools.combinations(universe, k):
                    # Pruning inside loop is hard with itertools directly.
                    # But sum(comb) is O(k).
                    if sum(comb) == target:
                        forbidden_sets.append(frozenset(comb))
            except MemoryError:
                # Fallback: break
                break
        
        if not forbidden_sets:
            return set(universe)
        
        # Greedy Hitting Set
        # Count occurrences of each element in forbidden_sets
        from collections import Counter
        
        # We need to iteratively remove elements.
        # To speed up, we can just do a few passes.
        # Actually, a standard greedy is:
        # 1. Count frequency of each element in current forbidden sets.
        # 2. Pick element with max freq.
        # 3. Remove it from universe.
        # 4. Remove all forbidden sets containing it.
        # 5. Repeat.
        
        remaining_universe = set(universe)
        current_forbidden = forbidden_sets
        removed = set()
        
        while current_forbidden:
            # Count frequencies
            counter = Counter()
            for fs in current_forbidden:
                for e in fs:
                    if e in remaining_universe:
                        counter[e] += 1
            
            if not counter:
                break
                
            # Pick best element
            best_elem, max_count = max(counter.items(), key=lambda x: x[1])
            
            # Remove it
            remaining_universe.remove(best_elem)
            removed.add(best_elem)
            
            # Filter forbidden sets
            current_forbidden = [fs for fs in current_forbidden if best_elem not in fs]
            
        return remaining_universe

    best_set = set()
    
    # Try different starting points
    # Start from 30 to 70
    for start in range(30, 71):
        universe = list(range(start, 101))
        valid_set = solve_hitting_set(universe)
        if len(valid_set) > len(best_set):
            best_set = valid_set
            
    # Final check
    # Verify no subset sums to 200
    # Use DP or meet-in-middle for final verification
    def has_subset_sum(target, elements):
        elements = sorted(elements)
        n = len(elements)
        if n <= 20:
            reachable = {0}
            for x in elements:
                new = {s + x for s in reachable if s + x <= target}
                if target in new:
                    return True
                reachable |= new
            return target in reachable
        else:
            half = n // 2
            left = elements[:half]
            right = elements[half:]
            
            left_sums = {0}
            for x in left:
                new = {s + x for s in left_sums if s + x <= target}
                left_sums |= new
            
            right_sums = {0}
            for x in right:
                new = {s + x for s in right_sums if s + x <= target}
                right_sums |= new
            
            for s in left_sums:
                if target - s in right_sums:
                    return True
            return False

    if not has_subset_sum(200, list(best_set)):
        return list(best_set)
    else:
        # Fallback to known valid solution {66..100}
        return list(range(66, 101))
# EVOLVE-BLOCK-END
