import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # We want a Sidon set containing 0, with all differences 1..30 present.
    # The set must be a Golomb ruler (all pairwise differences distinct).
    # We search for the minimum diameter D such that a Golomb ruler of some size k
    # within [0, D] covers all differences 1..30.
    #
    # Key insight: A Golomb ruler of size k has k*(k-1)/2 differences.
    # We need at least 30 differences, so k*(k-1)/2 >= 30 => k >= 8 (8*7/2=28, not enough)
    # k=9 gives 36 differences, which is enough.
    # But we also need ALL of 1..30 to be present, not just 30 distinct ones.
    #
    # Strategy: DFS to build a Golomb ruler incrementally, tracking covered differences.
    # For each candidate diameter D (starting from lower bound), try to find a valid ruler.
    #
    # Lower bound on D: A Golomb ruler of size k with diameter D has max difference D.
    # The minimum possible D for a Golomb ruler of size k is known for small k.
    # For k=9, the optimal Golomb ruler has diameter 42: {0, 1, 4, 10, 12, 17, 20, 24, 42}
    # But we need to check if it covers 1..30. Let's just search.
    
    # First, let's verify the lower bound more carefully.
    # With k elements, we have C(k,2) differences. We need all of 1..30.
    # So C(k,2) >= 30 => k >= 8 (28 < 30, no), k=9 (36 >= 30, yes).
    # So minimum k is 9.
    
    # The diameter must be at least 30 (to have difference 30).
    # Also, for a Golomb ruler of size 9, minimum diameter is 42 (known result).
    # Let's start searching from D=30 upward, but we can start from a tighter bound.
    # Actually, let's just do a BFS/DFS search over possible rulers.
    
    # DFS approach:
    # State: (current_elements, used_differences, current_max)
    # We build the ruler by adding elements from left to right (sorted).
    # The first element is always 0.
    # We try to add elements one by one, ensuring all pairwise differences are unique
    # and tracking which of 1..30 are covered.
    
    def is_valid_extension(elements, new_elem):
        """Check if adding new_elem to elements keeps it a Golomb ruler."""
        for x in elements:
            d = new_elem - x
            if d <= 0:
                continue
            # Check if d is already a difference
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    if elements[j] - elements[i] == d:
                        return False
        return True
    
    # More efficient: maintain a set of used differences
    def can_add(elements, used_diffs, new_elem):
        """Check if new_elem can be added without creating duplicate differences."""
        for x in elements:
            d = new_elem - x
            if d < 0:
                continue
            if d == 0:
                return False
            if d in used_diffs:
                return False
        return True
    
    # DFS to find a Golomb ruler of size k within [0, D] covering all 1..30
    # We use a recursive search with pruning.
    
    full_mask = (1 << 31) - 2  # bits 1 through 30 set
    
    def dfs(elements, used_diffs, covered_mask, depth, max_diameter, target_k):
        """
        elements: sorted list of current ruler elements (always starts with 0)
        used_diffs: set of all pairwise differences in current ruler
        covered_mask: bitmask of which differences in 1..30 are covered
        depth: current number of elements
        max_diameter: maximum allowed value
        target_k: target number of elements
        Returns a valid ruler or None.
        """
        if time.time() > deadline:
            return None
        
        if depth == target_k:
            if covered_mask == full_mask:
                return list(elements)
            return None
        
        # Pruning: can we still cover all remaining differences?
        remaining_to_cover = 30 - bin(covered_mask & full_mask).count('1')
        remaining_elements = target_k - depth
        # Max new differences from adding remaining_elements: 
        # each new element adds (depth + i) new differences for i-th new element
        # Upper bound: remaining_elements * depth + remaining_elements*(remaining_elements-1)//2
        max_new_diffs = remaining_elements * depth + remaining_elements * (remaining_elements - 1) // 2
        if max_new_diffs < remaining_to_cover:
            return None
        
        last_elem = elements[-1] if elements else 0
        
        # Try adding next element
        # The next element must be > last_elem and <= max_diameter
        # Also, the difference from 0 (i.e., the element itself) must not be in used_diffs
        # and must not create any duplicate differences with existing elements
        
        # We can bound the search: the next element x must satisfy:
        # x - 0 = x not in used_diffs
        # x - e not in used_diffs for all e in elements
        # x <= max_diameter
        
        start = last_elem + 1
        end = min(max_diameter, start + 500)  # Limit search width for speed
        
        for x in range(start, end + 1):
            if time.time() > deadline:
                return None
            
            # Check if x can be added
            new_diffs = []
            valid = True
            for e in elements:
                d = x - e
                if d <= 0:
                    continue
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            new_used = used_diffs | set(new_diffs)
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << d)
            
            result = dfs(elements + [x], new_used, new_mask, depth + 1, max_diameter, target_k)
            if result is not None:
                return result
        
        return None
    
    # Try different target sizes and diameters
    # Minimum size is 9 (since C(8,2)=28 < 30)
    # Try k=9 first with increasing diameter
    
    for target_k in range(9, 13):  # Try sizes 9 to 12
        # Lower bound on diameter: at least 30
        # For Golomb rulers, known lower bounds exist but let's just search
        for D in range(30, 200):
            if time.time() > deadline:
                break
            
            # Quick prune: C(target_k, 2) must be >= 30
            if target_k * (target_k - 1) // 2 < 30:
                continue
            
            result = dfs([0], {0}, 0, 1, D, target_k)
            if result is not None:
                return result
        
        if time.time() > deadline:
            break
    
    # Fallback: use the greedy approach from the seed
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
