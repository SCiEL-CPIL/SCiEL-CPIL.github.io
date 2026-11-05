import sys
import time

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with the minimum possible largest element.
    Uses a custom branch-and-bound solver with bitmask optimization.
    """
    required_diffs_mask = 0
    for d in range(1, 31):
        required_diffs_mask |= (1 << d)
    
    # We need to find the minimum D.
    # Lower bound for D:
    # If size is k, max differences is k*(k-1)/2.
    # We need at least 30 differences.
    # k=9 -> 36 diffs.
    # k=10 -> 45 diffs.
    # The differences must be distinct and <= D.
    # So D must be at least the 30th smallest possible difference in a Sidon set?
    # Actually, just D >= 30 is the trivial bound.
    # A tighter bound: The sum of differences? No.
    # Let's start from D=30.
    
    def solve_for_D(D):
        """
        Branch and bound search for a Sidon set in [0, D] covering 1..30.
        Returns the set if found, else None.
        """
        # We try sizes k from 9 upwards.
        # Max k is such that k*(k-1)/2 <= D (since diffs are distinct and <= D).
        # Actually, diffs are in 1..D, so there are D possible diffs.
        # So k*(k-1)/2 <= D is a necessary condition.
        max_k = int((1 + (1 + 8 * D) ** 0.5) / 2)
        
        for k in range(9, max_k + 1):
            result = _bbsolve(D, k, required_diffs_mask)
            if result is not None:
                return result
        return None

    def _bbsolve(D, k, req_mask):
        """
        Recursively builds the set.
        state:
        - current_set: list of chosen elements
        - used_diffs_mask: bitmask of all differences generated so far
        - covered_mask: bitmask of required diffs covered so far
        """
        current_set = [0]
        used_diffs_mask = 0
        covered_mask = 0
        
        # We need to choose k-1 more elements from 1..D.
        
        def backtrack(idx, last_elem, curr_set, used_mask, cov_mask):
            if idx == k:
                if (cov_mask & req_mask) == req_mask:
                    return curr_set[:]
                return None
            
            # Pruning 1: Can we cover the rest?
            remaining_req = req_mask & ~cov_mask
            if remaining_req == 0:
                # We already covered everything. We just need to fill the rest?
                # No, the size must be exactly k?
                # The problem says "at most 32 distinct integers".
                # It doesn't say exactly k.
                # However, smaller k is generally easier to find?
                # Actually, we are iterating k. If we find a solution for k, we return it.
                # If we covered everything before reaching size k, we could just return curr_set?
                # But we are trying to find *a* set of size k?
                # The task says "at most 32". It doesn't require minimal size.
                # But minimizing D is the goal.
                # If a smaller k works with smaller D, we would have found it in a previous k iteration?
                # No, we iterate k in a fixed D loop.
                # If k=9 works for D=40, we return it.
                # If k=9 doesn't work, we try k=10.
                # If cov_mask covers req_mask, we can return curr_set immediately?
                # Yes, because we want *any* valid set. Smaller sets are fine.
                # But wait, if we return early, we might miss a solution with smaller max element?
                # No, we are fixing D. We just need *some* set in [0,D].
                # So if we have a valid set, we can return it.
                return curr_set[:]

            # Pruning 2: Max possible new differences
            m = k - idx
            # Current size is idx.
            # New elements will form differences with idx existing elements and each other.
            max_new_diffs = m * idx + m * (m - 1) // 2
            if bin(remaining_req).count('1') > max_new_diffs:
                return None

            # Pruning 3: Range check
            # We need m elements in [last_elem + 1, D].
            if D - last_elem < m:
                return None

            # Heuristic: Try elements that cover more required diffs first?
            # Or just iterate?
            # To speed up, we can restrict the range of x.
            # x must be such that x - curr_set[i] doesn't collide with used_mask.
            
            # Optimization: Generate candidate x's.
            # x ranges from last_elem + 1 to D.
            # We can also prune x if x - curr_set[0] > D (always true since x<=D).
            
            # Let's try to be smart about the next element.
            # If there is a required difference r that is very large, 
            # we might need a large x.
            
            for x in range(last_elem + 1, D + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for a in curr_set:
                    d = x - a
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check for collisions among new_diffs?
                # new_diffs are x - a_i. Since a_i are distinct, new_diffs are distinct.
                # So no collision within new_diffs.
                
                new_used_mask = used_mask
                new_cov_mask = cov_mask
                for d in new_diffs:
                    bit = 1 << d
                    new_used_mask |= bit
                    new_cov_mask |= bit
                
                # Recurse
                res = backtrack(idx + 1, x, curr_set + [x], new_used_mask, new_cov_mask)
                if res is not None:
                    return res
            
            return None
        
        return backtrack(1, 0, [0], 0, 0)

    start_time = time.time()
    timeout = 110
    
    # Iterate D
    # Lower bound:
    # For k=9, max diff D. 36 diffs.
    # The differences must be distinct.
    # The smallest possible set of differences for a Sidon set?
    # Let's just start from 30.
    
    for D in range(30, 1000):
        if time.time() - start_time > timeout:
            return greedy_construct()
        
        result = solve_for_D(D)
        if result is not None:
            return sorted(result)
    
    return greedy_construct()

def greedy_construct():
    """Fallback greedy constructor."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)
