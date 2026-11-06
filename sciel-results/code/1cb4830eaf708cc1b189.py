"""Rewritten program: DFS-based search for minimal-diameter Sidon set covering 1..30."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    best_result = None
    best_max = float('inf')
    
    # We need a set containing 0, with all positive differences distinct,
    # and every d in 1..30 appearing as a difference.
    # Minimize the largest element.
    
    # Lower bound: we need at least enough elements to cover 30 distinct differences.
    # A set of size n has n*(n-1)/2 positive differences.
    # n*(n-1)/2 >= 30 => n >= 8 (8*7/2=28 < 30, so n>=9; 9*8/2=36>=30)
    # But we also need 0 in the set, and differences are between pairs.
    # Actually with 0 in the set, if we have elements {0, a1, a2, ..., a_{n-1}},
    # differences include ai - 0 = ai for each ai, and ai - aj for i>j.
    # Total differences = n*(n-1)/2 where n is total count including 0.
    # Need n*(n-1)/2 >= 30, so n >= 9.
    
    # Let's use DFS to build the set incrementally.
    # State: current set (sorted list), set of used differences, set of required diffs remaining.
    # We try to add elements in increasing order to keep the span small.
    
    # Strategy: DFS where we build the set by choosing the next element.
    # We know 0 is in the set. We'll build up the set.
    # At each step, we can either:
    #   1. Add a new element that is the sum of an existing element and a needed difference
    #   2. Try small integers to see if they work
    
    # Better approach: Since we want to minimize the max element, we can do a BFS/DFS
    # where we try to find a set with max element <= M, increasing M.
    
    # Actually, let's use a more direct DFS:
    # - Start with {0}
    # - The set of required differences is {1, 2, ..., 30}
    # - At each step, pick an uncovered difference d, and try to realize it by adding
    #   a new element x such that x - a = d or a - x = d for some a in the set,
    #   or by having two existing elements whose difference is d.
    
    # More practical DFS:
    # Build the set element by element. At each step, try candidate values for the next element.
    # Candidates should be values that help cover remaining differences.
    
    # Let me think about this differently. A Sidon set (B2 set) has all pairwise sums distinct
    # (equivalently, all positive differences distinct). 
    
    # Key insight: if 0 is in the set S, then for each s in S\{0}, the difference s-0 = s is
    # a difference. So the elements of S\{0} themselves are differences. This means if we need
    # difference d, and d < max(S), then d must either be an element of S or equal to a_i - a_j
    # for some pair.
    
    # Let's do DFS with the following approach:
    # - Maintain sorted list of elements, starting with [0]
    # - Maintain set of all differences produced so far
    # - Maintain set of required differences not yet covered
    # - At each step, if all required differences are covered, we have a solution
    # - Otherwise, try adding a new element. The new element must be larger than all existing
    #   (to keep things simple, we build in increasing order)
    # - Candidate values for new element: for each uncovered difference d, and each existing
    #   element a, candidate = a + d (if positive and not already in set)
    # - Also try small values to fill gaps
    
    # To keep the search tractable, we'll limit the number of elements (at most 32) and
    # use pruning.
    
    def dfs(elements, used_diffs, remaining, depth, max_allowed):
        """
        elements: sorted list of elements in the set (includes 0)
        used_diffs: set of all positive differences produced
        remaining: set of differences from 1..30 not yet covered
        depth: number of elements added so far
        max_allowed: upper bound on the largest element
        Returns: (success, solution) where solution is the full set if found
        """
        nonlocal best_result, best_max, deadline
        
        if time.time() > deadline:
            return False, None
        
        if not remaining:
            # All differences covered!
            sol_max = max(elements)
            if sol_max < best_max:
                best_max = sol_max
                best_result = list(elements)
            return True, list(elements)
        
        if len(elements) >= 32:
            return False, None
        
        if max(elements) >= max_allowed:
            return False, None
        
        # Generate candidates for the next element
        # Candidates: for each remaining diff d and each element a in elements,
        # candidate = a + d (must be > max(elements) to maintain sorted order, or we allow any new value)
        # Actually, we should allow candidates that are greater than current max to keep sorted.
        # But we could also insert in the middle... that complicates things.
        # Let's just add elements in increasing order.
        
        current_max = elements[-1]
        
        candidates = set()
        for d in remaining:
            for a in elements:
                c = a + d
                if c > current_max and c <= max_allowed:
                    candidates.add(c)
        
        # Also try current_max + 1, current_max + 2, etc. up to some small range
        # to allow for elements that don't directly correspond to a remaining diff
        for offset in range(1, 15):
            c = current_max + offset
            if c <= max_allowed:
                candidates.add(c)
        
        # Sort candidates to try smaller ones first (minimizes span)
        candidates = sorted(candidates)
        
        # Limit candidates to keep search tractable
        if len(candidates) > 50:
            # Prioritize candidates that cover more remaining differences
            def coverage_score(c):
                new_diffs = set()
                for a in elements:
                    diff = abs(c - a)
                    if diff > 0:
                        new_diffs.add(diff)
                return len(new_diffs & remaining)
            
            candidates.sort(key=lambda c: (-coverage_score(c), c))
            candidates = candidates[:50]
        
        for c in candidates:
            if time.time() > deadline:
                return False, None
            
            # Check if adding c maintains Sidon property
            valid = True
            new_elements = elements + [c]
            new_used_diffs = set(used_diffs)
            
            for i, a in enumerate(elements):
                diff = c - a  # c > a since c > current_max >= a
                if diff in new_used_diffs:
                    valid = False
                    break
                new_used_diffs.add(diff)
            
            if not valid:
                continue
            
            # Check which remaining differences are now covered
            new_remaining = set(remaining)
            for diff in list(new_remaining):
                if diff in new_used_diffs:
                    new_remaining.discard(diff)
            
            # Also check if any difference between existing elements equals a remaining diff
            # (this should already be handled since used_diffs includes all pairwise diffs)
            
            success, sol = dfs(new_elements, new_used_diffs, new_remaining, depth + 1, max_allowed)
            if success:
                return True, sol
        
        return False, None
    
    # Try increasing max_allowed values
    # Lower bound on max element: we need at least 9 elements (including 0), 
    # and the differences 1..30 must all appear.
    # The smallest possible max element is at least 30 (since 30 must be a difference,
    # and if 0 is in the set, either 30 is in the set or 30 = a_i - a_j).
    # Actually, if 30 = a_i - a_j with a_j > 0, then a_i > 30. If 30 is in the set, max >= 30.
    # So max >= 30.
    
    # Let's start from a reasonable lower bound and increase
    # For n elements in [0, M], max differences = n*(n-1)/2
    # We need 30 differences, so n >= 9
    # With 9 elements in [0, M], we can have at most 36 differences
    # The minimum M for a Sidon set of size 9 covering 1..30...
    
    # Let's try M from 30 upward
    for max_allowed in range(30, 200):
        if time.time() > deadline:
            break
        
        # Reset for each M
        initial_elements = [0]
        initial_diffs = set()
        initial_remaining = set(range(1, 31))
        
        success, sol = dfs(initial_elements, initial_diffs, initial_remaining, 1, max_allowed)
        
        if success:
            if best_result is not None:
                break
            # Found a solution with this max_allowed, try to find better
            # Actually, since we're increasing max_allowed, the first success is the minimum
            # But DFS might not find the optimal within this M, so let's continue a bit
            # Actually, if we found a solution with max = max_allowed, we can stop
            # because we're searching in order of increasing max_allowed
            break
    
    # If no solution found, use a fallback
    if best_result is None:
        # Fallback: greedy construction
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
        best_result = a
    
    return best_result

# Let's also try a randomized restart approach if the initial DFS doesn't find a good solution
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    best_result = None
    best_max = float('inf')
    
    def try_dfs(max_allowed, time_limit=10):
        """Try DFS with given max_allowed, return solution or None."""
        nonlocal best_result, best_max
        dfs_start = time.time()
        
        # Use iterative deepening or just direct DFS
        # State: (elements_tuple, used_diffs_frozenset, remaining_frozenset)
        # This is expensive, so let's use lists and sets but with memoization on (tuple(elements), frozenset(remaining))
        
        memo = set()
        
        def dfs(elements, used_diffs, remaining):
            nonlocal best_result, best_max
            
            if time.time() - dfs_start > time_limit:
                return False, None
            
            if time.time() > deadline:
                return False, None
            
            if not remaining:
                sol_max = max(elements)
                if sol_max < best_max:
                    best_max = sol_max
                    best_result = list(elements)
                return True, list(elements)
            
            if len(elements) >= 32:
                return False, None
            
            current_max = elements[-1]
            
            if current_max >= max_allowed:
                return False, None
            
            # Memoization key
            key = (tuple(elements), frozenset(remaining))
            if key in memo:
                return False, None
            memo.add(key)
            
            # Generate candidates
            candidates = set()
            for d in remaining:
                for a in elements:
                    c = a + d
                    if c > current_max and c <= max_allowed:
                        candidates.add(c)
            
            # Also try small increments
            for offset in range(1, 10):
                c = current_max + offset
                if c <= max_allowed:
                    candidates.add(c)
            
            # Score and limit candidates
            def coverage_score(c):
                count = 0
                for a in elements:
                    diff = c - a
                    if diff > 0 and diff in remaining:
                        count += 1
                return count
            
            candidates = sorted(candidates, key=lambda c: (-coverage_score(c), c))
            candidates = candidates[:40]  # Limit for tractability
            
            for c in candidates:
                if time.time() - dfs_start > time_limit:
                    return False, None
                
                # Check Sidon property
                valid = True
                new_used_diffs = set(used_diffs)
                for a in elements:
                    diff = c - a
                    if diff in new_used_diffs:
                        valid = False
                        break
                    new_used_diffs.add(diff)
                
                if not valid:
                    continue
                
                new_remaining = remaining - new_used_diffs
                
                new_elements = elements + [c]
                success, sol = dfs(new_elements, new_used_diffs, new_remaining)
                if success:
                    return True, sol
            
            return False, None
        
        initial_elements = [0]
        initial_diffs = set()
        initial_remaining = frozenset(range(1, 31))
        
        success, sol = dfs(initial_elements, initial_diffs, initial_remaining)
        return sol
    
    # Try increasing max_allowed
    for max_allowed in range(30, 150):
        if time.time() > deadline:
            break
        
        sol = try_dfs(max_allowed, time_limit=5)
        if sol is not None:
            if best_result is None or max(sol) < max(best_result):
                best_result = sol
                best_max = max(sol)
            # Found a solution, we can stop since we're increasing max_allowed
            break
    
    # If no solution found, try with larger bounds and more time
    if best_result is None:
        for max_allowed in range(150, 500):
            if time.time() > deadline:
                break
            sol = try_dfs(max_allowed, time_limit=5)
            if sol is not None:
                best_result = sol
                best_max = max(sol)
                break
    
    # Fallback if still no solution
    if best_result is None:
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
        best_result = a
    
    return best_result
