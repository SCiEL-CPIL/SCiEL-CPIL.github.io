import math
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def is_valid(arr):
        """
        Checks if arr is a valid Sidon set covering 1..30.
        Returns (is_valid, diffs_set, diameter).
        """
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None

        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)

        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def find_min_diameter_set(max_candidate=80):
        """
        Uses backtracking search with forward checking to find a Sidon set
        with minimal maximum element.
        
        We search for a subset S of {0, 1, ..., max_candidate} such that:
        1. 0 in S
        2. S is a Sidon set (all pairwise differences are distinct)
        3. All d in 1..30 are differences in S
        4. Max(S) is minimized
        
        Strategy:
        - Iterate through potential max elements D from lower bound upwards.
        - For each D, perform backtracking to see if a valid set exists with max <= D.
        - Backtracking state:
          - current_set: list of chosen elements (sorted)
          - diffs: set of differences so far
          - candidates: remaining elements to consider (from current_set[-1]+1 to D)
          - required_diffs: set of differences 1..30 not yet covered
        
        Optimization:
        - If the number of remaining candidates is insufficient to cover the 
          remaining required differences, prune.
          (Max new diffs from k elements added to set of size m is k*(m+k-1)/2)
        - Forward checking: When adding a candidate x, check if it creates any 
          duplicate difference with existing elements.
        """
        
        # Lower bound on max element:
        # To cover 1..30, we need at least 30 distinct differences.
        # A set of size k has k(k-1)/2 differences.
        # k=8 -> 28 diffs (not enough)
        # k=9 -> 36 diffs (enough)
        # So we need at least 9 elements including 0.
        # The minimal diameter for a Sidon set of size 9 covering 1..30?
        # Let's just start from a reasonable lower bound.
        # If we have elements 0, a1, ..., a8, max is a8.
        # Differences include a1, a2-0, a2-a1, etc.
        # A known small Sidon set covering 1..N often has diameter ~ N * sqrt(N) or similar?
        # Let's start from D=40 and go up.
        
        best_set = None
        
        # We will try to find a set with max element exactly D.
        # Actually, it's easier to search for any set with max <= D.
        # We iterate D from low to high.
        
        # Heuristic lower bound:
        # To get difference 30, we need an element >= 30 (since 0 is in set).
        # So D >= 30.
        # But we need 30 distinct differences.
        # Let's start D from 35.
        
        for D in range(35, max_candidate + 1):
            if time.time() > deadline - 5:
                break
            
            # Backtracking search for a valid set with max <= D
            # We fix 0 in the set.
            # We want to select a subset of {1, ..., D} to add to {0}.
            
            # State for backtracking:
            # current_elements: sorted list, starts with [0]
            # diffs: set of differences
            # covered: set of differences in 1..30 that are in diffs
            # candidates: list of integers from current_elements[-1]+1 to D
            
            # To improve search, we can order candidates.
            # Candidates that cover more "missing" differences first?
            # Or just try all?
            
            found = None
            
            def backtrack(current_elements, diffs, covered, start_idx, candidates_list):
                nonlocal found
                if found is not None:
                    return
                
                if time.time() > deadline - 5:
                    return
                
                # Check if goal met
                if len(covered) == 30:
                    found = current_elements[:]
                    return
                
                # Pruning:
                # 1. Can we cover the rest?
                # Missing diffs: 30 - len(covered)
                # Remaining candidates count: len(candidates_list) - start_idx
                # Let k be number of elements we can still add.
                # Max new differences from adding k elements to a set of size m:
                # Each new element x creates diffs (x - y) for y in current_elements.
                # Also diffs between new elements.
                # This is complex. Simple bound:
                # If we add all remaining candidates, how many diffs do we get?
                # Total elements would be m + k.
                # Total diffs = (m+k)(m+k-1)/2.
                # Current diffs = m(m-1)/2.
                # New diffs = (m+k)(m+k-1)/2 - m(m-1)/2.
                # This must be >= missing_diffs.
                
                m = len(current_elements)
                k = len(candidates_list) - start_idx
                missing = 30 - len(covered)
                
                # Calculate max possible new differences
                # If we add all k candidates:
                new_total_elements = m + k
                max_possible_diffs = new_total_elements * (new_total_elements - 1) // 2
                current_diffs_count = m * (m - 1) // 2
                max_new_diffs = max_possible_diffs - current_diffs_count
                
                if max_new_diffs < missing:
                    return
                
                # 2. Try adding candidates
                for i in range(start_idx, len(candidates_list)):
                    c = candidates_list[i]
                    
                    # Check if adding c is valid (no duplicate diffs)
                    new_diffs = []
                    ok = True
                    for x in current_elements:
                        d = c - x
                        if d in diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    # Check for conflicts among new_diffs themselves?
                    # No, because c is the largest element, so all new diffs are c-x.
                    # These are distinct if x are distinct.
                    # So we just need to check against existing diffs.
                    
                    # Update state
                    new_covered = set(covered)
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_covered.add(d)
                    
                    new_diffs_set = diffs.union(new_diffs)
                    new_elements = current_elements + [c]
                    
                    # Recurse
                    backtrack(new_elements, new_diffs_set, new_covered, i + 1, candidates_list)
                    
                    if found is not None:
                        return
            
            # Initial call
            # Candidates are 1..D
            # We can optimize candidate ordering.
            # For now, standard order.
            # To find minimal max element, we are iterating D.
            # Within a fixed D, we just need *any* solution.
            
            # Heuristic: Try candidates that are "useful" first?
            # Let's just use standard order for simplicity and correctness first.
            # If it's too slow, we can randomize or sort by potential.
            
            candidates = list(range(1, D + 1))
            
            # Optimization: If we don't cover all diffs, we might need more elements.
            # But we are limited by D.
            
            backtrack([0], set(), set(), 0, candidates)
            
            if found is not None:
                return found
                
        return None

    def local_search_refine(initial_set, time_limit):
        """
        Quick local search to potentially reduce diameter further if backtracking
        found a solution but not the absolute minimum, or if backtracking failed
        to find the minimal one within time.
        """
        if time_limit <= 0:
            return initial_set
            
        current = list(initial_set)
        current.sort()
        valid, _, cur_diam = is_valid(current)
        if not valid:
            return initial_set
            
        best_set = current[:]
        best_diam = cur_diam
        
        T = 10.0
        T_min = 0.5
        cooling = 0.99
        
        while time.time() < time_limit:
            if T < T_min:
                T = T_min
                
            neighbor = current[:]
            move_type = random.random()
            
            if move_type < 0.6:
                # Replace an element
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    max_val = max(neighbor)
                    # Try to shrink
                    if random.random() < 0.5:
                        new_val = random.randint(1, max_val)
                    else:
                        new_val = random.randint(1, max_val + 5)
                        
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:]
                        
            elif move_type < 0.8:
                # Remove
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    neighbor.pop(idx)
                    neighbor.sort()
                    
            else:
                # Add
                if len(neighbor) < 32:
                    max_val = max(neighbor)
                    new_val = random.randint(1, max_val + 5)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        neighbor = current[:]
                        
            if len(neighbor) > 32 or 0 not in neighbor:
                continue
            if max(neighbor) > 10**6:
                continue
                
            # Fast Sidon check
            diffs = set()
            n = len(neighbor)
            valid_sidon = True
            for i in range(n):
                for j in range(i + 1, n):
                    d = neighbor[j] - neighbor[i]
                    if d in diffs:
                        valid_sidon = False
                        break
                    diffs.add(d)
                if not valid_sidon:
                    break
                    
            if not valid_sidon:
                continue
                
            # Check coverage
            if not all(d in diffs for d in range(1, 31)):
                continue
                
            new_diam = max(neighbor) - min(neighbor)
            delta = new_diam - best_diam
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current = neighbor
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_set = neighbor[:]
                    
            T *= cooling
            
        return best_set

    # Main Execution
    
    # 1. Try to find the minimal diameter set using Backtracking
    # We search for max element D from 35 upwards.
    # This is exact for small D.
    best_set = find_min_diameter_set(max_candidate=60)
    
    if best_set is None:
        # If backtracking didn't find anything (maybe too slow or no solution in range),
        # fall back to a heuristic construction
        print("Backtracking failed, using heuristic", file=sys.stderr)
        
        # Simple greedy
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            if valid:
                for y in a:
                    if x > y:
                        diffs.add(x - y)
                a.append(x)
            x += 1
        best_set = a
    else:
        # 2. Refine with local search if time permits
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5:
            refined = local_search_refine(best_set, deadline - 5)
            if refined:
                v, _, diam_ref = is_valid(refined)
                v_orig, _, diam_orig = is_valid(best_set)
                if v and diam_ref < diam_orig:
                    best_set = refined
                    print(f"Refined diameter from {diam_orig} to {diam_ref}", file=sys.stderr)
    
    # Final validation
    if best_set is None:
        # Emergency fallback
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                if x > y:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
            if valid:
                for y in a:
                    if x > y:
                        diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    return best_set

# EVOLVE-BLOCK-END
