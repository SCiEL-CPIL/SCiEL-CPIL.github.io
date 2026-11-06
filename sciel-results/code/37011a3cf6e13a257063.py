import sys
import time

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30, 
    with the minimum possible largest element.
    """
    required_diffs = set(range(1, 31))
    
    # Minimum size k such that k*(k-1)/2 >= 30
    # k=8 -> 28 < 30
    # k=9 -> 36 >= 30
    min_k = 9
    
    # Lower bound for D:
    # If k=9, min D is 36.
    # If k=10, min D is 45.
    # We can start checking D from 36.
    
    def find_set_for_D(D, k):
        """
        Tries to find a Sidon set of size k in [0, D] covering 1..30.
        Returns the set if found, else None.
        """
        # State
        A = [0]
        diffs = set()
        covered = set()
        
        # We fix the last element to be D to enforce minimal diameter usage.
        # We need to choose k-2 elements from [1, D-1].
        
        def solve(idx, current_max, remaining_slots):
            # idx: number of elements chosen so far (excluding 0 and potentially D)
            # current_max: last chosen element
            # remaining_slots: how many more elements to choose (excluding D)
            
            if remaining_slots == 0:
                # We have chosen k-1 elements (including 0). We must add D.
                # Check if adding D is valid and covers all diffs.
                if D in diffs:
                    return None
                # Check if D - a is in diffs for all a in A
                new_diffs = []
                valid = True
                for a in A:
                    d = D - a
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    return None
                
                # Check coverage
                all_diffs = diffs.union(new_diffs)
                if required_diffs.issubset(all_diffs):
                    return A + [D]
                return None

            # Pruning 1: Can we cover all remaining required diffs?
            remaining_req = required_diffs - covered
            # Max new diffs we can add:
            # We will add 'remaining_slots' elements and then D.
            # Let m = remaining_slots + 1 (including D).
            # Current size is len(A).
            # New diffs from m elements with len(A) existing elements: m * len(A) + m*(m-1)/2.
            m = remaining_slots + 1
            max_new_diffs = m * len(A) + m * (m - 1) // 2
            if len(remaining_req) > max_new_diffs:
                return None

            # Pruning 2: Can we fit the elements?
            # We need to place 'remaining_slots' elements in (current_max, D).
            # So we need D - current_max - 1 >= remaining_slots.
            if D - current_max - 1 < remaining_slots:
                return None

            # Pruning 3: Specific diff feasibility
            # The largest required diff not yet covered must be <= D.
            # Also, if a large diff r is required, it must be formed by D - a_i or a_j - a_i.
            # If r > D, impossible. (Already guaranteed by D >= 30).
            # If r is not in diffs, it must be formed by future elements.
            # The max diff formable with future elements (including D) is D - 0 = D.
            # This is always true.
            
            # Heuristic: Try values that help cover the largest uncovered diffs first?
            # Or just iterate?
            # Let's iterate x from current_max + 1 to D - remaining_slots.
            # We must leave room for the remaining_slots - 1 elements and D.
            # So x <= D - remaining_slots.
            
            # Optimization: If the smallest uncovered diff is small, and we pick x too large,
            # we might miss covering small diffs? No, small diffs are covered by close elements.
            
            # Let's use a simple heuristic: iterate x.
            # To break symmetry and speed up, we can note that if x is small, it creates small diffs.
            
            max_x = D - remaining_slots
            for x in range(current_max + 1, max_x + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for a in A:
                    d = x - a
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Update state
                A.append(x)
                old_diffs_size = len(diffs)
                diffs.update(new_diffs)
                old_covered_size = len(covered)
                new_covered = required_diffs.intersection(new_diffs)
                covered.update(new_covered)
                
                res = solve(idx + 1, x, remaining_slots - 1)
                if res is not None:
                    return res
                
                # Backtrack
                A.pop()
                # Restore diffs (simplest is to recompute or store old set, but set difference_update works if we stored new_diffs)
                diffs.difference_update(new_diffs)
                covered.difference_update(new_covered)
            
            return None
        
        return solve(1, 0, k - 2)

    # Iterate D upwards
    # Start from theoretical minimum for k=9 which is 36.
    # However, k=9 might not have a solution for D=36.
    # We should try k=9 first for increasing D.
    # If k=9 fails for all reasonable D, try k=10, etc.
    # But usually the minimal D is found with the minimal k or k+1.
    
    # Let's iterate D and for each D, try minimal k first.
    
    start_time = time.time()
    timeout = 110
    
    # Theoretical lower bound for D is 30, but Sidon constraint pushes it up.
    # For k=9, D >= 36.
    
    for D in range(36, 100):
        if time.time() - start_time > timeout:
            # Fallback
            return greedy_construct()
        
        # Try minimal k first
        for k in range(9, 13):
            # Check if k*(k-1)/2 <= D is possible? No, diffs are distinct, so we need at least 30 distinct diffs.
            # The max number of distinct diffs in [0, D] is D.
            # So we need D >= 30.
            # Also k*(k-1)/2 <= D is NOT required, only that the diffs fit in 1..D.
            # So k can be large as long as k*(k-1)/2 <= D?
            # Yes, because all C(k,2) diffs must be distinct and in 1..D.
            # So C(k,2) <= D.
            if k * (k - 1) // 2 > D:
                break # Larger k will also fail
            
            result = find_set_for_D(D, k)
            if result is not None:
                return sorted(result)
    
    # If loop finishes without finding
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
