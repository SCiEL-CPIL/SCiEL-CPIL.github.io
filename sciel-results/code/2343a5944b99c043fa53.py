import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # Target differences 1..30
    TARGET_MASK = (1 << 30) - 1

    best_set = None
    best_diameter = float('inf')

    def get_info(A):
        if len(A) < 2: return None
        diffs = [y - x for i, x in enumerate(A) for y in A[i+1:]]
        diff_set = set(diffs)
        duplicates = len(diffs) - len(diff_set)
        missing = sum(1 for d in range(1, 31) if d not in diff_set)
        return diffs, diff_set, duplicates, missing

    def solve_for_D(D):
        """
        Backtracking solver to find a Sidon set in [0, D] containing 0,
        covering all differences 1..30.
        """
        # Precompute for each x in 1..D the bitmask of diffs 1..30 it covers
        # and the list of diffs it produces with existing elements (handled in DFS)
        
        # We'll use a recursive DFS.
        # State:
        #   elements: list of chosen elements (sorted)
        #   used_diffs: set of all differences produced so far
        #   covered_mask: bitmask of which diffs 1..30 are covered
        #   last_elem: last element added (for ordering)
        
        # To speed up, we can precompute for each x in 1..D:
        #   cover_mask[x]: bitmask of d in 1..30 such that d is a potential diff
        #   Actually, the diff depends on the other element y.
        #   So we can't precompute a simple mask for x alone.
        
        # However, we can precompute for each pair (x, y) with x > y, the diff d = x - y.
        # And if d <= 30, it contributes to the mask.
        
        # Let's try a simple DFS with pruning.
        
        # Pruning 1: If the number of remaining elements to choose is insufficient to cover missing diffs.
        # Pruning 2: If adding any element would create a duplicate diff.
        
        # To make it faster, we can order candidates by how many new missing diffs they cover.
        
        target_diffs = set(range(1, 31))
        
        # We need at least 1 element (0) + enough others to cover 30 diffs.
        # A set of size k has k*(k-1)/2 diffs.
        # k=9 -> 36 diffs. Enough.
        # k=8 -> 28 diffs. Not enough.
        # So min size is 9.
        
        # We will try to find a set of size 9, then 10, etc.
        # But the solver just needs to find *any* valid set.
        
        def dfs(elements, used_diffs, covered_mask, last_elem):
            if time.time() > deadline:
                return None
            
            if covered_mask == TARGET_MASK:
                return elements[:]
            
            # Pruning: Check if we can cover the rest.
            # Max elements we can add: D - last_elem (roughly)
            # But Sidon constraint is tight.
            
            # Candidates: x in [last_elem + 1, D]
            # Sort candidates by potential coverage?
            # For each x, compute how many missing diffs it covers with existing elements.
            # This is O(k) per candidate.
            
            candidates = []
            for x in range(last_elem + 1, D + 1):
                # Check validity
                valid = True
                new_covered = 0
                for y in elements:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    if d <= 30 and (1 << (d-1)) not in covered_mask:
                        new_covered += 1
                
                if valid:
                    candidates.append((new_covered, x))
            
            # Sort by coverage (descending) to find solutions faster
            candidates.sort(key=lambda t: -t[0])
            
            for _, x in candidates:
                new_elements = elements + [x]
                new_used_diffs = used_diffs.copy()
                new_covered_mask = covered_mask
                
                for y in elements:
                    d = x - y
                    new_used_diffs.add(d)
                    if d <= 30:
                        new_covered_mask |= (1 << (d-1))
                
                # Recurse
                res = dfs(new_elements, new_used_diffs, new_covered_mask, x)
                if res is not None:
                    return res
            
            return None
        
        # Start with 0
        initial_elements = [0]
        initial_used_diffs = set()
        initial_covered_mask = 0
        
        return dfs(initial_elements, initial_used_diffs, initial_covered_mask, 0)

    # Try small diameters first
    # Lower bound for D:
    # If we have a Sidon set of size k, the number of diffs is k(k-1)/2.
    # We need at least 30 diffs.
    # k=9 -> 36 diffs.
    # The maximum possible diff is D.
    # The sum of diffs is at least 1+...+30 = 465.
    # Also, for a Sidon set in [0, D], the number of distinct diffs is at most D.
    # More tightly, the diameter D must be large enough to accommodate the structure.
    # Known smallest Sidon set covering 1..30?
    # Let's start from D=60 and go up.
    
    for D in range(60, 200):
        if time.time() > deadline:
            break
            
        # Try to solve for this D
        result = solve_for_D(D)
        
        if result is not None:
            # Verify
            info = get_info(result)
            if info:
                _, _, dups, miss = info
                if dups == 0 and miss == 0:
                    diam = max(result)
                    if diam < best_diameter:
                        best_diameter = diam
                        best_set = result
                        # Since we iterate D in increasing order, the first valid one is optimal?
                        # Not necessarily, because a larger D might allow a smaller set?
                        # No, the diameter is max(element). If we find one in [0, D], its diameter <= D.
                        # So the first D that yields a solution is the minimal diameter?
                        # Yes, because we are scanning D upwards.
                        return best_set

    # Fallback: Use SA or Greedy if backtracking fails or times out
    if best_set is None:
        # SA Fallback
        for n in range(9, 16):
            if time.time() > deadline:
                break
                
            for restart in range(10):
                if time.time() > deadline:
                    break
                    
                A = [0] + random.sample(range(1, 300), n-1)
                A.sort()
                
                T = 100.0
                T_min = 1.0
                cooling = 0.99
                max_iter = 2000
                
                curr_A = A
                info = get_info(curr_A)
                if info:
                    _, _, dups, miss = info
                    curr_cost = dups * 1000 + miss * 100 + max(curr_A)
                else:
                    curr_cost = 1000000
                
                best_local_A = curr_A
                best_local_cost = curr_cost
                
                for i in range(max_iter):
                    if time.time() > deadline:
                        break
                        
                    idx = random.randint(1, n-1)
                    max_val = max(curr_A)
                    new_val = random.randint(0, max_val + 10)
                    
                    if new_val in curr_A:
                        continue
                        
                    new_A = curr_A.copy()
                    new_A[idx] = new_val
                    new_A.sort()
                    
                    info = get_info(new_A)
                    if not info:
                        new_cost = 1000000
                    else:
                        _, _, dups, miss = info
                        new_cost = dups * 1000 + miss * 100 + max(new_A)
                    
                    delta = new_cost - curr_cost
                    if delta < 0:
                        curr_A = new_A
                        curr_cost = new_cost
                        if curr_cost < best_local_cost:
                            best_local_cost = curr_cost
                            best_local_A = curr_A
                    else:
                        if random.random() < pow(2.718, -delta / T):
                            curr_A = new_A
                            curr_cost = new_cost
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                
                info = get_info(best_local_A)
                if info:
                    _, _, dups, miss = info
                    if dups == 0 and miss == 0:
                        diam = max(best_local_A)
                        if diam < best_diameter:
                            best_diameter = diam
                            best_set = best_local_A

    # Final Fallback: Greedy
    if best_set is None:
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
        best_set = a

    return best_set
