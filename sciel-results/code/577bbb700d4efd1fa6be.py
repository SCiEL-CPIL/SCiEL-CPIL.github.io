import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Required differences: 1..30
    # Bitmask for 1..30
    # Bit i represents difference i+1? Or difference i?
    # Let's use bit d for difference d.
    # So bit 0 is unused, bit 1 is diff 1, ..., bit 30 is diff 30.
    req_mask = 0
    for d in range(1, 31):
        req_mask |= (1 << d)
    
    best_solution = None
    best_M = 10**6

    def solve_for_M(M, max_n=12):
        """
        Solve for a Sidon set in [0, M] covering 1..30.
        Tries different sizes n.
        """
        # We know n must be at least 9.
        # We can iterate n or just let DFS find any size <= max_n.
        # It's better to fix n or let it grow?
        # If we let it grow, we might add useless elements.
        # But we want to minimize M, so for a fixed M, we just want ANY valid set.
        # Smaller n is preferred? Not necessarily, but it's easier to satisfy.
        # Actually, if a set of size n works, a set of size n+1 might not (harder constraints).
        # So we should try to find a solution with minimal n first?
        # Or just find any solution.
        
        def dfs(current_set, covered_mask, missing_mask, depth):
            nonlocal best_solution, best_M
            if time.time() > deadline:
                return None
            
            if not missing_mask:
                return list(current_set)
            
            # Pruning: If we have too many elements, we might exceed max_n?
            # The problem says "at most 32".
            # But larger n makes it harder.
            # We can limit depth to, say, 12.
            if depth >= max_n:
                return None
                
            # Heuristic: Check if remaining missing diffs can be covered.
            # Max new diffs from k more elements: k * len(current_set) + k*(k-1)/2.
            # This is an upper bound on new diffs.
            # We need to cover popcount(missing_mask) diffs.
            # If popcount(missing_mask) > max_possible_new_diffs, prune.
            # But this is loose.
            
            next_min = current_set[-1] + 1
            candidates = []
            
            # Optimization: If 1 is missing, and 1 not in set, we need x-y=1.
            # This implies some pair is consecutive.
            
            for x in range(next_min, M + 1):
                # Check Sidon validity
                valid = True
                new_diffs_mask = 0
                for y in current_set:
                    d = x - y
                    # d is always positive since x > y
                    if d > M: continue # Should not happen
                    # Check if d is already covered
                    if (covered_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    # Calculate gain: how many missing diffs does this cover?
                    # gain_mask = new_diffs_mask & missing_mask
                    # gain_count = popcount(gain_mask)
                    gain_mask = new_diffs_mask & missing_mask
                    # Python 3.10+ has int.bit_count()
                    # For compatibility, bin(x).count("1")
                    gain = bin(gain_mask).count("1")
                    
                    # Prioritize candidates that cover more missing diffs.
                    # Also, prefer smaller x?
                    candidates.append((x, new_diffs_mask, gain))
            
            # Sort by gain descending, then x ascending
            candidates.sort(key=lambda t: (-t[2], t[0]))
            
            for x, new_diffs_mask, gain in candidates:
                next_covered = covered_mask | new_diffs_mask
                next_missing = missing_mask & ~new_diffs_mask
                
                res = dfs(current_set + [x], next_covered, next_missing, depth + 1)
                if res is not None:
                    return res
            
            return None

        # Start search
        # Initial set: [0]
        # Covered: 0
        # Missing: req_mask
        return dfs([0], 0, req_mask, 1)

    # Iterate M from lower bound
    # Lower bound for n=9 is 44.
    # Let's start from 44.
    # If we find a solution with M, we return it.
    # We want to minimize M.
    
    for M in range(44, 100):
        if time.time() > deadline:
            break
        
        sol = solve_for_M(M)
        if sol is not None:
            return sol
            
    # Fallback
    # If no solution found, return a known valid set (greedy)
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
