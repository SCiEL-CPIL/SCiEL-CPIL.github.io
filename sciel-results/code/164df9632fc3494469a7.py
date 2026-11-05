import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Required differences
    req_diffs = set(range(1, 31))
    
    # We know M must be at least 30.
    # Upper bound: A known Sidon set with diameter ~100 exists.
    # We will try M from 30 upwards.
    
    best_solution = None
    
    def is_valid_sidon_addition(current_set, x, covered_diffs):
        """Check if adding x to current_set maintains Sidon property."""
        for y in current_set:
            d = x - y
            if d in covered_diffs:
                return False
        return True

    def search(M):
        """
        Search for a valid Sidon set in [0, M] containing 0, covering req_diffs.
        Returns the set if found, else None.
        """
        # State: current_set (list), covered_diffs (set)
        # We use recursion with backtracking.
        
        # Optimization: Use a set for covered_diffs for O(1) lookup.
        # We also track 'missing' diffs to check completion quickly.
        
        def dfs(current_set, covered_diffs, missing_diffs):
            # Check time
            if time.time() > deadline:
                return None
            
            if not missing_diffs:
                return list(current_set)
            
            # Pruning: Can we cover remaining missing diffs?
            # Max elements we can add is roughly bounded by M.
            # Let's estimate max possible new differences.
            # If we add k elements, max new diffs is k * len(current_set) + k*(k-1)/2.
            # We don't know k, but max k is M - current_set[-1].
            # This bound is loose.
            
            # Better pruning: If the largest missing diff is greater than M - current_set[0] (which is M), 
            # it's impossible. But M is fixed.
            
            # Heuristic: Order candidates by how many missing diffs they cover.
            next_min = current_set[-1] + 1
            candidates = []
            
            for x in range(next_min, M + 1):
                # Check Sidon validity
                valid = True
                new_diffs = set()
                for y in current_set:
                    d = x - y
                    if d in covered_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    # Calculate gain
                    gain = len(new_diffs & missing_diffs)
                    # We only care about covering missing ones, but adding new ones is good.
                    # Actually, we just need to cover req_diffs.
                    # If gain == 0, it might still be useful to bridge gaps? 
                    # No, Sidon sets don't "bridge" gaps in differences directly, 
                    # but they allow larger differences.
                    # However, if we don't cover any missing diff, we are just increasing size.
                    # We should prioritize high gain.
                    candidates.append((x, new_diffs, gain))
            
            # Sort by gain descending, then by x ascending (or whatever)
            # Ties in gain: prefer smaller x to keep diameter small? 
            # But M is fixed, so we just want to find *any*.
            # Smaller x leaves more room for larger x later?
            # Let's sort by gain desc, then x asc.
            candidates.sort(key=lambda t: (-t[2], t[0]))
            
            for x, new_diffs, gain in candidates:
                # Pruning: If gain is 0 and we have no more "easy" wins?
                # Just recurse.
                
                next_covered = covered_diffs | new_diffs
                next_missing = missing_diffs - new_diffs
                
                res = dfs(current_set + [x], next_covered, next_missing)
                if res is not None:
                    return res
            
            return None

        # Start search
        # Initial set: [0]
        # Covered diffs: empty
        # Missing: req_diffs
        return dfs([0], set(), req_diffs)

    # Iterate M from lower bound
    # Lower bound: 30
    # We can also use a binary search if we had a fast "no solution" check, 
    # but linear is fine for small M.
    
    for M in range(30, 200):
        if time.time() > deadline:
            break
        
        sol = search(M)
        if sol is not None:
            return sol
        
    # If no solution found within time or range, return a known valid set (fallback)
    # A known Sidon set covering 1..30:
    # From literature, a perfect difference set of order 4 has size 13, diameter 15? No.
    # Let's just return the greedy result from the seed as a fallback to ensure validity.
    # But we want to minimize M.
    
    # Fallback: Greedy construction
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
