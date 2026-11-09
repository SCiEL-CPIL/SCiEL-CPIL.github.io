import time
import sys

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def check_sidon(s):
        """Check if sorted list s is a Sidon set (all positive differences distinct)."""
        if len(s) < 2:
            return True
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def get_diffs(s):
        """Get all positive differences of sorted list s."""
        diffs = []
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(s[j] - s[i])
        return diffs
    
    def has_all_1_30(s):
        """Check if all differences 1-30 are present in sorted list s."""
        diff_set = set()
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d = s[j] - s[i]
                if 1 <= d <= 30:
                    diff_set.add(d)
                if d > 30:
                    continue
        return all(d in diff_set for d in range(1, 31))
    
    def is_valid_sidon(s):
        """Check Sidon property efficiently."""
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d = s[j] - s[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    # Backtracking search: build Sidon set containing 0, realize diffs 1-30
    # Try to minimize max element
    
    def search(current, missing_diffs, max_val, best_result):
        """
        current: sorted list of elements (includes 0)
        missing_diffs: set of differences in 1-30 not yet realized
        max_val: current maximum element
        best_result: list to store best found (min max element)
        """
        if time.time() > deadline:
            return
        if not missing_diffs:
            # All differences realized
            if max_val < best_result[0]:
                best_result[0] = max_val
                best_result[1] = list(current)
            return
        if len(current) >= 32:
            return
        if max_val >= best_result[0]:
            return
        
        # Try adding new elements, starting from smallest possible
        # The new element must be > max_val (to keep sorted and avoid negative)
        # Actually, we could insert before max_val too, but let's append for now
        # and also try inserting
        
        start_val = max_val + 1
        # Upper bound: don't go too high
        upper = min(start_val + 200, best_result[0] if best_result[0] > 0 else 10**6)
        
        for new_val in range(start_val, upper + 1):
            if time.time() > deadline:
                return
            trial = current + [new_val]
            # Quick Sidon check
            if not is_valid_sidon(trial):
                continue
            # Check what differences this realizes
            new_diffs = set()
            for x in current:
                d = new_val - x
                if 1 <= d <= 30:
                    new_diffs.add(d)
            remaining = missing_diffs - new_diffs
            search(trial, remaining, new_val, best_result)
            if best_result[0] <= max_val:
                return  # No point continuing if we found something better
        
        # Also try inserting new_val before max_val (between existing elements)
        # Try inserting at various positions
        for pos in range(len(current) - 1, 0, -1):
            if time.time() > deadline:
                return
            # Try values between current[pos-1] and current[pos]
            lo = current[pos-1] + 1
            hi = current[pos] - 1
            if lo >= hi:
                continue
            for new_val in range(hi, lo - 1, -1):  # Try larger first (closer to current[pos])
                if time.time() > deadline:
                    return
                trial = current[:pos] + [new_val] + current[pos:]
                if not is_valid_sidon(trial):
                    continue
                new_diffs = set()
                for x in current:
                    d = abs(new_val - x)
                    if 1 <= d <= 30:
                        new_diffs.add(d)
                remaining = missing_diffs - new_diffs
                search(trial, remaining, max_val, best_result)
    
    # Initial best: use a known construction or greedy
    # Let's first try a quick greedy to get an initial upper bound
    greedy_set = [0]
    greedy_missing = set(range(1, 31))
    for d in range(1, 31):
        if d not in greedy_missing:
            continue
        found = False
        # Try adding one element to realize d
        for base in greedy_set:
            for candidate in [base + d, base - d]:
                if candidate < 0 or candidate in greedy_set:
                    continue
                trial = sorted(greedy_set + [candidate])
                if is_valid_sidon(trial):
                    # Check if it realizes d
                    realized = set()
                    for i in range(len(trial)):
                        for j in range(i+1, len(trial)):
                            dd = trial[j] - trial[i]
                            if 1 <= dd <= 30:
                                realized.add(dd)
                    greedy_missing -= realized
                    greedy_set = trial
                    found = True
                    break
            if found:
                break
        if not found:
            # Try two elements
            for base in greedy_set:
                for a_off in range(1, 50):
                    for b_off in range(1, 50):
                        a = base + a_off
                        b = base + a_off + b_off
                        if a in greedy_set or b in greedy_set or a < 0 or b < 0:
                            continue
                        trial = sorted(greedy_set + [a, b])
                        if is_valid_sidon(trial):
                            realized = set()
                            for i in range(len(trial)):
                                for j in range(i+1, len(trial)):
                                    dd = trial[j] - trial[i]
                                    if 1 <= dd <= 30:
                                        realized.add(dd)
                            if d in realized:
                                greedy_missing -= realized
                                greedy_set = trial
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
        if not found:
            # Fallback: add element far away
            new_el = 2 * max(greedy_set) + 3 * d + 10
            while True:
                trial = sorted(greedy_set + [new_el])
                if is_valid_sidon(trial):
                    realized = set()
                    for i in range(len(trial)):
                        for j in range(i+1, len(trial)):
                            dd = trial[j] - trial[i]
                            if 1 <= dd <= 30:
                                realized.add(dd)
                    greedy_missing -= realized
                    greedy_set = trial
                    break
                new_el += 1
    greedy_max = max(greedy_set) if greedy_set else 0
    
    # Now do proper search with this upper bound
    best = [greedy_max + 1, list(greedy_set)]
    
    # Try search starting from [0]
    search([0], set(range(1, 31)), 0, best)
    
    if best[1] and best[0] < greedy_max:
        return best[1]
    return greedy_set

