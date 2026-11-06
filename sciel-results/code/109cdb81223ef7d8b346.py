import time
import random

def construct():
    """
    Return up to 48 distinct positive integers in [1, 512] such that no element is the
    arithmetic mean of any subset of at least two other elements.
    Maximize cardinality.
    """
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    
    MAX_N = 512
    TARGET = 48

    def check_violation(x, others):
        """
        Check if x is the mean of any subset of `others` with size >= 2.
        Equivalent: exists subset S of others, |S| >= 2, sum(S) = |S| * x
        Let vals = [y - x for y in others]. Need non-empty subset summing to 0.
        Since all y != x, no val is 0, so any subset summing to 0 has size >= 1.
        But we need size >= 2. A single element summing to 0 means y-x=0, i.e., y=x, which is excluded.
        So any non-empty subset summing to 0 automatically has size >= 2.
        
        We use bitset DP with offset.
        """
        vals = [y - x for y in others]
        min_s = sum(v for v in vals if v < 0)
        max_s = sum(v for v in vals if v > 0)
        if min_s == 0 and max_s == 0:
            # All vals are 0, but that's impossible since y != x
            return False
        offset = -min_s
        # bits[i] means sum i is possible
        # Initialize: only sum 0 is possible (empty set)
        bits = 1 << offset  # sum 0 at index offset
        
        for v in vals:
            if v >= 0:
                bits |= bits << v
            else:
                bits |= bits >> (-v)
        
        # Check if sum 0 (at index offset) is reachable by a NON-EMPTY subset.
        # The empty set always gives sum 0. So bit offset is always set initially.
        # We need to distinguish: was bit offset set only by the empty set, or also by some non-empty subset?
        # Trick: if any single v == 0, that's a size-1 subset. But v != 0 always.
        # So if the final bits has bit offset set, it could be from empty set OR non-empty.
        # We need to check if there's a non-empty subset summing to 0.
        # Alternative: check if any pair sums to 0? Too slow.
        # Better approach: run DP but track whether 0 is reachable from non-empty.
        # Actually, we can just check: if we remove the initial contribution...
        # Simpler: since no v is 0, the only way to get sum 0 is with >= 2 elements.
        # But the bitset includes the empty set. So bit offset is ALWAYS set.
        # We need a different way.
        
        # Let's use a set-based DP for this specific check since it's called infrequently.
        # Or: check if there's a subset of size >= 2 summing to 0.
        # We can do: for each pair, check if sum is 0. If yes, return True.
        # Then for triples, etc. But that's O(n^3).
        
        # Actually, the cleanest way: use the fact that if a non-empty subset sums to 0,
        # then the bit at offset would be set in a DP that starts with NO elements contributing to sum 0.
        # But that breaks the DP.
        
        # Let's just use a set of reachable sums, excluding the empty set's contribution to 0.
        # Start with sums = set()
        # For each v: new_sums = {s + v for s in sums} | {v}
        # If 0 is in final sums, return True.
        
        sums = set()
        for v in vals:
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            new_sums.add(v)
            sums = sums | new_sums
            if 0 in sums:
                return True
        return False

    def check_violation_fast(x, others):
        """
        Faster check using bitset. The key insight: since no element is 0,
        a subset summing to 0 must have size >= 2. The empty set sums to 0 but has size 0.
        We can check if bit offset is set in the final DP, but we need to exclude the empty set.
        
        Approach: Instead of starting with bits = 1 << offset (empty set),
        start with bits = 0, and for each v, do:
          new_bits = (bits << v) | (1 << (offset + v))
        This tracks only non-empty subsets.
        Then check if bit offset is set.
        """
        vals = [y - x for y in others]
        min_s = sum(v for v in vals if v < 0)
        max_s = sum(v for v in vals if v > 0)
        if min_s > 0 or max_s < 0:
            return False
        offset = -min_s
        bits = 0  # no non-empty subset yet
        for v in vals:
            if v >= 0:
                bits |= (bits << v) | (1 << (offset + v))
            else:
                bits |= (bits >> (-v)) | (1 << (offset + v))
        return (bits >> offset) & 1

    # Greedy construction: add elements one by one, picking the one that causes the fewest conflicts
    def greedy_construct():
        selected = []
        available = set(range(1, MAX_N + 1))
        
        for _ in range(TARGET):
            if time.time() > deadline:
                break
            best_x = None
            best_conflicts = float('inf')
            
            # Try all available elements, pick the one with fewest violations
            # For speed, shuffle and take first few that work
            candidates = list(available)
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x is mean of any subset of selected
                if check_violation_fast(x, selected):
                    continue
                # Check if any y in selected is mean of subset including x and others
                conflict = False
                for y in selected:
                    others = [z for z in selected if z != y] + [x]
                    if check_violation_fast(y, others):
                        conflict = True
                        break
                if not conflict:
                    best_x = x
                    break
            
            if best_x is not None:
                selected.append(best_x)
                available.remove(best_x)
            else:
                # No element works, try harder: check all
                found = False
                for x in available:
                    if check_violation_fast(x, selected):
                        continue
                    conflict = False
                    for y in selected:
                        others = [z for z in selected if z != y] + [x]
                        if check_violation_fast(y, others):
                            conflict = True
                            break
                    if not conflict:
                        selected.append(x)
                        available.remove(x)
                        found = True
                        break
                if not found:
                    break
        
        return selected

    # Local search: swap elements to reduce violations
    def count_violations(selected):
        count = 0
        n = len(selected)
        for i in range(n):
            x = selected[i]
            others = [selected[j] for j in range(n) if j != i]
            if check_violation_fast(x, others):
                count += 1
        return count

    def local_search(initial):
        current = list(initial)
        current_violations = count_violations(current)
        best = list(current)
        best_violations = current_violations
        
        used = set(current)
        available = set(range(1, MAX_N + 1)) - used
        
        while time.time() < deadline and best_violations > 0:
            improved = False
            # Try swapping each element with a random available one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                old = current[i]
                if not available:
                    break
                # Try a few swaps
                for _ in range(3):
                    new = random.choice(list(available))
                    candidate = list(current)
                    candidate[i] = new
                    
                    # Quick check: does new cause a violation?
                    others = [z for j, z in enumerate(candidate) if j != i]
                    if check_violation_fast(new, others):
                        continue
                    # Check if any existing y is now violated
                    valid = True
                    for j, y in enumerate(candidate):
                        if j == i:
                            continue
                        others_j = [z for k, z in enumerate(candidate) if k != j]
                        if check_violation_fast(y, others_j):
                            valid = False
                            break
                    if valid:
                        # Check if it reduces violations
                        new_violations = count_violations(candidate)
                        if new_violations < current_violations:
                            current = candidate
                            current_violations = new_violations
                            used.remove(old)
                            used.add(new)
                            available.remove(new)
                            available.add(old)
                            improved = True
                            if current_violations == 0:
                                return current
                            if current_violations < best_violations:
                                best = list(current)
                                best_violations = current_violations
                            break
        
            if not improved:
                # Try removing the most-violating element and adding a new one
                if len(current) > 1:
                    # Find element with most violations
                    worst_idx = -1
                    worst_count = -1
                    for i in range(len(current)):
                        x = current[i]
                        others = [z for j, z in enumerate(current) if j != i]
                        # Count how many subsets make x a mean - just check if x is violated
                        # For simplicity, just remove a random element
                        pass
                    # Remove a random element
                    idx = random.randint(0, len(current) - 1)
                    old = current[idx]
                    current.pop(idx)
                    used.remove(old)
                    available.add(old)
                    
                    # Add a new random element
                    if available:
                        new = random.choice(list(available))
                        current.append(new)
                        used.add(new)
                        available.remove(new)
                    
                    current_violations = count_violations(current)
                    if current_violations < best_violations:
                        best = list(current)
                        best_violations = current_violations
        
        return best

    # Strategy: Try greedy first, then local search
    result = greedy_construct()
    
    if len(result) < TARGET:
        # Pad with local search
        result = local_search(result)
    
    # Final: try to extend if possible
    if len(result) < TARGET and time.time() < deadline:
        available = set(range(1, MAX_N + 1)) - set(result)
        while time.time() < deadline and len(result) < TARGET:
            found = False
            for x in available:
                if check_violation_fast(x, result):
                    continue
                conflict = False
                for y in result:
                    others = [z for z in result if z != y] + [x]
                    if check_violation_fast(y, others):
                        conflict = True
                        break
                if not conflict:
                    result.append(x)
                    available.remove(x)
                    found = True
                    break
            if not found:
                break
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements")
    print(result)
