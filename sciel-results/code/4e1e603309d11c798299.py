"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE_FRACTION = 0.85  # use 85% of the 60s budget as a safety margin

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    best_set = None
    best_size = N + 1
    
    # Helper: check if a set is a valid Sidon set
    def is_sidon(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                s_val = s[i] + s[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True
    
    # Helper: check if a Sidon set is inclusion-maximal in [1, N]
    def is_maximal(s):
        if not is_sidon(s):
            return False
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        s_set = set(s)
        for x in range(1, N + 1):
            if x in s_set:
                continue
            # Try adding x: check all new sums
            ok = True
            for a in s:
                if a + x in sums:
                    ok = False
                    break
            if ok and 2 * x in sums:
                ok = False
            if ok:
                return False
        return True
    
    # Helper: maximalize a set by greedily adding the most blocking element
    def maximalize(s, time_limit=None):
        s = list(s)
        s_set = set(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        
        while True:
            if time_limit is not None and time.time() > time_limit:
                break
            candidates = []
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                # Check if x can be added
                ok = True
                for a in s:
                    if a + x in sums:
                        ok = False
                        break
                if ok and 2 * x in sums:
                    ok = False
                if ok:
                    # Count how many other candidates this would block
                    block_count = 0
                    for y in range(1, N + 1):
                        if y in s_set or y == x:
                            continue
                        # If we add x, can y still be added?
                        # y is blocked if y+x is in sums or in {x+a for a in s} or 2y in sums or 2y in {x+a}
                        # Actually, after adding x, the new sums include x+x=2x and x+a for all a in s
                        # y is blocked if for some a in s ∪ {x}, y+a is in sums ∪ {2x} ∪ {x+b for b in s}
                        # Simpler: just check if y could be added after x is added
                        new_sums = sums | {2*x} | {x + a for a in s}
                        y_ok = True
                        for a in s + [x]:
                            if a + y in new_sums:
                                y_ok = False
                                break
                        if y_ok and 2 * y in new_sums:
                            y_ok = False
                        if not y_ok:
                            block_count += 1
                    candidates.append((block_count, x))
            
            if not candidates:
                break
            
            # Pick the candidate that blocks the most
            candidates.sort(reverse=True)
            best_block, best_x = candidates[0]
            
            # Add best_x
            s.append(best_x)
            s_set.add(best_x)
            sums.add(2 * best_x)
            for a in s[:-1]:
                sums.add(a + best_x)
        
        return s
    
    # Strategy 1: Randomized greedy maximalization with many restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomly shuffle the order in which we consider elements
        # But use a smarter approach: start with a random subset, then maximalize
        # Try different sizes of initial random subsets
        initial_size = rng.randint(1, 8)
        initial_set = rng.sample(range(1, N + 1), initial_size)
        
        # Check if initial set is Sidon
        if not is_sidon(initial_set):
            continue
        
        # Maximalize
        result = maximalize(initial_set, time_limit=deadline)
        
        if result is None or not result:
            continue
        
        # Verify it's actually maximal and Sidon
        if is_maximal(result):
            if len(result) < best_size:
                best_size = len(result)
                best_set = result
        
        # Also try a simpler randomized greedy: pick random available element each step
        s2 = []
        s2_set = set()
        sums2 = set()
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        for x in order:
            if x in s2_set:
                continue
            ok = True
            for a in s2:
                if a + x in sums2:
                    ok = False
                    break
            if ok and 2 * x in sums2:
                ok = False
            if ok:
                s2.append(x)
                s2_set.add(x)
                sums2.add(2 * x)
                for a in s2[:-1]:
                    sums2.add(a + x)
        
        if is_maximal(s2):
            if len(s2) < best_size:
                best_size = len(s2)
                best_set = s2
    
    # Fallback: deterministic ascending greedy
    if best_set is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected
    
    return best_set
# EVOLVE-BLOCK-END
