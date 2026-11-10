import time
import random

def construct():
    deadline = time.time() + 85.0  # Safety margin
    N = 256
    MAX_SIZE = 60

    # Fast validity check using bitsets
    # sums[k] is an integer where bit i is 1 if sum i is achievable with k elements
    def check_valid(vals):
        if not vals:
            return True
        n = len(vals)
        # Initialize sums
        # sums[0] = 1 (bit 0 set, sum 0 with 0 elements)
        sums = [0] * (n + 1)
        sums[0] = 1
        
        # DP: add each value
        for v in vals:
            # Update from high k to low k to avoid reusing same element
            # We only need to update up to current max k
            # But n is small, so just iterate n-1 down to 0
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        
        # Check disjointness
        # sums[i] & sums[j] == 0 for i != j
        # Optimization: we can just check if any sum appears in multiple layers.
        # A sum s is in layer k if (sums[k] >> s) & 1
        # It is invalid if there exists s such that count of layers containing s > 1.
        # Simpler: pairwise AND.
        # n is ~40, n^2/2 = 800 checks. Each check is a big int AND.
        # Big ints are ~15000 bits ~ 200 bytes. AND is very fast.
        for i in range(n + 1):
            if not sums[i]:
                continue
            for j in range(i + 1, n + 1):
                if sums[j] and (sums[i] & sums[j]):
                    return False
        return True

    def greedy_build(pool, order):
        """Build a set by iterating 'order' and adding if valid."""
        result = []
        result_set = set()
        for v in order:
            if v in result_set:
                continue
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
                result_set.add(v)
        return result

    best_set = []
    best_size = 0

    pool = list(range(1, N + 1))
    
    # Strategy 1: Deterministic Greedies
    # Ascending
    s = greedy_build(pool, sorted(pool))
    if len(s) > best_size:
        best_set = s
        best_size = len(s)
        
    # Descending
    s = greedy_build(pool, sorted(pool, reverse=True))
    if len(s) > best_size:
        best_set = s
        best_size = len(s)
        
    # Strategy 2: Randomized Greedies
    # We will run as many as possible
    rng = random.Random(42)
    
    # Local Search / Improvement
    def improve(current_set):
        """Try to improve a valid set by adding/removing/swapping."""
        cur = list(current_set)
        cur_set = set(cur)
        improved = False
        # Try to add any missing element
        # Sort candidates by value to prefer smaller ones (keeps sums tight)
        candidates = [v for v in pool if v not in cur_set]
        candidates.sort()
        
        # Phase 1: Try to add elements
        for v in candidates:
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
                improved = True
                # Re-evaluate candidates? Or just continue?
                # Since we add small ones first, subsequent small ones might become invalid.
                # But we are iterating a static list. 
                # If we add v, the validity of other candidates changes.
                # To be safe, we should re-check or break and restart this phase.
                # Given speed, let's just do one pass.
        
        # Phase 2: Local Search (Remove one, try to add two or more)
        # This is complex. Let's stick to Randomized Greedy restarts which are simpler and robust.
        return cur

    # Run many randomized trials
    # Bias: Smaller numbers are more likely to be picked early?
    # Let's just shuffle.
    
    start_time = time.time()
    iterations = 0
    
    while time.time() < deadline - 5.0:
        iterations += 1
        
        # Generate a random order
        # To exploit the "smallest first" insight, we can sort by a key that is mostly value, but with noise
        # key = v + rng.uniform(0, 10)
        # Or just shuffle.
        order = pool[:]
        rng.shuffle(order)
        
        s = greedy_build(pool, order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # Also try a "biased" order: sort by v * (1 + small_random)
        # This tends to keep small numbers early but allows some mixing
        if iterations % 2 == 0:
            order = sorted(pool, key=lambda x: x + rng.uniform(0, 5))
            s = greedy_build(pool, order)
            if len(s) > best_size:
                best_set = s
                best_size = len(s)

    # Final improvement on the best set found
    if best_set and time.time() < deadline - 2.0:
        # Try to grow the best set further
        cur = list(best_set)
        cur_set = set(cur)
        candidates = [v for v in pool if v not in cur_set]
        candidates.sort()
        
        # Try adding in ascending order
        for v in candidates:
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
        
        if len(cur) > best_size:
            best_set = cur
            best_size = len(cur)
            
        # Try a few random additions/removals on the best set
        for _ in range(100):
            if time.time() > deadline - 1.0:
                break
            # Remove a random element
            if not cur:
                continue
            idx = rng.randint(0, len(cur) - 1)
            removed = cur.pop(idx)
            cur_set.discard(removed)
            
            # Try to add random elements to fill back and potentially exceed
            # Add back the removed one? Or others?
            # Let's try to add all missing ones in random order
            missing = [v for v in pool if v not in cur_set]
            rng.shuffle(missing)
            
            for v in missing:
                cand = cur + [v]
                if check_valid(cand):
                    cur = cand
                    cur_set.add(v)
            
            if len(cur) > best_size:
                best_set = cur
                best_size = len(cur)
            else:
                # Revert if we didn't improve? 
                # Actually, if we removed one and didn't get a bigger set, 
                # we might have lost the previous best state.
                # We should only update best_set if we improve.
                # But 'cur' is now a different set.
                # If len(cur) < best_size, we should probably restore 'best_set' to continue from the known good state?
                # Or just keep 'cur' as the new base for next local search step?
                # Keeping 'cur' allows exploration.
                pass

    result = [int(x) for x in best_set]
    return result

