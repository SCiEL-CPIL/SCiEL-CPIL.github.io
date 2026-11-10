import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    N = 256
    MAX_SIZE = 60

    def check_valid(vals):
        """Check if vals satisfy the cardinality-separated subset sum property.
        Returns True if valid. Uses bitset DP per cardinality."""
        if not vals:
            return True
        n = len(vals)
        # Quick necessary condition: min sum of k+1 elements > max sum of k elements
        # This is sufficient but not necessary, but helps reject bad candidates fast
        sorted_vals = sorted(vals)
        # For cardinality k, min sum is sum of k smallest
        # For cardinality k+1, min sum is sum of k+1 smallest
        # We need: min sum of k+1 > max sum of k for all k
        # Actually the condition is: no sum in layer k equals any sum in layer j for k != j
        # A quick check: the minimum achievable sum with k+1 elements must exceed
        # the maximum achievable sum with k elements? No, that's too strong.
        # Let's just do the DP but with early termination
        
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        sums = [0] * (n + 1)
        sums[0] = 1  # sum 0 with 0 elements
        for v in vals:
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        
        # Check pairwise disjointness of sums[k] for different k
        # More efficient: accumulate a "seen" bitset
        seen = 0
        for i in range(n + 1):
            if sums[i] & seen:
                return False
            seen |= sums[i]
        return True

    def try_greedy_descending(pool, max_size):
        """Try to build a valid set by greedily adding elements from largest to smallest."""
        result = []
        for v in sorted(pool, reverse=True):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_greedy_ascending(pool, max_size):
        """Try to build a valid set by greedily adding elements from smallest to largest."""
        result = []
        for v in sorted(pool):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if check_valid(candidate):
                result = candidate
        return result

    def try_grow(initial, all_vals):
        """Grow a valid initial set by trying to add remaining values (smallest first)."""
        cur = list(initial)
        cur_set = set(cur)
        for v in sorted(all_vals):
            if v in cur_set:
                continue
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
        return cur

    def try_backtracking(pool, max_size, time_limit=5.0):
        """Backtracking search that tries to build a valid set of size max_size.
        Uses descending order as primary heuristic with backtracking."""
        pool = sorted(pool, reverse=True)
        n_pool = len(pool)
        best = []
        
        def backtrack(idx, current, depth):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(current) > len(best):
                best = list(current)
            if len(current) >= max_size:
                return
            if idx >= n_pool:
                return
            # Try adding pool[idx]
            cand = current + [pool[idx]]
            if check_valid(cand):
                backtrack(idx + 1, cand, depth + 1)
            # Try skipping pool[idx]
            backtrack(idx + 1, current, depth + 1)
        
        backtrack(0, [], 0)
        return best

    def try_swap_improve(cur, pool, max_size, time_limit=3.0):
        """Try to improve current set by swapping elements."""
        cur = list(cur)
        cur_set = set(cur)
        pool = [v for v in pool if v not in cur_set]
        pool.sort(reverse=True)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i, v_out in enumerate(sorted(cur, key=lambda x: -x)):  # try removing largest first
                for v_in in pool:
                    if v_in == v_out:
                        continue
                    cand = [x for x in cur if x != v_out] + [v_in]
                    if check_valid(cand) and len(cand) == len(cur):
                        # Check if this swap actually helps (e.g., allows adding more)
                        cur = cand
                        cur_set = set(cur)
                        pool = [v for v in pool if v not in cur_set]
                        improved = True
                        break
                if improved:
                    break
        return cur

    pool = list(range(1, N + 1))
    best = []
    
    # Strategy 1: Superincreasing baseline
    sup = [1]
    for i in range(1, 60):
        nxt = sum(sup) + 1
        if nxt <= N:
            sup.append(nxt)
        else:
            break
    if check_valid(sup):
        best = list(sup)
    
    # Strategy 2: Descending greedy
    desc = try_greedy_descending(pool, MAX_SIZE)
    if len(desc) > len(best):
        best = desc
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 3: Ascending greedy
    asc = try_greedy_ascending(pool, MAX_SIZE)
    if len(asc) > len(best):
        best = asc
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 4: Grow from superincreasing
    grown = try_grow(sup, pool)
    if len(grown) > len(best):
        best = grown
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 5: Backtracking search with time limit
    remaining = deadline - time.time()
    if remaining > 10:
        bt_time = min(10.0, remaining * 0.3)
        bt_result = try_backtracking(pool, MAX_SIZE, time_limit=bt_time)
        if len(bt_result) > len(best):
            best = bt_result
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 6: Try subsets of the pool - maybe not all values are useful
    remaining = deadline - time.time()
    if remaining > 5:
        for start in range(1, 50):
            if time.time() > deadline - 5:
                break
            sub_pool = list(range(start, N + 1))
            desc = try_greedy_descending(sub_pool, MAX_SIZE)
            if len(desc) > len(best):
                best = desc
    
    if time.time() > deadline:
        return [int(x) for x in best]
    
    # Strategy 7: Randomized attempts
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(42)
        iterations = min(100, int(remaining_time * 30))
        for _ in range(iterations):
            if time.time() > deadline - 1:
                break
            shuffled = pool[:]
            rng.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= MAX_SIZE:
                    break
                candidate = result + [v]
                if check_valid(candidate):
                    result = candidate
            if len(result) > len(best):
                best = result
    
    # Strategy 8: Random small seeds then grow
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(12345)
        for trial in range(min(50, int(remaining_time * 10))):
            if time.time() > deadline - 1:
                break
            size = rng.randint(1, 6)
            seed_vals = rng.sample(range(1, N + 1), size)
            if not check_valid(seed_vals):
                continue
            cur = try_grow(sorted(seed_vals), pool)
            if len(cur) > len(best):
                best = cur

    # Ensure we return a list of Python ints
    best = [int(x) for x in best]
    return best
