"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    # The original greedy gives a baseline; let's run it first.
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best_seq = values[:]
    best_len = len(best_seq)

    # Now try randomized restarts with a smarter DFS/backtracking approach.
    # Strategy: pick elements in increasing order (to maintain strictly increasing),
    # but at each position try several candidate values, preferring smaller ones first
    # to leave room for longer sequences. Use randomized ordering to escape local optima.
    
    def try_build(max_steps=200000, seed=None):
        """Attempt to build a sequence using randomized greedy with backtracking hints."""
        if seed is not None:
            random.seed(seed)
        rng = random.Random(seed) if seed is not None else random
        
        for attempt in range(max_steps):
            if time.time() > deadline:
                break
            # Randomly shuffle the priority: sometimes prefer small, sometimes random
            bias = rng.random()
            if bias < 0.3:
                order = list(range(1, 257))
            elif bias < 0.6:
                order = list(range(1, 257))
                rng.shuffle(order)
                # Partially sort: keep small numbers somewhat prioritized
                order.sort(key=lambda x: rng.random() * x)
            else:
                order = list(range(1, 257))
                rng.shuffle(order)
            
            vals, sums = [], set()
            for x in order:
                if x <= (vals[-1] if vals else 0):
                    continue
                # Compute all new tail sums
                tails = [x]
                total = x
                for a in reversed(vals):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    vals.append(x)
                    sums.update(tails)
                    if len(vals) > best_len:
                        return vals[:]
        
        return best_seq
    
    # Run multiple randomized attempts
    base_seed = 42
    for i in range(50):
        if time.time() > deadline - 2.0:
            break
        result = try_build(max_steps=1000, seed=base_seed + i)
        if len(result) > best_len:
            best_len = len(result)
            best_seq = result
    
    # Also try a more structured approach: use gaps representation
    # Search over gap sizes (each gap >= 1) to maximize length
    # Total sum must be <= some bound; with max element 256, n elements means sum <= n*256
    # But we need all consecutive block sums distinct: there are n*(n+1)/2 blocks
    # Max sum of any block <= sum of all elements <= n * 256
    # So we need n*(n+1)/2 distinct sums, all in [min_element, total_sum]
    # This gives an upper bound on n.
    
    # Let's try a direct backtracking on the sequence with pruning
    def backtrack_dfs():
        """DFS building sequence left to right, trying values in increasing order."""
        nonlocal best_len, best_seq
        vals = []
        sums = set()
        
        def _dfs(idx, last_val, depth_limit):
            nonlocal best_len, best_seq
            if time.time() > deadline - 1.0:
                return
            if idx == depth_limit:
                if len(vals) > best_len:
                    best_len = len(vals)
                    best_seq = vals[:]
                return
            
            # Try candidate values
            start_val = last_val + 1
            # Upper bound: even with minimum gaps, total sum constraint
            remaining = depth_limit - idx
            # Each new element at least start_val, start_val+1, ...
            # Just try values; prune if no valid choice
            found = False
            for x in range(start_val, 257):
                # Compute tails
                tails = [x]
                total = x
                for a in reversed(vals):
                    total += a
                    tails.append(total)
                if not sums.intersection(tails):
                    vals.append(x)
                    new_sums = set(tails)
                    sums.update(new_sums)
                    found = True
                    _dfs(idx + 1, x, depth_limit)
                    sums.difference_update(new_sums)
                    vals.pop()
                    if time.time() > deadline - 1.0:
                        return
                if found and time.time() > deadline - 1.0:
                    return
            if not found and idx < depth_limit:
                # No valid value at this position; prune
                return
        
        # Try increasing depth limits
        current_depth = best_len + 1
        while current_depth <= 20 and time.time() < deadline - 2.0:
            _dfs(0, 0, current_depth)
            current_depth += 1
    
    # Run the DFS with a time limit
    if time.time() < deadline - 5.0:
        backtrack_dfs()
    
    return best_seq
# EVOLVE-BLOCK-END
