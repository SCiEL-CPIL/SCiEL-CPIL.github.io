"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0  # stay safely under 30 s
    best_S = None
    best_len = N + 1
    
    # Precompute all possible pairs? No, N=256 is small enough for dynamic checks.
    
    def update_state(S, T, addable_set, x):
        """
        Returns new_S, new_T, new_addable_set after adding x.
        S: list of current elements
        T: set of current sums
        addable_set: set of currently addable elements
        x: element to add
        """
        new_S = S + [x]
        new_sums = {2 * x}
        for s in S:
            new_sums.add(x + s)
        new_T = T | new_sums
        
        # Determine which elements in addable_set remain addable
        # An element y is addable if:
        # 1. 2*y not in new_T
        # 2. y+s not in new_T for all s in new_S
        # Note: We only need to check y in addable_set.
        
        # Optimization: 
        # y is blocked if y in blocked_set.
        # blocked by 2*y: y = t/2 for t in new_sums? No, 2*y in new_T.
        # blocked by y+s: y = t - s for t in new_sums, s in new_S?
        
        # Simpler: iterate through addable_set and check.
        # |addable_set| <= 256. |new_S| grows.
        # Total work per step is O(|addable| * |S|).
        # Sum of |S| over steps is O(K^2). Total O(K^2 * N).
        # K is ~15-20. 20^2 * 256 ~ 100k ops. Very fast.
        
        new_addable = set()
        S_set = set(new_S)
        for y in addable_set:
            if 2 * y in new_T:
                continue
            ok = True
            for s in new_S:
                if y + s in new_T:
                    ok = False
                    break
            if ok:
                new_addable.add(y)
        
        return new_S, new_T, new_addable

    def run_greedy_randomized(temp_start, temp_end, seed):
        random.seed(seed)
        S = []
        T = set()
        addable = set(range(1, N + 1))
        
        steps = 0
        # Estimate max steps: N
        while addable and steps < N:
            steps += 1
            # Calculate scores for each candidate
            candidates = list(addable)
            if not candidates:
                break
            
            # If temperature is low, pick best. If high, pick random.
            # Score = -len(remaining_addable) i.e., maximize reduction.
            # To save time, we only compute exact impact for top candidates or all if small.
            # Let's compute impact for all, it's fast enough.
            
            best_score = -1
            best_x = None
            scores = []
            
            # Optimization: We can pre-compute which elements are "dangerous"
            # But let's just do the check.
            
            for x in candidates:
                # Simulate adding x
                # We don't need full new_S, just the new sums and the check for addability.
                new_sums = {2 * x}
                for s in S:
                    new_sums.add(x + s)
                
                # Check which elements in addable remain
                # y is blocked if 2y in new_T (which is T | new_sums) 
                # or y+s in new_T for s in S U {x}
                
                # 2y in T or 2y in new_sums
                # y+s in T or y+s in new_sums
                
                # Let's count how many y in addable are still valid.
                valid_count = 0
                for y in addable:
                    if y == x: continue # x is removed from addable anyway
                    if 2 * y in T or 2 * y in new_sums:
                        continue
                    ok = True
                    for s in S:
                        if (y + s) in T or (y + s) in new_sums:
                            ok = False
                            break
                    if ok:
                        valid_count += 1
                # We want to MINIMIZE valid_count (maximize blocked)
                score = -valid_count
                scores.append((score, x))
            
            # Sort by score (ascending, so most negative / max reduction first)
            scores.sort()
            
            # Selection strategy based on temperature
            # We can use a simple "pick from top K" or softmax.
            # Let's use a simple heuristic: 
            # If temp > 0.5, pick uniformly from top 5.
            # If temp < 0.5, pick the best.
            # Or just pick best if temp is low.
            
            if temp_start == temp_end: # deterministic
                best_x = scores[0][1]
            else:
                # Simple randomization: pick randomly from the top 10% or top 5
                top_k = max(1, min(5, len(scores)))
                # Weighted random choice from top_k?
                # Let's just pick randomly from top_k to explore.
                idx = random.randint(0, top_k - 1)
                best_x = scores[idx][1]
            
            # Update state
            S, T, addable = update_state(S, T, addable, best_x)
            
            # Annealing
            # We don't really need complex annealing if we restart.
            # But let's decrease temp.
            # Actually, for a single run, we can just use a fixed strategy: 
            # "Greedy with noise".
            
        return S

    # Strategy: Multiple randomized restarts.
    # Each run is O(K^2 * N). K~20, N=256. ~100k ops.
    # 1000 runs ~ 100M ops. Python might be too slow for 1000 runs in 25s.
    # Let's estimate: 100k ops per run. 1000 runs = 100M ops.
    # Python does ~10M-50M simple ops/sec. 100M might take 2-10s.
    # We can afford ~500-1000 runs.
    
    # To speed up, we can optimize `update_state` and the inner loop.
    # The inner loop `for y in addable: for s in S:` is the bottleneck.
    
    # Let's try 200 runs first.
    
    for i in range(300):
        if time.time() > deadline - 2:
            break
        
        S = run_greedy_randomized(0, 0, i) # Deterministic? 
        # Wait, my run_greedy_randomized doesn't use temp correctly if I pass 0,0.
        # I need to pass a flag or handle randomness inside.
        
        # Let's modify run_greedy_randomized to take a 'randomize' flag.
        # Actually, I'll just rewrite the inner loop to be efficient and randomized.

    # Re-implementing the loop for efficiency and randomization
    def run_trial(seed):
        random.seed(seed)
        S = []
        T = set()
        addable = set(range(1, N + 1))
        
        while addable:
            candidates = list(addable)
            if not candidates:
                break
            
            # Calculate "impact" (number of elements blocked)
            # We want to pick x that blocks the most.
            # Impact(x) = |addable| - |addable_after_x|
            
            # Optimization: 
            # Instead of checking all y for all x, we can precompute conflicts?
            # For now, just do it.
            
            best_x = None
            min_remaining = len(addable) + 1
            
            # Shuffle candidates to break ties randomly
            random.shuffle(candidates)
            
            for x in candidates:
                # Check how many remain
                # 2*x is in new sums
                # x+s is in new sums
                
                # An element y is blocked if:
                # 1. 2y == 2x => y=x (already handled by removing x)
                # 2. 2y == x+s => y = (x+s)/2
                # 3. y+s' == 2x => y = 2x - s'
                # 4. y+s' == x+s => y = x + s - s'
                
                # This algebraic approach is faster than set lookups?
                # Maybe. But set lookups are O(1).
                
                # Let's stick to set logic but optimize the loop.
                new_sums = {2 * x}
                for s in S:
                    new_sums.add(x + s)
                
                # Count valid y
                valid = 0
                for y in addable:
                    if y == x:
                        continue
                    if 2 * y in new_sums or 2 * y in T:
                        continue
                    ok = True
                    for s in S:
                        if (y + s) in new_sums or (y + s) in T:
                            ok = False
                            break
                    if ok:
                        valid += 1
                
                if valid < min_remaining:
                    min_remaining = valid
                    best_x = x
                    # Early exit if 0?
                    if valid == 0:
                        break
            
            if best_x is None:
                break
                
            # Update
            new_sums = {2 * best_x}
            for s in S:
                new_sums.add(best_x + s)
            T.update(new_sums)
            S.append(best_x)
            
            # Update addable
            new_addable = set()
            for y in addable:
                if y == best_x:
                    continue
                if 2 * y in T:
                    continue
                ok = True
                for s in S:
                    if y + s in T:
                        ok = False
                        break
                if ok:
                    new_addable.add(y)
            addable = new_addable
            
        return S

    # Run trials
    num_trials = 500
    for i in range(num_trials):
        if time.time() > deadline - 2:
            break
        S = run_trial(i)
        if len(S) < best_len:
            best_len = len(S)
            best_S = S
            # If we hit a theoretical lower bound? 
            # Minimal maximal Sidon set size for N=256 is likely around 15-20.
            # If we find 15, we can stop early?
            # Let's check if best_len <= 15: break

    if best_S is None:
        # Fallback: simple greedy
        S = []
        T = set()
        for x in range(1, N + 1):
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not T.intersection(added):
                S.append(x)
                T.update(added)
        return S
        
    return best_S
# EVOLVE-BLOCK-END
