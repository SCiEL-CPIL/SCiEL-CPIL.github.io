import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # The universe of numbers is [2, 128], so 127 elements.
    # We can represent the set of "blocked" numbers or "used" numbers as a bitmask.
    # However, the constraint is on products of consecutive blocks.
    # Let's analyze the constraint:
    # Sequence a_1, a_2, ..., a_k.
    # Products: P(i, j) = a_i * a_{i+1} * ... * a_j for 1 <= i <= j <= k.
    # All P(i, j) must be distinct.
    # Note that P(i, j) = P(i, j-1) * a_j.
    # Also P(i, j) = P(i+1, j) * a_i.
    
    # A key observation for greedy construction:
    # When appending a new number x to the current sequence S, we generate new products:
    # x, S_last * x, S_last-1 * S_last * x, ..., S_1 * ... * S_last * x.
    # Let the current suffix products be suffix_prods = [p_1, p_2, ..., p_m] where p_1 is the last element, p_2 is last two, etc.
    # Actually, it's easier to keep the list of products ending at the current last element.
    # Let current_suffix_prods be a list where current_suffix_prods[i] is the product of the last i+1 elements.
    # Wait, standard greedy:
    # Maintain a set of all products seen so far: `all_prods`.
    # Maintain the current sequence `seq`.
    # To add x:
    #   new_prods = [x]
    #   prod = x
    #   for a in reversed(seq):
    #       prod *= a
    #       new_prods.append(prod)
    #   If any p in new_prods is in `all_prods`, x cannot be added.
    #   Otherwise, add x to seq, update `all_prods` with new_prods, and update the suffix products for the next step.
    #   Actually, the "suffix products" for the next step are exactly `new_prods` reversed?
    #   Let's check.
    #   If seq = [a, b], suffix prods for next addition x are:
    #   x, b*x, a*b*x.
    #   These are the new_prods generated when adding x.
    #   So, we can just keep the list of new_prods as the "current suffix products" for the next iteration.
    #   Let's call it `cur_suffix`.
    #   Initially cur_suffix = [].
    #   To add x:
    #       new_suffix = [x]
    #       prod = x
    #       for p in cur_suffix: # p is the product of the last k elements. Multiplying by x gives product of last k+1 elements ending in x?
    #           No. cur_suffix contains products ending at the current last element.
    #           Let cur_suffix = [s_1, s_2, ..., s_m] where s_i is product of last i elements.
    #           If we append x, the new suffixes are:
    #           x (length 1)
    #           s_1 * x (length 2)
    #           s_2 * x (length 3)
    #           ...
    #           s_m * x (length m+1)
    #           So new_suffix = [x] + [s * x for s in cur_suffix]
    #       Check if any element in new_suffix is in `all_prods`.
    #       If not, `all_prods.update(new_suffix)`, `seq.append(x)`, `cur_suffix = new_suffix`.
    
    # This is O(L^2) per addition if we store all prods in a set, but L is small (max 127).
    # The number of products is L*(L+1)/2. For L=100, that's 5050 products.
    # Storing them in a set is fast.
    
    # The main bottleneck in the previous program was likely the DFS exploring too many branches or being slow due to Python overhead.
    # Randomized greedy is O(N^3) roughly (N candidates * N steps * N suffix check). 127^3 ~ 2M ops per run.
    # We can do many runs in 50s.
    
    best_seq = []
    best_len = 0
    
    # Pre-compute something? No, just run multiple greedy passes with different random orders.
    # Strategy:
    # 1. Start with an empty sequence.
    # 2. Have a pool of candidates [2..128].
    # 3. At each step, pick a candidate x from the pool such that it can be appended.
    #    To diversify, we can shuffle the pool initially, or pick randomly from valid candidates.
    #    Or, we can try a "greedy with lookahead" or just multiple random restarts.
    #    Let's try multiple restarts where we shuffle the candidate pool and pick the first valid one?
    #    No, that's just one specific order.
    #    Better: At each step, collect ALL valid candidates, then pick one randomly.
    #    This allows exploring different paths.
    
    # Even better: "Greedy with random tie-breaking" or "Randomized Hill Climbing"?
    # Let's stick to "Randomized Greedy Construction" with multiple restarts.
    
    def run_greedy(seed):
        rng = random.Random(seed)
        candidates = list(range(2, 129))
        rng.shuffle(candidates)
        
        seq = []
        all_prods = set()
        cur_suffix = []
        
        # We iterate through the shuffled candidates? No, that restricts us to that order.
        # We should pick the "best" or "random valid" from the remaining pool.
        # If we just shuffle and take the first valid, we might miss a better one later in the shuffle.
        # But since we want density, any valid move is good.
        # Let's do this:
        # While there are candidates left:
        #   Find all x in candidates that can be added.
        #   If none, break.
        #   Pick one randomly from the valid ones.
        #   Add it.
        #   Remove it from candidates.
        
        # To make it faster, we can pre-filter?
        # Checking validity is O(L). L grows.
        # Number of steps is at most 127.
        # In each step, we check up to 127 candidates.
        # Total checks per run ~ 127 * 127 = 16k.
        # Each check is O(L) ~ 100.
        # Total ops per run ~ 1.6M.
        # 50s / 1.6M ops... Python does ~10M-50M simple ops per second.
        # So ~1-5 runs per second? Maybe less due to overhead.
        # Let's optimize the validity check.
        
        def can_add(x, cur_suffix, all_prods):
            # Generate new suffixes
            # new_suffix = [x] + [s * x for s in cur_suffix]
            # Check intersection with all_prods
            # We can do this without creating the list if we want to be faster, but list creation is cheap.
            new_suffix = [x]
            prod = x
            for s in cur_suffix:
                prod = s * x
                new_suffix.append(prod)
                if prod in all_prods:
                    return False
            if x in all_prods:
                return False
            # If we got here, no conflict
            return True

        # Wait, the loop above checks `prod in all_prods` inside, but `x in all_prods` is separate.
        # Let's refine can_add to be as fast as possible.
        
        def is_valid(x, cur_suffix, all_prods):
            # Check x
            if x in all_prods:
                return False, []
            
            new_suffix = [x]
            prod = x
            for s in cur_suffix:
                prod = s * x
                new_suffix.append(prod)
                if prod in all_prods:
                    return False, []
            
            return True, new_suffix

        # Main loop of one greedy run
        remaining = set(candidates)
        
        while remaining:
            valid_moves = []
            # Iterate over remaining candidates
            # To make it deterministic per seed but random in choice, we can shuffle the list of remaining at each step?
            # Or just pick a random subset?
            # Let's just iterate through a shuffled list of remaining candidates and pick the first valid one?
            # No, that biases towards the first valid in the shuffle.
            # Let's collect all valid moves, then pick one.
            
            # Optimization: If we just want *a* long sequence, picking any valid move is fine.
            # But some moves might be "better" (block fewer future moves).
            # For now, random choice from valid moves.
            
            # To avoid O(N^2) scanning if we pick randomly, we can just shuffle the remaining list once at the start?
            # If we shuffle once, we are committed to that order of *consideration*.
            # If we pick the first valid one in that fixed order, it's a specific greedy strategy.
            # Let's try: Shuffle candidates once. Iterate through them. If valid, add. Else, keep for later?
            # No, if we skip, we might come back to it.
            
            # Strategy A: Fixed random order, pick first valid.
            # Strategy B: At each step, scan all remaining, collect valid, pick random.
            
            # Strategy B is more robust.
            # Let's implement Strategy B.
            
            # To speed up, we can pre-calculate something? No.
            
            # Let's try a hybrid: 
            # 1. Shuffle remaining candidates.
            # 2. Iterate through them.
            # 3. Keep a list of valid candidates found in this pass.
            # 4. If we find one, pick it randomly from the found list? 
            #    Or just pick the first one found?
            #    Picking the first one found in a random order is equivalent to random selection if the order is uniform.
            #    Yes! If we shuffle the remaining candidates, the probability that a specific valid candidate is the "first valid" is proportional to its position, but since the shuffle is uniform, any valid candidate has an equal chance of being the first one encountered IF we stop at the first valid.
            #    Wait. If I have candidates [A, B, C] and A is invalid, B is valid, C is valid.
            #    Shuffle: [B, A, C]. First valid is B.
            #    Shuffle: [C, A, B]. First valid is C.
            #    Shuffle: [A, B, C]. First valid is B.
            #    It seems B is picked more often?
            #    No, the set of valid candidates is fixed. The shuffle determines the order.
            #    The first valid candidate in a random permutation of the set of all remaining candidates is uniformly distributed among the valid candidates?
            #    Let V be set of valid, I be set of invalid.
            #    We want P(x is first valid).
            #    This is true if we just pick the first valid in a random permutation.
            
            # So:
            # 1. Shuffle remaining list.
            # 2. Loop through list.
            # 3. Check validity.
            # 4. If valid, choose it, break loop.
            # 5. If no valid found, break outer loop.
            
            shuffled_rem = list(remaining)
            rng.shuffle(shuffled_rem)
            
            chosen = None
            new_suffix_val = []
            
            for x in shuffled_rem:
                ok, ns = is_valid(x, cur_suffix, all_prods)
                if ok:
                    chosen = x
                    new_suffix_val = ns
                    break
            
            if chosen is None:
                break
            
            # Commit to chosen
            seq.append(chosen)
            all_prods.update(new_suffix_val)
            cur_suffix = new_suffix_val
            remaining.remove(chosen)
            
            # Time check
            if time.time() > deadline:
                break
                
        return seq

    # Run multiple greedy instances with different seeds
    # How many can we run?
    # Let's estimate time per run.
    # 127 steps.
    # Each step shuffles up to 127 items (fast).
    # Checks validity of up to 127 items.
    # Validity check is O(L). L is avg 60.
    # 127 * 127 * 60 ~ 1M operations per run.
    # Python overhead for set lookup is high.
    # Let's say 0.05s per run.
    # 50s / 0.05s = 1000 runs.
    # Let's run as many as possible.
    
    num_runs = 0
    while time.time() < deadline - 5.0:
        seed = num_runs
        seq = run_greedy(seed)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
        num_runs += 1
        
        # Optional: If we find a very long sequence (e.g., > 40), we might try to improve it with local search?
        # For now, just greedy restarts.
        
    return best_seq

