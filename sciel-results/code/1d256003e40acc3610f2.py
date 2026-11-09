import random
import time

def construct():
    # We want to maximize length.
    # Greedy usually gives a decent baseline.
    # Let's first run a fast greedy to get a baseline.
    def get_greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    greedy_seq = get_greedy()
    best_len = len(greedy_seq)
    best_seq = greedy_seq
    
    # Try to find longer sequences using randomized backtracking
    # Max possible length? 
    # Number of blocks for length k is k(k+1)/2.
    # Max sum is approx 256*k.
    # We need distinct sums in [1, 256*k].
    # So k(k+1)/2 <= 256*k => k+1 <= 512 => k <= 511.
    # But sums are not just bounded by max sum, they are specific values.
    # However, the "density" of sums is low.
    # Let's try up to 15.
    
    start_time = time.time()
    max_time = 55 # Leave 5s for safety
    
    # Heuristic for ordering candidates: 
    # Prefer numbers that are "far" from existing numbers?
    # Or just random.
    
    def is_valid_extension(current_seq, x):
        # current_seq is a list
        # Check if x can be appended
        # New sums: x, x+last, x+last+second_last, ...
        # We need to check if any of these are in the existing set of sums
        # AND if they collide with each other (they don't, since seq is increasing and positive, suffix sums are strictly increasing)
        # Wait, suffix sums of a positive increasing sequence are strictly increasing?
        # Suffixes: x, x+a_last, x+a_last+a_second_last...
        # Yes, they are strictly increasing. So internal collisions are impossible.
        # We only need to check against existing sums.
        
        # To be efficient, we should maintain the set of sums.
        # But in backtracking, we pass the set.
        
        pass # Placeholder, logic moved to backtrack

    def backtrack(seq, sums, last_val, target_len, node_count, max_nodes):
        # Returns (success, seq)
        if len(seq) == target_len:
            return True, seq
        
        if time.time() - start_time > max_time:
            return False, None
            
        # Heuristic: We need to pick the next number.
        # Candidates: last_val + 1 to 256.
        # To increase diversity, we can shuffle a subset or pick randomly.
        # But we must respect increasing order.
        
        # Optimization: If we pick a small number, we have more room for future numbers.
        # If we pick a large number, we might block future options.
        # Usually, smaller is better for packing, but for Sidon-like properties,
        # "sparse" is better.
        
        # Let's try a mix: 
        # 1. Try the smallest valid candidate (greedy-like)
        # 2. If that fails, try a random valid candidate.
        
        candidates = []
        # We can't iterate all 256 every time if it's slow.
        # But 256 is small.
        
        # Sort candidates by a heuristic?
        # Let's just iterate. But to allow backtracking to explore different branches,
        # we can randomize the order of candidates if we are "deep" or if we want diversity.
        
        # Actually, standard backtracking with fixed order is deterministic.
        # To get "randomized" behavior, we can shuffle the candidates list.
        
        # However, shuffling 256 items at every node is expensive.
        # Alternative: Use a global random seed and pick the "next" candidate based on a random offset?
        
        # Let's stick to a simple randomized restart strategy.
        # Each restart uses a different random seed for candidate selection.
        
        # Inside this specific backtrack call, we iterate candidates.
        # To make it "randomized", the caller (restart loop) will pass a flag or use a global random state.
        
        # For simplicity and speed, let's do this:
        # In each restart, we shuffle the range of candidates at the top level? No.
        
        # Let's use a simple heuristic: 
        # Try candidates in increasing order, but if we fail, we backtrack.
        # The "randomness" comes from restarting with different "biases".
        
        # Actually, a very effective trick is **Weighted Random Selection** of the next number.
        # But that's complex.
        
        # Let's try this: 
        # In each restart, we pick the next number uniformly at random from valid candidates?
        # No, that might be too slow.
        
        # Let's stick to: 
        # 1. Greedy Baseline.
        # 2. Backtracking with **Forward Checking** and **Randomized Restart**.
        
        # Implementation detail:
        # We will have a `backtrack` function that takes a `random_offset` or similar.
        # Or simpler: The outer loop runs `backtrack` multiple times.
        # Each time, we slightly perturb the search strategy.
        
        # Strategy: 
        # In `backtrack`, instead of iterating `x` from `last+1` to `256` in order,
        # we iterate in a shuffled order? 
        # Shuffling 256 ints is fast. Let's do that.
        
        candidates = list(range(last_val + 1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check validity
            new_sums = [x]
            s = x
            for a in reversed(seq):
                s += a
                new_sums.append(s)
                
            # Check intersection
            valid = True
            for ns in new_sums:
                if ns in sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            new_sums_set = sums.union(new_sums)
            # Recurse
            success, res_seq = backtrack(seq + [x], new_sums_set, x, target_len, node_count + 1, max_nodes)
            if success:
                return True, res_seq
            
        return False, None

    # Try lengths from best_len + 1 upwards
    for target_len in range(best_len + 1, 16):
        found = False
        # Do a few restarts for each target length
        for restart in range(50):
            random.seed(restart * 12345 + target_len)
            # Use a node limit to prevent hanging on a single path
            success, seq = backtrack([], set(), 0, target_len, 0, 50000)
            if success:
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq
                found = True
                break # Found a sequence of this length, try to find longer
            
        if not found:
            # If we failed to find a sequence of this length after 50 restarts,
            # it might be impossible or too hard. 
            # Given the constraints, if 12 is hard, 13 is likely harder.
            # But we should check time.
            if time.time() - start_time > max_time:
                break

    return best_seq
