import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for the 60s limit

    def check_conflict(seq, x):
        """
        Check if adding x to the end of seq creates any duplicate consecutive block sums.
        Returns True if there is a conflict, False otherwise.
        Also returns the new tails sums if no conflict.
        """
        tails = [x]
        total = x
        for a in reversed(seq):
            total += a
            tails.append(total)
        return tails

    # 1. Greedy Baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails = check_conflict(values, x)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = values
    best_len = len(best_seq)
    
    # 2. Local Search / Hill Climbing to improve the baseline
    # The greedy approach is myopic. We can try to improve it by:
    # a) Removing an element and trying to fill the gap with a different one? No, sequence must be strictly increasing.
    # b) Actually, since the sequence is strictly increasing, we are selecting a subsequence of [1..256].
    # c) We can try to "repair" the greedy sequence or find a longer one via backtracking from the greedy prefix.
    
    # Strategy: Use the greedy sequence as a strong prior.
    # Try to extend it further if possible? The greedy already went through all numbers.
    # So the greedy sequence is maximal in the sense that no single additional number from the remaining pool (which is empty) can be added at the end.
    # But maybe a different choice earlier would allow a longer sequence.
    
    # Let's try a randomized restart search, but biased towards the greedy solution.
    # Or better: Backtracking search with pruning, starting from the greedy prefix.
    
    # Actually, a better strategy for "increasing sequence with unique consecutive block sums" (Consecutive Subset Sums distinct)
    # is often related to powers of 2 or similar exponential growth, but constrained to [1, 256].
    # The greedy algorithm usually picks 1, 2, 4, 8... but conflicts might force skips.
    
    # Let's implement a DFS that tries to beat the baseline length.
    # We will use the greedy sequence to guide the order of candidates.
    
    def dfs(current_seq, last_val, current_sums, depth, target_len):
        if time.time() > time_limit:
            return None
        
        if depth == target_len:
            return current_seq
        
        # Pruning: Can we even fit target_len - depth more elements?
        # Max possible length is 256 - last_val.
        if (target_len - depth) > (256 - last_val):
            return None
            
        # Heuristic: Try candidates that are in the greedy sequence first, then others.
        # Or just random shuffle but seeded?
        # Let's try to extend the greedy sequence if possible, or modify it.
        
        # A more effective approach: 
        # 1. Generate a few random valid sequences using a randomized greedy approach.
        # 2. Use local search to extend them.
        
        # Let's stick to a single DFS attempt for a specific target length, using a randomized candidate order.
        candidates = list(range(last_val + 1, 257))
        random.shuffle(candidates)
        
        # Bias: Put candidates that are "smaller" first? Or ones that appear in the greedy solution?
        # Greedy solution values are in best_seq.
        # If x is in best_seq and x > last_val, it's a good candidate.
        
        for x in candidates:
            tails = check_conflict(current_seq, x)
            if current_sums.intersection(tails):
                continue
            
            new_sums = current_sums.union(tails)
            result = dfs(current_seq + [x], x, new_sums, depth + 1, target_len)
            if result:
                return result
        
        return None

    # Try to find a sequence longer than the baseline
    # The baseline is likely quite long. Let's see its length.
    # For N=256, the greedy method usually produces a sequence of length around 10-15?
    # Let's check the baseline length first.
    
    # Actually, let's try a different approach: 
    # Construct the sequence using a "greedy with look-ahead" or simply a randomized restart of the greedy algorithm.
    
    # Randomized Greedy:
    # At each step, choose a random valid next element from the remaining pool, 
    # but bias towards smaller elements to leave room for more elements?
    # Or bias towards elements that are "compatible" with future elements?
    
    # Let's try multiple randomized greedy runs.
    
    def randomized_greedy():
        seq = []
        sums = set()
        available = set(range(1, 257))
        
        while available:
            # Filter available numbers that don't conflict
            valid_candidates = []
            for x in available:
                if x <= seq[-1] if seq else False:
                    continue
                tails = check_conflict(seq, x)
                if not sums.intersection(tails):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Strategy: Pick the smallest valid candidate? 
            # Or pick randomly?
            # Smallest usually allows for more future elements.
            # Let's try picking the smallest. This is the original greedy.
            # To vary, let's pick randomly from the bottom 20% of valid candidates.
            
            valid_candidates.sort()
            # Pick from the first few to keep numbers small
            k = min(len(valid_candidates), max(1, len(valid_candidates) // 5))
            choice = random.choice(valid_candidates[:k])
            
            seq.append(choice)
            sums.update(check_conflict(seq, choice))
            available.remove(choice)
            
        return seq

    best_len = len(best_seq)
    best_seq = best_seq[:]
    
    # Run several randomized greedy instances
    for _ in range(50):
        if time.time() > time_limit:
            break
        seq = randomized_greedy()
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    # Now, try to extend the best_seq found so far using backtracking?
    # Or just return the best found.
    
    # One more refinement: Try to extend the current best_seq by backtracking if it's short.
    # But randomized greedy already does a full pass.
    
    # Let's try a specific targeted search if best_len is small.
    # But given the constraints, randomized greedy is likely sufficient to find a good solution.
    # The theoretical maximum length for [1..N] is roughly log2(N) * something? 
    # For N=256, length might be around 10-12.
    
    return best_seq
