import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s kill
    
    best_seq = []
    best_len = 0
    
    # Helper to check if a sequence is valid (for verification if needed, but we construct valid ones)
    def is_valid(seq):
        if not seq: return True
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)): return False
        if any(x < 1 or x > 256 for x in seq): return False
        sums = set()
        prefix = 0
        for i in range(len(seq)):
            prefix += seq[i]
            # Check all blocks ending at i
            for j in range(i + 1):
                s = prefix - (sum(seq[:j]) if j > 0 else 0) # This is inefficient, better to use prefix sums array
        # Efficient check:
        p = [0]
        for x in seq:
            p.append(p[-1] + x)
        sums = set()
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                s = p[j] - p[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        
        seq = []
        prefix_sums = [0]
        # Use a set for O(1) lookup of forbidden sums
        forbidden_sums = set()
        
        # We want to pick numbers from 1 to 256.
        # To allow randomization, we can shuffle the candidate pool or pick randomly from valid candidates.
        # Strategy: Iterate through potential next numbers.
        # To make it randomized, we can try to pick from a range of "small" valid numbers first, 
        # but occasionally jump to larger ones if stuck? 
        # Simpler: Shuffle the entire list of candidates 1..256? No, we need strictly increasing.
        # So we must pick x > last.
        
        # Better Randomized Greedy:
        # At each step, find all valid x > last.
        # Pick one randomly from the valid set? Or pick the smallest with some probability?
        # Picking the smallest is usually best for packing. 
        # Randomizing the *order* of checking doesn't help if we always pick the first valid one.
        # We need to sometimes skip a valid small number to allow a better global configuration.
        
        # Approach: 
        # 1. Find all valid x > last.
        # 2. If no valid x, break.
        # 3. With probability 0.8, pick the smallest valid x.
        # 4. With probability 0.2, pick a random valid x from the top 20% of the valid range?
        
        last = 0
        for _ in range(256):
            valid_candidates = []
            # We can limit the search range for speed, but 256 is small.
            for x in range(last + 1, 257):
                new_prefix = prefix_sums[-1] + x
                # Check if new differences collide with forbidden_sums
                # Differences are new_prefix - p for all p in prefix_sums
                # If any difference is in forbidden_sums, x is invalid.
                # Also, the new differences themselves must be unique among themselves? 
                # No, if d1 = new_prefix - p1 and d2 = new_prefix - p2, since p1 != p2, d1 != d2.
                # So we only need to check against existing forbidden_sums.
                
                is_valid_candidate = True
                for p in prefix_sums:
                    d = new_prefix - p
                    if d in forbidden_sums:
                        is_valid_candidate = False
                        break
                
                if is_valid_candidate:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Selection Strategy
            # To maximize length, smaller numbers are generally better.
            # But sometimes picking a slightly larger number now allows more numbers later.
            # Let's try a mix:
            # 80% chance: pick smallest (valid_candidates[0])
            # 20% chance: pick random from valid_candidates
            
            if random.random() < 0.8:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)
            
            # Commit x
            seq.append(x)
            last = x
            new_prefix = prefix_sums[-1] + x
            prefix_sums.append(new_prefix)
            
            # Update forbidden_sums
            for p in prefix_sums[:-1]:
                forbidden_sums.add(new_prefix - p)
            
            # Early termination if we can't add more? 
            # The loop will just break if no valid candidates.
            
        return seq

    # Run multiple randomized greedy attempts
    # Each attempt is O(N^2 * K) where K is number of checks. N~20, K~20. Very fast.
    # We can run thousands of these.
    
    attempt_count = 0
    while time.time() - start_time < time_limit:
        seq = greedy_construct(seed=attempt_count)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            # Optional: If we reach a very high length, we could try to improve it with local search,
            # but for now, just keep finding better greedy sequences.
            # A length of ~20-25 is expected.
        
        attempt_count += 1
        
        # If we've been running for a while and found a good solution, 
        # we might want to stop randomizing and try to extend the best solution found?
        # Greedy is deterministic given the seed. Different seeds give different results.
        
        # To ensure we don't miss the deterministic greedy (which is often good), 
        # we already ran it implicitly if seed=0 led to the "smallest" path? 
        # Actually, seed=0 with 80% smallest bias is close to pure greedy.
        
        # Let's also try a pure greedy (always smallest) to ensure we have a strong baseline.
        if attempt_count == 1:
             # Force pure greedy logic by temporarily changing the selection? 
             # Or just run one specific seed that is known to be good?
             # The loop above handles it.
             pass

    return best_seq
