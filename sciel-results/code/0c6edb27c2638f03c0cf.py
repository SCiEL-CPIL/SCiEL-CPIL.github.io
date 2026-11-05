"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 55.0  # 5s buffer before 60s limit
    
    def get_valid_candidates(values, sums, last_val):
        """
        Returns a list of all x in (last_val, 256] such that appending x 
        to values does not create any duplicate consecutive block sums.
        """
        valid = []
        # We need to check x from last_val + 1 to 256
        for x in range(last_val + 1, 257):
            # Calculate new tails (suffix sums ending at x)
            # tails[0] = x
            # tails[1] = x + values[-1]
            # tails[2] = x + values[-1] + values[-2]
            # ...
            tails = [0] * (len(values) + 1)
            total = x
            tails[0] = total
            for i in range(len(values) - 1, -1, -1):
                total += values[i]
                tails[len(values) - i] = total
            
            # Check for collision
            # If any tail is in sums, collision
            if not sums.intersection(tails):
                valid.append(x)
        
        return valid

    def greedy_increasing():
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            # Greedy: pick smallest
            x = candidates[0]
            values.append(x)
            last_val = x
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        return values

    def greedy_decreasing():
        """
        Construct a strictly decreasing sequence b_1 > b_2 > ... > b_k
        such that consecutive sums are distinct.
        Then reverse it to get increasing a_1 < ... < a_k.
        The set of consecutive sums is invariant under reversal.
        """
        values = []
        sums = set()
        next_val = 257
        while next_val > 1:
            candidates = []
            for x in range(next_val - 1, 0, -1):
                # Check if x can be appended to the *right* of current decreasing sequence?
                # Wait, if we build decreasing b_1, b_2, ...
                # b_1 is largest.
                # Next b_2 < b_1.
                # The new sums added by appending b_2 to [b_1] are:
                # b_2, b_2+b_1.
                # This is the same logic as increasing, just values are decreasing.
                # Let's just reuse the logic by mapping values to 257-x?
                # No, simpler: just run the same candidate check but iterating downwards.
                
                tails = [0] * (len(values) + 1)
                total = x
                tails[0] = total
                for i in range(len(values) - 1, -1, -1):
                    total += values[i]
                    tails[len(values) - i] = total
                
                if not sums.intersection(tails):
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Greedy decreasing: pick largest valid (which is candidates[0] since we iterated down? 
            # No, range(next_val-1, 0, -1) goes 256, 255...
            # So candidates[0] is the largest valid.
            x = candidates[0]
            values.append(x)
            next_val = x
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        
        # Reverse to get increasing sequence
        return values[::-1]

    def randomized_trial(strategy, rand):
        values = []
        sums = set()
        last_val = 0
        
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            
            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                # Pick from the largest 10% or top 10
                k = min(10, len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                # Pick from the smallest 10% or top 10
                k = min(10, len(candidates))
                x = rand.choice(candidates[:k])
            else:
                x = rand.choice(candidates)
                
            values.append(x)
            last_val = x
            
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
                
        return values

    # 1. Baselines
    best_seq = greedy_increasing()
    best_len = len(best_seq)
    
    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)
        
    # 2. Randomized Search
    # We use a mix of strategies to explore the space
    strategies = ['random', 'large', 'small', 'random', 'large']
    
    trial_count = 0
    while time.time() - start_time < time_limit:
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count)
        
        # To ensure we don't spend too much time on one trial if it gets stuck in a long dead end
        # The greedy step is fast, so this is less of an issue, but we check time.
        
        seq = randomized_trial(strategy, rand)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            # If we found a significantly longer sequence, we might want to restart or continue?
            # Continue, but maybe bias towards strategies that worked?
            # For now, just continue.
            
        trial_count += 1
        
        # Small sleep to avoid CPU spin if trials are extremely fast? 
        # No, Python overhead is significant, trials are not microsecond-fast.
        
    return best_seq

# EVOLVE-BLOCK-END
