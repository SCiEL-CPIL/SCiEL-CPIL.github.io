import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Helper to check if adding x to seq creates distinct consecutive sums
    # Returns (is_valid, new_sums_to_add)
    def check_add(seq, x, sums):
        if seq and x <= seq[-1]:
            return False, set()
        
        # New sums are: x, x + last, x + last + second_last, ...
        # We need to check if any of these are in sums.
        # Also, the new sums themselves must be distinct (they are, because seq is strictly increasing and positive, so partial sums are strictly increasing).
        # Wait, if seq is [2, 5], x=3.
        # New sums: 3, 3+5=8, 3+5+2=10.
        # Are these distinct? Yes, because x > 0 and seq elements > 0.
        # So we only need to check against existing sums.
        
        new_sums = set()
        total = x
        if total in sums:
            return False, set()
        new_sums.add(total)
        
        for a in reversed(seq):
            total += a
            if total in sums:
                return False, set()
            new_sums.add(total)
            
        return True, new_sums

    # Helper to compute all consecutive sums for a sequence
    def compute_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    # Strategy 1: Greedy with specific ordering
    # Small numbers tend to be better for length because they allow more numbers to fit in [1, 256].
    # However, small numbers create small sums which collide easily.
    # Let's try a few different greedy strategies.
    
    best_seq = []
    
    # Strategy A: Greedy Ascending
    def greedy_asc():
        seq = []
        sums = set()
        for x in range(1, 257):
            valid, new_sums = check_add(seq, x, sums)
            if valid:
                seq.append(x)
                sums.update(new_sums)
        return seq

    # Strategy B: Greedy Descending
    def greedy_desc():
        seq = []
        sums = set()
        for x in range(256, 0, -1):
            # We need strictly increasing sequence, so we can't just pick descending.
            # This strategy is invalid for the "strictly increasing" constraint if we pick in descending order of value.
            # We must pick values such that seq is increasing.
            # So "Greedy Descending" means we consider larger values first? No, that violates increasing order.
            # We can only pick x if x > last.
            # So we can only iterate 1..256 or subsets.
            pass
        return seq

    # Strategy C: Randomized Greedy
    def greedy_random(seed):
        random.seed(seed)
        seq = []
        sums = set()
        available = list(range(1, 257))
        random.shuffle(available)
        
        # We must maintain increasing order.
        # This simple shuffle doesn't work directly because we need to pick the next element > last.
        # Better approach:
        # Pick a random start, then pick next from remaining > last.
        
        # Actually, let's stick to a modified greedy:
        # At each step, pick the smallest valid x > last? (That's Ascending Greedy)
        # Pick the largest valid x > last?
        
        def greedy_largest_last():
            seq = []
            sums = set()
            last = 0
            for x in range(1, 257):
                # We can't just iterate 1..256 and pick largest valid because
                # picking a large x early blocks many small x later.
                # But maybe it's better for sum distinctness?
                pass
            
            # Let's implement a proper "Greedy Largest Valid"
            seq = []
            sums = set()
            last = 0
            # We need to iterate through candidates.
            # If we pick 256 first, we can't pick anything else.
            # So we should pick small numbers first to leave room?
            # Or pick numbers that are "sparse"?
            
            return seq
        
        # Let's try a different randomization:
        # Shuffle the range, but enforce increasing order by filtering.
        # This is tricky.
        
        # Instead, let's use the "Ascending Greedy" as a base and then improve it.
        
        return greedy_asc()

    # Get baseline
    base_seq = greedy_asc()
    if len(base_seq) > len(best_seq):
        best_seq = base_seq[:]
    
    # Strategy 2: Local Search / Improvement
    # We have a valid sequence. Can we make it longer?
    # Ideas:
    # 1. Try to insert a skipped number.
    # 2. Try to replace a number with a smaller/larger one to free up space for others.
    # 3. Try to swap two numbers? (Hard to maintain distinct sums)
    
    def try_insert(seq, sums):
        # Find indices where we can insert a number
        # Let's try inserting every possible x in [1, 256] that is not in seq
        # and check if it fits between seq[i-1] and seq[i]
        
        in_seq = set(seq)
        improved = False
        new_seq = seq[:]
        new_sums = sums.copy()
        
        # We need to re-evaluate sums if we insert.
        # Inserting x at position i creates new sums:
        # x, x+next, x+next+nextnext... (suffixes starting at x)
        # ...prev+x, ...prev+prev2+x (prefixes ending at x)
        # And sums that span across x?
        # Actually, any consecutive block that includes x is new.
        # Blocks not including x are unchanged.
        
        # To be safe, let's just recompute sums for the whole sequence if we make a change.
        # It's O(N^2) which is fine for N~20.
        
        # Iterate through possible insertion points
        # We want to try inserting small numbers first? Or large?
        # Let's try all x not in seq.
        
        candidates = [x for x in range(1, 257) if x not in in_seq]
        random.shuffle(candidates) # Randomize order to avoid bias
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Find valid insertion positions
            # x must be > seq[i-1] and < seq[i] (with boundaries)
            # Let's check all positions 0..len(seq)
            
            for i in range(len(new_seq) + 1):
                # Check bounds
                if i > 0 and x <= new_seq[i-1]:
                    continue
                if i < len(new_seq) and x >= new_seq[i]:
                    continue
                
                # Try inserting x at i
                test_seq = new_seq[:i] + [x] + new_seq[i:]
                # Check validity
                test_sums = compute_sums(test_seq)
                if len(test_sums) == len(test_seq) * (len(test_seq) + 1) // 2:
                    # Valid
                    new_seq = test_seq
                    new_sums = test_sums
                    in_seq.add(x)
                    improved = True
                    break # Move to next candidate x? Or break out entirely?
                    # Breaking out of inner loop, continuing outer loop (next x)
            
            if time.time() > deadline:
                break
                
        return new_seq, improved

    # Strategy 3: Local Search Loop
    current_seq = best_seq[:]
    for _ in range(100): # Max improvements
        if time.time() > deadline:
            break
            
        new_seq, improved = try_insert(current_seq, compute_sums(current_seq))
        if not improved:
            # Try a randomized restart of the local search?
            # Or try a different initial greedy?
            
            # Let's try a "perturbation": remove a random element and try to insert different ones.
            if len(current_seq) > 2:
                idx = random.randint(0, len(current_seq)-1)
                perturbed_seq = current_seq[:idx] + current_seq[idx+1:]
                # Try to fill the gap
                new_seq2, improved2 = try_insert(perturbed_seq, compute_sums(perturbed_seq))
                if len(new_seq2) > len(current_seq):
                    current_seq = new_seq2
                    improved = True
                else:
                    # Maybe the perturbation itself was worse, ignore
                    pass
            else:
                break # Can't improve further with current method
            
        if improved:
            if len(new_seq) > len(current_seq):
                current_seq = new_seq
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq[:]
            else:
                # No length increase, might be stuck
                # If no length increase in last few iterations, break?
                # For now, just break if no improvement in length
                # Actually, try_insert only returns improved=True if length increased (because it adds an element)
                # So if improved is True, length increased.
                pass
        else:
            # Try a few random restarts of the greedy phase with different biases
            # Bias: Randomly skip some small numbers to see if it allows a longer sequence?
            # This is complex.
            
            # Let's try a simple random restart of the whole process with a different random seed for the insertion order
            # But try_insert already shuffles.
            
            # Maybe the greedy baseline is suboptimal.
            # Let's try a "Greedy with Random Skip"
            def greedy_random_skip(seed, skip_prob=0.1):
                random.seed(seed)
                seq = []
                sums = set()
                for x in range(1, 257):
                    if random.random() < skip_prob:
                        continue
                    valid, new_sums = check_add(seq, x, sums)
                    if valid:
                        seq.append(x)
                        sums.update(new_sums)
                return seq
            
            # Try a few random skips
            for s in range(10):
                if time.time() > deadline:
                    break
                cand_seq = greedy_random_skip(s, 0.2)
                if len(cand_seq) > len(current_seq):
                    current_seq = cand_seq
                    if len(current_seq) > len(best_seq):
                        best_seq = current_seq[:]
                    # And then try to improve this new base
                    new_seq, imp = try_insert(current_seq, compute_sums(current_seq))
                    if imp:
                        current_seq = new_seq
                        if len(current_seq) > len(best_seq):
                            best_seq = current_seq[:]
            
            # If we still can't improve, we might be at a local optimum.
            # Break to save time?
            # Let's do one more check: if we haven't improved in 5 iterations, break.
            # But we don't track iterations without improvement easily here.
            # Let's just break if no improvement found in this outer loop iteration.
            break

    return best_seq

# EVOLVE-BLOCK-END
