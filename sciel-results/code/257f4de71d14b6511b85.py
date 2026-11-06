import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_len = 0
    best_seq = []
    
    # Pre-compute log products for fast comparison? 
    # Actually, integer products are exact. 
    # To speed up, we can maintain the set of products.
    # But set intersection is O(N). 
    # With N ~ 100, N^2 is 10k operations. Very fast.
    # The bottleneck is the number of restarts/improvements.
    
    def get_products(seq):
        """Compute all consecutive block products for a sequence."""
        n = len(seq)
        if n == 0:
            return set()
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
        prods = get_products(seq)
        return len(prods) == len(seq) * (len(seq) + 1) // 2

    def greedy_construct(start_val=2, end_val=129):
        """Original greedy approach, used as a baseline or starting point."""
        values = []
        products = set()
        for x in range(start_val, end_val):
            tails = []
            prod = x
            # Check if adding x creates duplicate products
            # New products are x * (suffix products of values)
            # Specifically: x, x*last, x*last*second_last, ...
            # We need to check if any of these are in 'products'
            # AND if any of these are equal to each other (impossible since x>1 and values strictly increasing? 
            # Wait, x * a = x * b implies a=b. So new tails are distinct.
            # We only need to check intersection with existing 'products'.
            
            # Optimization: 
            # Existing products are all blocks in 'values'.
            # New products are blocks ending at x.
            # We just need to check if any new product is in 'products'.
            
            # Let's stick to the simple set intersection for clarity, 
            # but we can optimize by only checking the new tails against the old set.
            
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    # Initial baseline
    base_seq = greedy_construct()
    best_len = len(base_seq)
    best_seq = base_seq
    
    if time.time() > deadline - 5:
        return best_seq

    # Stochastic Local Search
    # Strategy: 
    # 1. Start with a random subset or a perturbed greedy sequence.
    # 2. Try to insert missing numbers or swap numbers to increase length/validity.
    # 3. Or, more simply: Randomized Greedy. 
    #    Shuffle the candidate range, then greedily add? No, order matters for products.
    #    The sequence must be strictly increasing. So we are selecting a subsequence of [2,128].
    #    The "order" is fixed by value. We are just choosing which values to include.
    #    
    #    So the problem is: Select a subset S of {2, ..., 128} such that 
    #    all products of consecutive elements in the sorted S are distinct.
    #    Maximize |S|.
    
    # This is a subset selection problem.
    # Greedy: Iterate 2..128, include if valid.
    # Randomized: 
    #   1. Start with empty set.
    #   2. Iterate 2..128.
    #   3. With probability p, include x if valid. 
    #      But this might miss early small numbers which are "cheaper" in terms of product growth?
    #      Actually, small numbers produce smaller products, which might collide with products of larger numbers?
    #      Or maybe larger numbers are "safer" because their products grow faster and are less likely to collide?
    
    # Let's try a different heuristic:
    # "Backward Greedy": Start from 128 and go down?
    # Or "Randomized Restart with Perturbation":
    # Take the best sequence found so far, remove a random element, and re-greedy fill?
    
    # Better: Randomized Greedy Construction
    # Instead of always taking the smallest valid number, sometimes skip a valid number to allow a larger one?
    # Or, consider the "conflict" graph.
    
    # Let's implement a simple Randomized Local Search:
    # 1. Current sequence S.
    # 2. Generate a neighbor by removing one element from S and inserting a random missing element?
    #    This is complex.
    
    # Simpler: Multiple Randomized Greedy Passes
    # The greedy algorithm's choice at step i depends on previous choices.
    # We can introduce randomness: when multiple choices are available? 
    # No, the choice is binary: include x or not. 
    # But the order is fixed. 
    # So we can randomize the "threshold" or use a different ordering? 
    # No, the sequence must be increasing integers.
    
    # What if we try to construct the sequence from the END?
    # Pick x128, then x127...
    # Products are the same regardless of direction.
    
    # Let's try: Randomized Greedy with "Lookahead" or "Random Skipping"
    # Idea: Sometimes skip a valid number to see if it allows more numbers later?
    # This is hard to evaluate locally.
    
    # Alternative: Simulated Annealing on the subset?
    # State: Bitmask of included numbers (127 bits).
    # Move: Flip a bit.
    # Cost: Number of duplicate products. We want 0 duplicates and max length.
    # Objective: Maximize Length - LargePenalty * Duplicates.
    
    # This is promising. 
    # Validity check (count duplicates) is O(N^2). N=128. 128^2 = 16k ops.
    # We can do ~1000 moves per second? Maybe less in Python. 
    # 50s * 1000 = 50k moves.
    # Enough to explore some local minima.
    
    def evaluate(seq):
        """Returns (length, num_duplicates)."""
        if not seq:
            return 0, 0
        prods = []
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.append(p)
        # Count duplicates
        unique = set(prods)
        duplicates = len(prods) - len(unique)
        return len(seq), duplicates

    def random_init():
        # Random subset of [2, 128]
        # Bias towards including numbers?
        # Or just random 50% probability?
        seq = []
        for x in range(2, 129):
            if random.random() < 0.5:
                seq.append(x)
        return seq

    def local_search(seq, time_limit=1.0):
        """Perform local search starting from seq."""
        start = time.time()
        current_seq = seq[:]
        current_len, current_dups = evaluate(current_seq)
        current_score = current_len - 100 * current_dups
        
        # We want to maximize length, so we prioritize removing duplicates first, then adding length.
        # Actually, if dups > 0, it's invalid. 
        # We can define score as:
        # If dups == 0: score = length
        # Else: score = -dups (or similar)
        
        # Let's use: score = length if valid, else -dups
        def get_score(seq):
            l, d = evaluate(seq)
            if d == 0:
                return l
            else:
                return -d
        
        current_score = get_score(current_seq)
        
        while time.time() < start + time_limit:
            # Generate neighbor: flip a random bit
            # Options: 
            # 1. Remove a random element from current_seq.
            # 2. Add a random element from [2,128] not in current_seq.
            
            if current_dups > 0:
                # Try to remove an element involved in a duplicate? 
                # Or just random removal?
                # Random removal is safer to avoid getting stuck.
                if current_seq:
                    idx = random.randint(0, len(current_seq) - 1)
                    new_seq = current_seq[:idx] + current_seq[idx+1:]
                else:
                    break
            else:
                # Try to add an element
                missing = [x for x in range(2, 129) if x not in current_seq]
                if not missing:
                    break
                x = random.choice(missing)
                # Insert in sorted position
                # Since we want increasing order, we can just append and sort?
                # Or use bisect.
                import bisect
                pos = bisect.bisect_left(current_seq, x)
                new_seq = current_seq[:pos] + [x] + current_seq[pos:]
            
            new_len, new_dups = evaluate(new_seq)
            new_score = get_score(new_seq)
            
            if new_score > current_score:
                current_seq = new_seq
                current_len = new_len
                current_dups = new_dups
                current_score = new_score
            # Else: accept with some probability? 
            # For simplicity, let's stick to Hill Climbing (only accept improvements).
            # This might get stuck. 
            # Let's add some randomness: if new_score == current_score, accept with 50%?
            elif new_score == current_score and random.random() < 0.5:
                current_seq = new_seq
                current_len = new_len
                current_dups = new_dups
                current_score = new_score

        return current_seq

    # Main Loop
    while time.time() < deadline - 5:
        # Random restart
        # Sometimes start from best, sometimes from random
        if random.random() < 0.3:
            start_seq = best_seq[:]
            # Perturb best_seq
            if start_seq:
                idx = random.randint(0, len(start_seq) - 1)
                start_seq.pop(idx)
        else:
            start_seq = random_init()
            
        # Run local search for a short duration
        # Estimate time per iteration: 
        # Evaluate is O(N^2). N=100. 10k ops. 
        # Python loop overhead. Maybe 1ms per iteration?
        # So 1s = 1000 iterations.
        # Let's run for 1-2 seconds.
        improved_seq = local_search(start_seq, time_limit=2.0)
        
        improved_len, improved_dups = evaluate(improved_seq)
        if improved_dups == 0 and improved_len > best_len:
            best_len = improved_len
            best_seq = improved_seq
            
            # If we found a very long sequence, we can try to optimize it further
            # Or just continue searching for even longer ones.
            
        # Check time
        if time.time() > deadline - 5:
            break

    return best_seq
