import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Candidates are 2 to 128
    candidates = list(range(2, 129))
    
    best_seq = []
    best_prods = set()
    best_len = 0
    
    # Simple Greedy: Iterate through candidates, add if no conflict
    # Since the sequence must be strictly increasing, we process in sorted order
    current_seq = []
    current_prods = set()
    
    for x in candidates:
        if time.time() > deadline - 5.0:
            break
        
        # Check if x can be appended to current_seq
        # New blocks are: [x], [last, x], [second_last, last, x], ...
        # Let's compute the products of suffixes of current_seq
        # suffix_prod = 1
        # for i in reversed(range(len(current_seq))):
        #    suffix_prod *= current_seq[i]
        #    new_prod = suffix_prod * x
        #    if new_prod in current_prods or new_prod in current_seq:
        #        conflict = True
        #    # Also need to check if new_prod is in the set of NEW products
        #    # But new products are distinct because seq is increasing
        
        # We also need to ensure x itself is not in current_prods (except as a singleton of itself, which is new)
        # Actually, if x is in current_prods, it means some block product equals x.
        # Since x is not in current_seq (we are adding it), this is a conflict.
        
        if x in current_prods:
            continue
            
        # Check suffix products
        conflict = False
        suffix_prod = 1
        # We need to check products: x, s_k*x, s_{k-1}*s_k*x, ...
        # Note: The product x itself is checked above (x in current_prods).
        # Now check x * s_k, x * s_k * s_{k-1}, etc.
        
        # Optimization: If the product grows too large, it's unlikely to be in current_prods 
        # unless current_prods contains very large numbers. 
        # Max product in current_prods is bounded by the product of all elements in current_seq.
        # Since elements are <= 128, and length is small, products get huge quickly.
        
        for i in range(len(current_seq) - 1, -1, -1):
            suffix_prod *= current_seq[i]
            new_prod = suffix_prod * x
            
            # If new_prod is in current_prods, conflict
            if new_prod in current_prods:
                conflict = True
                break
            
            # If new_prod is in current_seq, conflict
            # Note: new_prod > x > current_seq[i] ... so new_prod is larger than any element in current_seq
            # Wait, current_seq is increasing. x > current_seq[-1].
            # new_prod = x * (product of some suffix).
            # Since suffix product >= 1, new_prod >= x.
            # If suffix product > 1, new_prod > x > current_seq[-1].
            # So new_prod cannot be in current_seq.
            # Only x itself (suffix product 1) could be in current_seq, but x is not in current_seq yet.
            # So we don't need to check new_prod in current_seq for i < len-1.
            # For i = len-1 (first step), new_prod = x * current_seq[-1]. This is > x.
            
            # So we only need to check against current_prods.
            
        if not conflict:
            # Accept x
            current_seq.append(x)
            
            # Update current_prods
            # Add x
            current_prods.add(x)
            
            # Add other new products
            suffix_prod = 1
            for i in range(len(current_seq) - 2, -1, -1):
                suffix_prod *= current_seq[i]
                current_prods.add(suffix_prod * x)
            
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = list(current_seq)
    
    # The greedy approach is deterministic.
    # Can we do better?
    # The greedy choice might block future numbers.
    # Example: Adding 4 might block 8, 12, 16, etc.
    # Skipping 4 might allow 8?
    # If we skip 4, we don't add products involving 4.
    # Then 8 might be addable.
    # But 4 is smaller than 8.
    # Generally, smaller numbers create smaller products, which are more likely to collide with other small numbers or early products.
    # However, 4 is a square. 2*2.
    
    # Let's try a second pass with a different strategy:
    # Randomized Greedy with Restart?
    # Or "Greedy with Lookahead"?
    
    # Given the time budget, a simple improvement is to try multiple random permutations of the "decision" order?
    # No, the order is fixed by value for the sequence.
    # But we can randomize the *selection* if we allow out-of-order? No, "strictly increasing integers" implies the output list is sorted.
    # The set is what matters.
    
    # Let's try a local search on the set found by greedy.
    # Try removing an element and re-adding others?
    # Or try a "Greedy from the right"?
    # If we pick large numbers first, products get huge, reducing collisions.
    # But the sequence must be increasing.
    # So we must pick a subset.
    
    # Let's try a strategy:
    # 1. Start with empty set.
    # 2. Iterate x from 128 down to 2.
    # 3. Try to INSERT x into the correct sorted position?
    # This is complex.
    
    # Alternative:
    # The greedy "left-to-right" (2 to 128) is standard.
    # Let's verify the "new_prod in current_prods" logic.
    # Is it possible that skipping a number allows adding TWO numbers later?
    # Yes.
    
    # Let's run a few randomized trials where we randomly skip some candidates with low probability to explore the space.
    
    best_len = len(current_seq)
    best_seq = list(current_seq)
    
    # Randomized restarts
    import random
    random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        
        current_seq = []
        current_prods = set()
        
        # Randomly decide to skip small numbers?
        # Or just shuffle the "skip" decisions?
        # Let's try: For each x, with prob 0.2, skip it even if valid.
        # This explores different subsets.
        
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            
            # Check validity
            if x in current_prods:
                continue
                
            conflict = False
            suffix_prod = 1
            for i in range(len(current_seq) - 1, -1, -1):
                suffix_prod *= current_seq[i]
                new_prod = suffix_prod * x
                if new_prod in current_prods:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Decide to include
            # Heuristic: Include with high prob, but sometimes skip to allow others
            # Skip prob increases for small numbers?
            # Let's just use a fixed skip prob of 0.1 for exploration
            if random.random() < 0.1:
                continue
                
            # Accept
            current_seq.append(x)
            current_prods.add(x)
            suffix_prod = 1
            for i in range(len(current_seq) - 2, -1, -1):
                suffix_prod *= current_seq[i]
                current_prods.add(suffix_prod * x)
                
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            
    return best_seq
