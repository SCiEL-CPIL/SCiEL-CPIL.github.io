import time
import random

def is_valid(seq):
    """
    Check if a strictly increasing sequence has distinct products for all non-empty consecutive blocks.
    """
    n = len(seq)
    if n == 0:
        return True
    
    products = set()
    # Generate all subsequence products
    # seq is sorted
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
            
    return True

def construct():
    deadline = time.time() + 80.0
    best_seq = []
    best_len = 0
    
    # 1. Try Ascending Greedy (similar to seed but with full check for consistency, or just use seed logic)
    # Seed logic is faster, let's use a fast incremental check for the initial greedy runs.
    
    def fast_greedy(order):
        values = []
        products = set()
        for x in order:
            # Compute tails for x added to end of values
            # Note: This only works if we append. 
            # In "Ascending", we append.
            # In "Descending", we don't append in sorted order, so we can't use this simple tail check 
            # unless we sort and re-check. 
            # So for Descending, we must use the set-based approach or sort-and-check.
            # Let's just use the set-based approach for all randomized/generic runs to be safe.
            pass
        return values

    # Strategy 1: Standard Ascending Greedy (Seed Logic)
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    if len(values) > best_len:
        best_len = len(values)
        best_seq = values

    # Strategy 2: Descending Greedy (Set-based)
    candidates = list(range(128, 1, -1))
    current_set = set()
    for x in candidates:
        new_set = current_set | {x}
        new_seq = sorted(new_set)
        if is_valid(new_seq):
            current_set = new_set
    seq = sorted(current_set)
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq

    # Strategy 3: Randomized Greedy
    # Run as many iterations as possible
    while time.time() < deadline:
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        current_set = set()
        
        for x in candidates:
            new_set = current_set | {x}
            new_seq = sorted(new_set)
            if is_valid(new_seq):
                current_set = new_set
        
        seq = sorted(current_set)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            # Optional: Early exit if we hit a theoretical max? 
            # Max possible is 127. Unlikely.
            # If we get close to 127, we can stop.
            if best_len > 100:
                break

    return best_seq
