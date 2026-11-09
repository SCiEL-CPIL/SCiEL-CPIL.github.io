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
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    best_seq = []
    best_len = 0
    
    def fast_greedy(order):
        """
        Greedily add elements from 'order' to the sequence.
        Uses incremental check for speed.
        """
        values = []
        products = set()
        for x in order:
            # Check if x can be appended to values
            # New products are: x, values[-1]*x, values[-2]*values[-1]*x, ...
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
                
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        
        return values

    # Strategy 1: Standard Ascending Greedy
    values = fast_greedy(range(2, 129))
    if len(values) > best_len:
        best_len = len(values)
        best_seq = values

    # Strategy 2: Descending Greedy (using full is_valid is slow, but let's try a few random descending orders)
    # Actually, let's just rely on randomized ascending orders which cover the space better.
    
    # Strategy 3: Randomized Greedy with many restarts
    # We can also try shuffling and then sorting? No, the order of addition matters for greedy.
    # The resulting sequence is always sorted because we append.
    # Wait, if we append x, and x < values[-1], the sequence is no longer sorted.
    # The problem requires the *returned* sequence to be strictly increasing.
    # If we build a set, we can sort it. But the incremental check `tails` assumes we are appending to the *end* of the current list.
    # If we want to maintain a sorted list, we can only append if x > current_max.
    # If we allow arbitrary order, we must re-sort and re-check validity from scratch or use a different data structure.
    
    # Let's stick to the constraint that we build a sorted sequence by only adding x if x > last element.
    # This means we are effectively choosing a subsequence of [2, 128] in increasing order.
    # The "order" of consideration determines which elements we pick.
    # If we shuffle [2, 128] and process in that order, we might skip a small number and pick a larger one, then we can never pick the small one again (since it's < last).
    # This is a valid strategy for finding a subsequence.
    
    # To explore more, we can try different "biases" in the shuffle.
    
    num_restarts = 0
    while time.time() < deadline:
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        # We process candidates in the shuffled order.
        # We maintain a sorted list `values`.
        # We can only add x if x > values[-1] (to keep it sorted and allow incremental check).
        # This restricts us to increasing subsequences.
        
        values = []
        products = set()
        
        for x in candidates:
            if values and x <= values[-1]:
                continue
            
            # Check if x can be appended
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
                
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        
        if len(values) > best_len:
            best_len = len(values)
            best_seq = values
            num_restarts += 1
            
            # If we find a very long sequence, we might want to try to polish it
            # But for now, just keep searching.
            
            # Break if we hit a theoretical max or close to it?
            # Max length is 127. Unlikely.
            if best_len > 100:
                break
        
        # Check time before next iteration
        if time.time() > deadline - 1.0:
            break

    return best_seq
