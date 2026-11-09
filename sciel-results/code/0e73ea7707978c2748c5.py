import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    # State for DFS
    current_seq = []
    current_prods = set()
    
    # Helper to check validity of adding x to current_seq
    def can_add(x):
        # Calculate new suffix products
        prod = x
        # Check singleton
        if prod in current_prods:
            return False
        
        # Check longer suffixes
        for i in range(len(current_seq) - 1, -1, -1):
            prod *= current_seq[i]
            if prod in current_prods:
                return False
            # Early termination if product is too large? 
            # Max product in set is bounded by 128^128, which is huge.
            # But we only care if it's in the set.
            # The set contains products from previous steps.
            # The maximum product in the set is determined by the longest suffix of the current sequence.
            # If prod exceeds the maximum possible product in the set, we can stop?
            # The max product in the set is the product of the entire current_seq.
            # Let's precompute the max product in the set? No, just check membership.
            # The products grow very fast. The number of suffixes to check is small.
        return True

    def get_new_prods(x):
        prods = [x]
        prod = x
        for i in range(len(current_seq) - 1, -1, -1):
            prod *= current_seq[i]
            prods.append(prod)
        return prods

    def dfs(last_val):
        nonlocal best_seq, best_len, current_seq, current_prods
        
        # Time check
        if time.time() > deadline:
            return

        # Pruning: if we can't beat the best length, stop
        if len(current_seq) + (128 - last_val) <= best_len:
            return

        # Try adding next numbers
        for x in range(last_val + 1, 129):
            if time.time() > deadline:
                return
            
            if can_add(x):
                # Add x
                new_prods = get_new_prods(x)
                
                current_seq.append(x)
                current_prods.update(new_prods)
                
                # Update best
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq.copy()
                
                # Recurse
                dfs(x)
                
                # Backtrack
                current_seq.pop()
                for p in new_prods:
                    current_prods.discard(p)

    # Start with the first number
    # We can start with any number from 2 to 128.
    # To be efficient, we can iterate the starting number.
    # But the DFS above assumes we start from scratch for each start?
    # Better: The DFS above starts from last_val=1 (so x starts from 2).
    # It will try 2, then 3, etc.
    # This covers all sequences starting with 2.
    # What if the best sequence starts with 3?
    # The loop `for x in range(last_val + 1, 129)` inside DFS handles this.
    # If 2 is chosen, it goes deeper. If 2 leads to a short sequence, it backtracks and tries 3 as the *first* element?
    # No, the initial call is dfs(1).
    # Inside dfs(1), it tries x=2.
    # If x=2 leads to a full exploration and backtracks, it then tries x=3.
    # So yes, it explores all sequences.
    
    dfs(1)
    
    # If the search didn't find anything (unlikely), fall back to greedy
    if not best_seq:
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        best_seq = values

    return best_seq
