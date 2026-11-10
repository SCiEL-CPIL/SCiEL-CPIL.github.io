import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We want to maximize the length of the sequence.
    # The values must be in [2, 128] and strictly increasing.
    # Products of all non-empty consecutive blocks must be distinct.
    
    best_length = 0
    best_sequence = []
    
    # To help the search, we can order candidates.
    # Primes are "safer".
    # Let's precompute primes.
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [i for i in range(2, 129) if is_prime(i)]
    # Sort candidates: maybe prioritize primes? Or just go in increasing order?
    # If we go in increasing order, we find the "smallest" valid sequence first.
    # To find the *longest*, we should explore all options.
    
    # A simple greedy that picks the smallest valid number might yield a long sequence.
    # Let's try a greedy first to get a baseline.
    def greedy_construct():
        seq = []
        products = set()
        for x in range(2, 129):
            if seq and x <= seq[-1]:
                continue
            # Check if x can be appended
            # New products: x, x*seq[-1], x*seq[-1]*seq[-2], ...
            new_prods = []
            prod = x
            for a in reversed(seq):
                new_prods.append(prod)
                prod *= a
            new_prods.append(prod) # The full product including x and all previous? 
            # Wait, the loop appends prod *before* multiplying by a?
            # Let's re-verify the product generation.
            # If seq = [a1, a2], append x.
            # Blocks ending at x: [x], [a2, x], [a1, a2, x].
            # Prods: x, a2*x, a1*a2*x.
            
            # My loop:
            # prod = x
            # for a in reversed([a1, a2]):
            #   a = a2: new_prods.append(x). prod = x*a2.
            #   a = a1: new_prods.append(x*a2). prod = x*a2*a1.
            # After loop, append prod (x*a2*a1).
            # So new_prods = [x, x*a2, x*a1*a2]. Correct.
            
            if not products.intersection(new_prods):
                seq.append(x)
                products.update(new_prods)
        return seq

    g_seq = greedy_construct()
    best_length = len(g_seq)
    best_sequence = g_seq
    
    # Now try to improve with backtracking.
    # Since the range is small, we can try to search for longer sequences.
    # We will use a recursive function.
    
    def backtrack(current_seq, current_products, last_val, depth):
        nonlocal best_length, best_sequence
        
        if time.time() > deadline:
            return
        
        # Pruning: if we can't beat best_length, stop.
        # Max possible length is 128 - last_val (if we could pick all remaining).
        if len(current_seq) + (128 - last_val) <= best_length:
            return
            
        # If we found a longer sequence, update best.
        if len(current_seq) > best_length:
            best_length = len(current_seq)
            best_sequence = current_seq[:]
            
        # Try to extend with next values
        for x in range(last_val + 1, 129):
            # Check time
            if time.time() > deadline:
                return
            
            # Calculate new products
            new_prods = []
            prod = x
            for a in reversed(current_seq):
                new_prods.append(prod)
                prod *= a
            new_prods.append(prod)
            
            # Check collision
            if current_products.intersection(new_prods):
                continue
                
            # If valid, recurse
            new_products = current_products.union(new_prods)
            backtrack(current_seq + [x], new_products, x, depth + 1)
            
            # Optional: If we want to explore more, we don't break.
            # But this is exponential. We need better pruning.
            
    # The backtracking above might be too slow.
    # Let's try a beam search or a more directed search.
    # Given the time limit, a simple backtracking with the pruning above might be okay if the depth is not too large.
    # The max depth is ~31. Branching factor is ~100. 100^31 is too big.
    # We need to be smarter.
    
    # Heuristic: Only consider candidates that are "likely" to be part of a long sequence.
    # Or, use the greedy result as a base and try to swap elements?
    
    # Let's try a different approach:
    # Iterate through all subsets? No.
    
    # Let's try to optimize the greedy by trying different orderings or starting points?
    # The sequence must be increasing, so the order is fixed by values.
    # The only choice is which values to include.
    
    # Let's try a local search on the greedy solution.
    # Try to replace a number with a larger one? No, that might break increasing order or collisions.
    # Try to insert a number?
    
    # Actually, the prime sequence is a very strong candidate.
    # Let's check if the greedy finds the prime sequence or something close.
    # Greedy picks 2. Products: {2}.
    # Picks 3. Prods: 3, 6. Set: {2, 3, 6}.
    # Picks 4? 4 in set? No. 4*3=12, 4*3*2=24.
    # If 4, 12, 24 not in set, pick 4.
    # So greedy might pick composites.
    
    # Let's run the backtracking but with a limit on the number of nodes visited or a smarter heuristic.
    # Given the 90s limit, we can afford a few seconds of search.
    
    # Let's try to search for a sequence of length best_length + 1.
    # If found, update best and search for +2, etc.
    
    # To make it faster, we can use the fact that if a number x is "bad" (causes many collisions), we skip it.
    
    # Let's stick to the backtracking but add a node limit or time check more aggressively.
    # And maybe sort candidates by "safety" (e.g., primes first).
    
    def backtrack_optimized(current_seq, current_products, last_val):
        nonlocal best_length, best_sequence
        
        if time.time() > deadline:
            return
        
        # Pruning
        if len(current_seq) + (128 - last_val) <= best_length:
            return
            
        if len(current_seq) > best_length:
            best_length = len(current_seq)
            best_sequence = current_seq[:]
            
        # Generate candidates
        candidates = list(range(last_val + 1, 129))
        # Sort candidates: Primes first?
        # This might help find long sequences faster if primes are better.
        candidates.sort(key=lambda x: (0 if is_prime(x) else 1, x))
        
        for x in candidates:
            if time.time() > deadline:
                return
            
            new_prods = []
            prod = x
            for a in reversed(current_seq):
                new_prods.append(prod)
                prod *= a
            new_prods.append(prod)
            
            if current_products.intersection(new_prods):
                continue
                
            new_products = current_products.union(new_prods)
            backtrack_optimized(current_seq + [x], new_products, x)
            
            # If we found a solution longer than current best, we might continue to see if we can go even longer.
            # But if we just want the max, we continue.
            
    # Start search
    backtrack_optimized([], set(), 1)
    
    return best_sequence

# Run it
# Note: The backtracking might still be slow.
# Let's add a fallback: if it takes too long, return the greedy result.
# The code above handles the deadline.

# One optimization: 
# If we reach a length that is very close to the theoretical max (127), we stop.
# But 127 is impossible.
# The prime count is 31.
# Let's assume the max is around 31-35.

# To ensure we don't time out, we can limit the depth of the backtracking or the number of branches.
# But the deadline check is there.

# Let's refine the candidate sorting.
# Maybe just increasing order is fine, as smaller numbers allow for more "room" in terms of value range, 
# but primes are safer.
# Let's try sorting by value first, but skip if it's a multiple of a previous number? 
# No, the product check handles that.

# Let's try to run the greedy first, then the backtracking.
# The backtracking starts from empty. It will re-explore the greedy path.
# We can pass the greedy solution as a "hint" or just let it find it.

# To speed up the backtracking, we can use a "beam" of best partial sequences.
# But simple backtracking with pruning is easier to implement correctly.

# Let's assume the backtracking will find the optimal or near-optimal solution within the time limit.
# If not, it returns the best found so far (which includes the greedy solution).

# One more thing: The problem asks for strictly increasing integers in [2, 128].
# The greedy and backtracking respect this.

# Final check on the product calculation.
# Sequence: [a1, a2, ..., ak]
# Append x.
# New blocks: [x], [ak, x], [ak-1, ak, x], ..., [a1, ..., ak, x].
# Products: x, ak*x, ak-1*ak*x, ..., a1*...*ak*x.
# My code:
# prod = x
# for a in reversed(seq):
#   new_prods.append(prod)
#   prod *= a
# new_prods.append(prod)
# Example seq=[2,3], x=5.
# reversed: 3, 2.
# 1. a=3: append 5. prod = 15.
# 2. a=2: append 15. prod = 30.
# Append 30.
# new_prods = [5, 15, 30]. Correct.

# The set `current_products` contains all products of all blocks in the current sequence.
# We check intersection.

# This looks correct.

# To improve performance, we can store `current_products` as a set of integers.
# Python sets are efficient.

# Let's add a small optimization: if the number of candidates is large, we can limit the branching.
# But with the deadline, it's safe.

# One potential issue: `current_products.union(new_prods)` creates a new set.
# This is O(N). N is the number of blocks, which is O(L^2).
# L is ~30. N is ~450.
# This is fine.

# Let's finalize the code.

