import time
import sys
from functools import lru_cache

# Increase recursion limit for deep backtracking
sys.setrecursionlimit(10000)

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute primes and composites
    all_numbers = list(range(2, 129))
    primes = [p for p in all_numbers if is_prime(p)]
    composites = [c for c in all_numbers if not is_prime(c)]
    
    # The set of all available numbers
    all_available = set(all_numbers)
    
    best_sequence = []
    best_len = 0
    
    # Memoization cache: stores the max additional length achievable from a given state
    # State is represented by the frozenset of used numbers.
    # However, the validity depends on the *order* (suffixes), not just the set.
    # But since the sequence must be strictly increasing, the order is fixed by the set!
    # Wait, the task says "strictly increasing integers". 
    # So if we choose a set S, the sequence is uniquely determined: sorted(S).
    # Therefore, the problem reduces to: Find the largest subset S of [2, 128] 
    # such that the sequence sorted(S) has distinct block products.
    
    # This is a Subset Selection problem.
    # We can use backtracking on the sorted list of all numbers.
    # At each step, we decide to include or exclude the next number.
    # But we need to check validity incrementally.
    
    # Since the order is fixed (sorted), we can build the sequence from left to right (smallest to largest).
    # Let's use a DFS that builds the sequence by considering numbers in increasing order.
    # State: current sequence (tuple), index in all_numbers.
    # Actually, simpler: iterate through numbers 2..128.
    # Decision: Include or Exclude.
    # If Include, check if adding this number to the current sequence maintains distinct block products.
    
    # Optimization: 
    # 1. If we exclude a number, we can't include it later.
    # 2. If we include a number, we must check validity.
    # 3. Pruning: If len(current_seq) + (128 - current_index) <= best_len, prune.
    
    # To make this efficient, we need a fast validity check for appending a new number.
    # When appending x to sequence S = [s1, ..., sk], new block products are:
    # x
    # s_k * x
    # s_{k-1} * s_k * x
    # ...
    # s_1 * ... * s_k * x
    # We need to ensure none of these are in the set of existing block products.
    # We also need to ensure x itself is not in existing block products (though x > max(S) usually helps, 
    # but small composites like 4 might collide with 2*2 if 2 was in S? No, S is strictly increasing, 
    # so 2*2 is not a block product unless 2 appears twice, which it can't. 
    # However, 4 could collide with a previous block product like 2*3*2? No, strictly increasing.
    # So we just need to check the new suffix products against the global set of seen products.)
    
    # Let's represent the state as (index, current_sequence_tuple, seen_products_frozenset)
    # This might be too large for lru_cache if seen_products is a frozenset of large integers.
    # Instead, we'll do manual backtracking with a global best and a "dead end" set if possible.
    # Given the small range (127 numbers), 2^127 is too big, but pruning is strong.
    
    # Let's try a different angle: 
    # The constraint is on *consecutive* blocks.
    # If we build the sequence in increasing order, the "suffix" of the current sequence is the whole sequence.
    # When we add a new number x, the new blocks are the suffixes of the new sequence ending at x.
    # These are: x, prev_last * x, prev_last2 * prev_last * x, etc.
    # We maintain a set `all_products` of all block products seen so far.
    # We maintain the current sequence `seq`.
    # We maintain the list of suffix products of the current seq: `suffix_prods`.
    #   suffix_prods[0] = seq[-1]
    #   suffix_prods[1] = seq[-2]*seq[-1]
    #   ...
    # When adding x:
    #   new_suffix_prods = [x]
    #   for p in reversed(suffix_prods):
    #       new_suffix_prods.append(p * x)
    #   Check if any p in new_suffix_prods is in all_products.
    #   If yes, invalid.
    #   If no, update all_products and suffix_prods.
    
    # This is O(N^2) per step, which is fine for N ~ 100.
    
    def dfs(idx, seq, all_products, suffix_prods):
        nonlocal best_sequence, best_len
        
        # Pruning: Max possible length is current len + remaining numbers
        if len(seq) + (128 - idx + 1) <= best_len:
            return
            
        if time.time() > deadline:
            return
            
        if idx > 128:
            if len(seq) > best_len:
                best_len = len(seq)
                best_sequence = list(seq)
            return
            
        # Option 1: Exclude all_numbers[idx]
        dfs(idx + 1, seq, all_products, suffix_prods)
        
        # Option 2: Include all_numbers[idx]
        x = all_numbers[idx]
        
        # Check validity of adding x
        # New suffix products
        new_suffix_prods = [x]
        valid = True
        # Multiply existing suffixes by x
        for p in reversed(suffix_prods):
            new_suffix_prods.append(p * x)
            
        # Check collisions
        for p in new_suffix_prods:
            if p in all_products:
                valid = False
                break
                
        if valid:
            # Update state
            new_all_products = set(all_products)
            new_all_products.update(new_suffix_prods)
            new_seq = seq + (x,)
            # new_suffix_prods is already calculated in reverse order of importance? 
            # suffix_prods was [last, last2*last, ...]
            # new_suffix_prods is [x, last*x, last2*last*x, ...]
            # This is the correct format for the next step.
            dfs(idx + 1, new_seq, new_all_products, new_suffix_prods)
            
    # Initial state
    initial_seq = ()
    initial_products = set()
    initial_suffixes = []
    
    dfs(0, initial_seq, initial_products, initial_suffixes)
    
    return best_sequence

# Fallback if the strict increasing constraint interpretation was wrong and order matters more than set?
# "Return strictly increasing integers" implies the output is sorted.
# The products of "consecutive blocks" depend on the order in the returned list.
# Since the list is strictly increasing, the order is fixed.
# So the set-based backtracking above is correct.

# However, DFS on 127 items might still be slow if pruning is weak.
# Let's add a heuristic: try including numbers first? Or excluding?
# Usually, primes are "safe" to include. Composites cause collisions.
# So we should prioritize including primes.
# My current DFS tries Exclude first, then Include.
# This will find a valid set quickly (e.g., just primes), but might miss larger sets if it doesn't backtrack enough.
# Actually, standard DFS explores the "Exclude" branch first, which leads to small sets.
# We want to find *large* sets.
# It's better to try "Include" first to grow the sequence quickly, then backtrack if stuck.
# But "Include" first might hit a dead end early.
# A hybrid: Greedy include if valid, else exclude? No, that's the current program's spirit.
# Let's stick to DFS but swap the order: Include first, then Exclude.
# This biases towards larger sets.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    all_numbers = list(range(2, 129))
    
    best_sequence = []
    best_len = 0
    
    # Use a list for seq to avoid tuple copying overhead? 
    # Tuples are hashable and immutable, good for cache, but we aren't caching states with lru_cache 
    # because all_products is a set.
    # We'll use lists for speed.
    
    def dfs(idx, seq, all_products, suffix_prods):
        nonlocal best_sequence, best_len
        
        if time.time() > deadline:
            return
            
        # Pruning
        if len(seq) + (128 - idx + 1) <= best_len:
            return
            
        if idx > 128:
            if len(seq) > best_len:
                best_len = len(seq)
                best_sequence = list(seq)
            return
            
        x = all_numbers[idx]
        
        # Try Include first (greedy bias for length)
        new_suffix_prods = [x]
        valid = True
        for p in reversed(suffix_prods):
            new_suffix_prods.append(p * x)
            
        for p in new_suffix_prods:
            if p in all_products:
                valid = False
                break
                
        if valid:
            # Copy state
            new_seq = seq + [x]
            new_all_products = set(all_products)
            new_all_products.update(new_suffix_prods)
            dfs(idx + 1, new_seq, new_all_products, new_suffix_prods)
            
        # Try Exclude
        dfs(idx + 1, seq, all_products, suffix_prods)
        
    dfs(0, [], set(), [])
    
    return best_sequence
