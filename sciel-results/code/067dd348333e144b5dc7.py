import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Precompute prime exponent vectors for all numbers 2..128
    prime_list = primes
    num_primes = len(prime_list)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    
    # Exponent vector for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)

    # Greedy initialization to get a good lower bound
    def build_greedy():
        seq = []
        seen = set()
        end_vecs = set()
        
        for n in all_numbers:
            v_n = exp_vecs[n]
            new_end_vecs = {v_n}
            for v in end_vecs:
                new_v = tuple(v[i] + v_n[i] for i in range(num_primes))
                new_end_vecs.add(new_v)
            
            # Check for conflicts
            conflict = False
            for v in new_end_vecs:
                if v in seen:
                    conflict = True
                    break
            
            if not conflict:
                seq.append(n)
                seen.update(new_end_vecs)
                end_vecs = new_end_vecs
        
        return seq

    best_seq = build_greedy()
    best_len = len(best_seq)
    
    # DFS Branch and Bound
    # State: index, current_seq, seen_products (set of tuples), end_vecs (set of tuples)
    # To save memory/time, we can store seen_products as a set of tuples.
    
    # Optimization: Use a global variable for best
    global_best_len = best_len
    global_best_seq = best_seq[:]
    
    # Heuristic: Order of numbers? 
    # The problem requires the final sequence to be sorted. 
    # Our DFS adds numbers in increasing order of index (2, 3, ..., 128).
    # This naturally builds the sorted sequence.
    
    # Pruning:
    # 1. If len(current) + remaining <= global_best_len, prune.
    # 2. We can also check if it's possible to add more numbers.
    
    # To make it faster, we can use a bitset for seen products? 
    # The products can be large (up to 128^127). 
    # But we use prime exponent vectors. The dimension is ~30 primes.
    # The values in the vector are small (exponents).
    # We can hash the tuple.
    
    def dfs(index, current_seq, seen_products, end_vecs):
        nonlocal global_best_len, global_best_seq
        
        if time.time() > deadline:
            return
            
        # Pruning
        remaining = 128 - index + 1
        if len(current_seq) + remaining <= global_best_len:
            return
            
        if index > 128:
            if len(current_seq) > global_best_len:
                global_best_len = len(current_seq)
                global_best_seq = current_seq[:]
            return
        
        n = index
        
        # Option 1: Exclude n
        dfs(index + 1, current_seq, seen_products, end_vecs)
        
        # Option 2: Include n
        v_n = exp_vecs[n]
        new_end_vecs = [v_n]
        for v in end_vecs:
            new_v = tuple(v[i] + v_n[i] for i in range(num_primes))
            new_end_vecs.append(new_v)
        
        # Convert to set for fast lookup and to avoid duplicates in new_end_vecs?
        # Duplicates in new_end_vecs would mean internal collision? 
        # If two different blocks ending at n have the same product, that's a collision.
        # So we must ensure new_end_vecs has unique elements.
        new_end_set = set(new_end_vecs)
        if len(new_end_set) < len(new_end_vecs):
            # Internal collision
            return
        
        # Check against seen_products
        conflict = False
        for v in new_end_set:
            if v in seen_products:
                conflict = True
                break
        
        if not conflict:
            new_seen = seen_products | new_end_set
            dfs(index + 1, current_seq + [n], new_seen, new_end_set)
            
    # Start DFS
    # Initial state: index=2, empty seq, empty seen, empty end_vecs
    # Note: seen_products is a set of tuples.
    # end_vecs is a set of tuples.
    
    # To speed up, we can try to include numbers that are "likely" to be in the solution first?
    # But the order is fixed (increasing).
    # We can randomize the "exclude" vs "include" order?
    # Usually, including is better for maximizing length, but might lead to dead ends.
    # Let's stick to Include first, then Exclude.
    
    # However, simple DFS might be too slow if the tree is large.
    # We can use a greedy lower bound to prune early.
    # We already have best_len from greedy.
    
    # Let's run the DFS
    dfs(2, [], set(), set())
    
    # If DFS didn't improve, return best from greedy or DFS
    if global_best_len > best_len:
        return global_best_seq
    else:
        return best_seq
