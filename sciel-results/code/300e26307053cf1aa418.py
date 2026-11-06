import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def is_valid_fast(seq):
        """Fast validity check with early exit."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def count_conflicts(seq):
        """Count the number of duplicate products (lower is better)."""
        products = []
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.append(prod)
        # Count duplicates
        from collections import Counter
        cnt = Counter(products)
        return sum(c - 1 for c in cnt.values() if c > 1)
    
    def try_add_element(seq, elem):
        """Try to add elem to seq, return new seq if valid."""
        new_seq = sorted(seq + [elem])
        if is_valid_fast(new_seq):
            return new_seq
        return None
    
    primes = get_primes(128)
    all_nums = list(range(2, 129))
    composites = [x for x in all_nums if x not in primes]
    
    best_seq = primes[:]
    best_len = len(primes)
    
    # Greedy Add - prioritize composites that are "easy" to add
    def greedy_add(seq, order=None):
        if order is None:
            order = sorted(composites)
        improved = True
        while improved:
            improved = False
            seq_set = set(seq)
            for c in order:
                if c in seq_set:
                    continue
                new_seq = try_add_element(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    seq_set = set(seq)
                    improved = True
                    break
        return seq
    
    # Local search with lookahead
    def local_search_with_lookahead(seq):
        """Try swaps with lookahead to see if they enable additions."""
        seq_set = set(seq)
        candidates = [c for c in composites if c not in seq_set]
        if not candidates:
            return seq
        
        best_new_seq = None
        best_add_count = -1
        
        for a in seq:
            for c in candidates:
                new_seq = sorted((seq_set - {a}) | {c})
                if not is_valid_fast(new_seq):
                    continue
                
                # Lookahead: how many elements can we add?
                new_seq_set = set(new_seq)
                add_count = 0
                temp_seq = new_seq
                temp_set = set(temp_seq)
                for d in candidates:
                    if d in temp_set:
                        continue
                    # Also try other composites
                    for e in composites:
                        if e in temp_set:
                            continue
                        test_seq = try_add_element(temp_seq, e)
                        if test_seq is not None:
                            add_count += 1
                            temp_seq = test_seq
                            temp_set = set(test_seq)
                            break
                
                if add_count > best_add_count:
                    best_add_count = add_count
                    best_new_seq = new_seq
        
        if best_new_seq is not None and best_add_count >= 0:
            return best_new_seq
        return seq
    
    # Main optimization loop
    def optimize(seq):
        """Run multiple rounds of greedy add and local search."""
        current = seq
        for _ in range(20):  # Max 20 rounds
            if time.time() > deadline - 5:
                break
            
            # Try greedy add
            new_seq = greedy_add(current)
            if len(new_seq) > len(current):
                current = new_seq
                continue
            
            # Try local search with lookahead
            new_seq = local_search_with_lookahead(current)
            if new_seq is not current and is_valid_fast(new_seq):
                # After swap, try to add
                new_seq = greedy_add(new_seq)
                if len(new_seq) >= len(current):
                    current = new_seq
                else:
                    break
            else:
                # No improvement from local search, try random perturbation
                if len(current) > 3:
                    # Remove a random element
                    a = random.choice(current)
                    perturbed = sorted(set(current) - {a})
                    if is_valid_fast(perturbed):
                        perturbed = greedy_add(perturbed)
                        if len(perturbed) >= len(current) - 1:
                            current = perturbed
                        else:
                            break
                    else:
                        break
                else:
                    break
        
        return current
    
    # Initial greedy construction
    current_seq = primes[:]
    current_seq = greedy_add(current_seq)
    if len(current_seq) > best_len:
        best_seq = current_seq
        best_len = len(current_seq)
    
    # Run optimization
    optimized = optimize(current_seq)
    if len(optimized) > best_len:
        best_seq = optimized
        best_len = len(optimized)
    
    # Try multiple random restarts
    restart_count = 0
    while time.time() < deadline - 10 and restart_count < 50:
        restart_count += 1
        
        # Randomized greedy
        shuffled_comps = composites[:]
        random.shuffle(shuffled_comps)
        # Also consider starting with a subset of primes
        start_primes = primes[:]
        random.shuffle(start_primes)
        # Keep first k primes where k varies
        k = random.randint(max(2, len(primes) - 5), len(primes))
        current_seq = sorted(start_primes[:k])
        
        # Try to add from shuffled composites
        current_seq = greedy_add(current_seq, order=shuffled_comps)
        
        # Optimize
        optimized = optimize(current_seq)
        
        if len(optimized) > best_len:
            best_seq = optimized
            best_len = len(optimized)
    
    return best_seq
