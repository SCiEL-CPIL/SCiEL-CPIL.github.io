import time
import random
import math
from collections import Counter

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
    
    def compute_all_products(seq):
        """Compute all consecutive block products. Returns list of (start, end, product)."""
        n = len(seq)
        prods = []
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.append((i, j, prod))
        return prods
    
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
        cnt = Counter(products)
        return sum(c - 1 for c in cnt.values() if c > 1)
    
    def try_add_element(seq, elem):
        """Try to add elem to seq, return new seq if valid."""
        new_seq = sorted(seq + [elem])
        if is_valid_fast(new_seq):
            return new_seq
        return None
    
    def polish(seq):
        """Final polish: try all single-element swaps and insertions systematically."""
        if time.time() > deadline - 2:
            return seq
        
        current = seq
        current_set = set(current)
        all_nums = list(range(2, 129))
        
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            improved = False
            
            # Try adding any missing element
            for c in all_nums:
                if c in current_set:
                    continue
                new_seq = sorted(current + [c])
                if is_valid_fast(new_seq):
                    current = new_seq
                    current_set = set(current)
                    improved = True
                    break
            
            # If no addition worked, try swaps
            if not improved:
                for a in current:
                    if time.time() > deadline - 2:
                        break
                    for c in all_nums:
                        if c in current_set:
                            continue
                        new_seq = sorted((current_set - {a}) | {c})
                        if is_valid_fast(new_seq):
                            # Check if this swap enables further additions
                            new_set = set(new_seq)
                            can_add = False
                            for d in all_nums:
                                if d in new_set:
                                    continue
                                test_seq = sorted(new_seq + [d])
                                if is_valid_fast(test_seq):
                                    can_add = True
                                    break
                            if can_add or len(new_seq) >= len(current):
                                current = new_seq
                                current_set = set(current)
                                improved = True
                                break
                    if improved:
                        break
        
        return current
    
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
            if time.time() > deadline - 5:
                break
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
    
    # Local search: try swaps that reduce conflicts
    def local_search_conflict(seq):
        """Try swaps that reduce the conflict count (smooth surrogate)."""
        if time.time() > deadline - 5:
            return seq
        
        seq_set = set(seq)
        candidates = [c for c in all_nums if c not in seq_set]
        if not candidates:
            return seq
        
        current_conflicts = count_conflicts(seq)
        best_new_seq = None
        best_conflicts = current_conflicts
        
        for a in seq:
            for c in candidates:
                new_seq = sorted((seq_set - {a}) | {c})
                new_conflicts = count_conflicts(new_seq)
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_new_seq = new_seq
        
        if best_new_seq is not None and best_conflicts < current_conflicts:
            return best_new_seq
        return seq
    
    # Main optimization loop
    def optimize(seq):
        """Run multiple rounds of greedy add and local search."""
        current = seq
        for _ in range(30):  # Max 30 rounds
            if time.time() > deadline - 5:
                break
            
            # Try greedy add
            new_seq = greedy_add(current)
            if len(new_seq) > len(current):
                current = new_seq
                continue
            
            # Try local search based on conflict reduction
            new_seq = local_search_conflict(current)
            if new_seq is not current:
                current = new_seq
                # After swap, try to add
                new_seq = greedy_add(current)
                if len(new_seq) > len(current):
                    current = new_seq
                continue
            
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
        
        # Final polish pass
        current = polish(current)
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
