import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    def get_primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def prime_factorization(n):
        """Return dict of prime->exponent for n."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    def is_valid(seq):
        """Check if all consecutive block products are distinct."""
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
    
    def get_all_products(seq):
        """Get set of all consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def is_valid_insert(seq, pos, x):
        """Check if inserting x at pos in seq keeps all products distinct."""
        # Build new sequence
        new_seq = seq[:pos] + [x] + seq[pos:]
        n = len(new_seq)
        
        # Check all new products involving x
        # Products that include x: blocks [i..j] where i <= pos <= j
        existing_products = set()
        
        # First, get all products NOT involving x (i.e., from original seq)
        # But we need to be careful - let's just compute all products of new_seq
        # and check for duplicates
        
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= new_seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def greedy_construct():
        """Greedy construction: add numbers in order of increasing value."""
        seq = []
        used = set()
        
        for x in range(2, 129):
            if time.time() > deadline:
                break
            # Try to insert x at the correct position to maintain sorted order
            # Find position
            pos = 0
            while pos < len(seq) and seq[pos] < x:
                pos += 1
            
            if is_valid_insert(seq, pos, x):
                seq.insert(pos, x)
                used.add(x)
        
        return seq
    
    def local_search(seq, max_iterations=1000):
        """Local search: try to improve by adding, removing, or swapping elements."""
        best_seq = seq[:]
        best_len = len(seq)
        iterations = 0
        improved = True
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline:
                break
            
            # Try adding each unused number
            used = set(best_seq)
            for x in range(2, 129):
                if x in used:
                    continue
                pos = 0
                while pos < len(best_seq) and best_seq[pos] < x:
                    pos += 1
                if is_valid_insert(best_seq, pos, x):
                    new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    if len(new_seq) > len(best_seq):
                        best_seq = new_seq
                        best_len = len(best_seq)
                        improved = True
                        break
            
            # Try removing elements to allow others to be added
            for i in range(len(best_seq)):
                if time.time() > deadline:
                    break
                trial_seq = best_seq[:i] + best_seq[i+1:]
                used = set(trial_seq)
                for x in range(2, 129):
                    if x in used:
                        continue
                    pos = 0
                    while pos < len(trial_seq) and trial_seq[pos] < x:
                        pos += 1
                    if is_valid_insert(trial_seq, pos, x):
                        new_seq = trial_seq[:pos] + [x] + trial_seq[pos:]
                        if len(new_seq) > len(best_seq):
                            best_seq = new_seq
                            best_len = len(best_seq)
                            improved = True
                            break
                if improved:
                    break
        
        return best_seq
    
    # Start with greedy construction
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Local search to improve
    improved_seq = local_search(best_seq)
    if len(improved_seq) > len(best_seq):
        best_seq = improved_seq
        best_len = len(best_seq)
    
    # Try multiple random restarts with different orderings
    import random
    rng = random.Random(42)
    
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Create a random permutation of candidates
        candidates = list(range(2, 129))
        rng.shuffle(candidates)
        
        seq = []
        for x in candidates:
            if time.time() > deadline - 2:
                break
            pos = 0
            while pos < len(seq) and seq[pos] < x:
                pos += 1
            if is_valid_insert(seq, pos, x):
                seq.insert(pos, x)
        
        if len(seq) > len(best_seq):
            best_seq = seq
            best_len = len(seq)
        
        # Local search on this result
        improved_seq = local_search(seq, max_iterations=200)
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
            best_len = len(best_seq)
    
    return best_seq
