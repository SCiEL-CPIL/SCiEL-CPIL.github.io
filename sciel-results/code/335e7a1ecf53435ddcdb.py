import random
import time
from collections import defaultdict

def construct():
    # Precompute primes and factorizations for numbers 2..128
    def factorize(n):
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

    numbers = list(range(2, 129))
    primes = [x for x in numbers if is_prime(x)]
    composites = [x for x in numbers if not is_prime(x)]
    
    # Precompute factorization vectors for each number
    factorizations = {n: factorize(n) for n in numbers}
    
    # For efficient product comparison, we can use the factorization vector as a key
    # Since products can be huge, we represent them as tuples of (prime, exponent)
    
    def product_key(factors_list):
        """Combine multiple factorization dicts into a single sorted tuple key."""
        combined = {}
        for f in factors_list:
            for p, e in f.items():
                combined[p] = combined.get(p, 0) + e
        return tuple(sorted(combined.items()))
    
    def check_valid(vals):
        """
        vals: sorted list of unique integers
        returns True if all consecutive block products are distinct
        """
        prods = set()
        n = len(vals)
        for i in range(n):
            current_prod = {}
            for j in range(i, n):
                # Multiply by vals[j]
                for p, e in factorizations[vals[j]].items():
                    current_prod[p] = current_prod.get(p, 0) + e
                key = tuple(sorted(current_prod.items()))
                if key in prods:
                    return False
                prods.add(key)
        return True

    # Greedy initialization: start with all primes, try to add composites
    def greedy_init():
        current_set = set(primes)
        # Try adding composites in order of "safety" - maybe larger composites first?
        # Or random restarts
        best_set = set(primes)
        
        for _ in range(100):  # 100 random restarts
            temp_set = set(primes)
            shuffled = composites[:]
            random.shuffle(shuffled)
            for c in shuffled:
                new_set = temp_set | {c}
                if check_valid(sorted(new_set)):
                    temp_set = new_set
            if len(temp_set) > len(best_set):
                best_set = temp_set
        return sorted(best_set)
    
    # Get initial solution
    start_time = time.time()
    initial_solution = greedy_init()
    best_len = len(initial_solution)
    best_solution = initial_solution
    
    # Now try to improve with local search
    # Try swapping or adding/removing elements
    
    while time.time() - start_time < 55:
        # Try adding one composite not in current solution
        current_set = set(best_solution)
        found_improvement = False
        
        # Try all composites not in current_set
        for c in composites:
            if c in current_set:
                continue
            new_set = current_set | {c}
            if check_valid(sorted(new_set)):
                best_solution = sorted(new_set)
                best_len = len(best_solution)
                found_improvement = True
                break
        
        if found_improvement:
            continue
        
        # Try removing one element and adding two others?
        # Or try a more complex local move
        # For now, let's try random perturbations
        if random.random() < 0.5:
            # Remove a random composite and try to add different ones
            composites_in_set = [x for x in best_solution if not is_prime(x)]
            if composites_in_set:
                to_remove = random.choice(composites_in_set)
                perturbed_set = set(best_solution) - {to_remove}
                
                # Try to add composites
                shuffled = composites[:]
                random.shuffle(shuffled)
                for c in shuffled:
                    if c in perturbed_set:
                        continue
                    new_set = perturbed_set | {c}
                    if check_valid(sorted(new_set)):
                        perturbed_set = new_set
                
                if len(perturbed_set) > best_len:
                    best_solution = sorted(perturbed_set)
                    best_len = len(best_solution)
    
    return best_solution
