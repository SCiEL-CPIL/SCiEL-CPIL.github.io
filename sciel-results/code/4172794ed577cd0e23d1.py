import math
import random

def construct():
    # Helper to check if a number is prime
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    # Helper to count divisors
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    # Function to check if appending x to seq creates duplicate block products
    def can_append(seq, x, existing_products):
        if not seq:
            return True, {x}
        
        # Calculate new block products ending at x
        new_prods = []
        prod = x
        for a in reversed(seq):
            new_prods.append(prod)
            prod *= a
        new_prods_set = set(new_prods)
        
        if existing_products.intersection(new_prods_set):
            return False, None
        return True, new_prods_set

    # Greedy construction given an order
    def greedy_construct(order):
        seq = []
        products = set()
        for x in order:
            ok, new_prods = can_append(seq, x, products)
            if ok:
                seq.append(x)
                products.update(new_prods)
        return seq

    # Generate different orders
    numbers = list(range(2, 129))
    
    orders = [
        numbers,  # Natural
        sorted(numbers, key=lambda x: (0 if is_prime(x) else 1, x)), # Primes first
        sorted(numbers, reverse=True), # Descending
        sorted(numbers, key=lambda x: num_divisors(x)), # Fewest divisors first
        sorted(numbers, key=lambda x: -num_divisors(x)), # Most divisors first (risky?)
    ]

    best_seq = []
    for order in orders:
        seq = greedy_construct(order)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Optional: Local Search / Hill Climbing on the best sequence
    # Try to improve by swapping elements or inserting skipped ones
    # Given time constraints, a simple pass of trying to insert skipped numbers 
    # at the end of the greedy list (in original order) might help if greedy 
    # missed some due to order.
    
    # Actually, the greedy above already considers all numbers in the given order.
    # If a number was skipped, it was because it collided with the *current* prefix.
    # A different order might allow it.
    
    # Let's do a second pass: take the best_seq, and try to insert any number 
    # from 2..128 not in best_seq, checking validity.
    # Since best_seq is strictly increasing, we can't just insert anywhere; 
    # the sequence must remain sorted. 
    # So we can only check if adding a number x (where x is not in best_seq) 
    # to the *end* of best_seq works? No, x must be placed in sorted position.
    # This complicates the "append" logic.
    
    # The greedy approach inherently builds the sorted sequence.
    # So the multi-start greedy is the primary improvement.
    
    return best_seq
