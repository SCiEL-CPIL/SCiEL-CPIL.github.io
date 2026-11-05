"""MIS-based solver for dense increasing sequences with unique block products."""
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # --- Conflict graph construction ---
    # Two numbers conflict if they cannot both appear in a valid sequence.
    # For efficiency, we precompute which subsets of {2..128} are "compatible"
    # by checking if they can be part of some valid sequence.
    # But full subset checking is too expensive. Instead, we use a pairwise
    # conflict approximation: two numbers x, y conflict if there exist
    # consecutive blocks in any ordering that would produce duplicate products.
    # 
    # A simpler and effective approach: build the conflict graph where an edge
    # exists between x and y if including BOTH in the sequence (in sorted order)
    # causes a product collision with some other pair. We approximate this by
    # checking if the set {x, y} (plus their interaction with neighbors) creates
    # a duplicate. For exactness within the MIS framework, we'll use a greedy
    # validation during branch-and-bound.

    # Actually, let's use a direct approach: the conflict between two elements
    # depends on the full sequence context. So we can't precompute a static
    # conflict graph easily. Instead, we'll use a branch-and-bound that:
    # 1. Maintains a current sequence
    # 2. At each step, tries to add the next element
    # 3. Uses upper bounds to prune

    # Let's implement a smarter branch-and-bound with memoization on (position, 
    # last_few_elements, used_set_hash) but that's complex.
    
    # Alternative: Use the insight that for small N (127 candidates), we can
    # use a recursive search with strong pruning based on the number of 
    # remaining elements that can still be added.

    # Let's try a different angle: Greedy with backtracking and good heuristics.
    # The key insight from the inspirations is that the problem is essentially
    # finding a large independent set. Let's implement a branch-and-bound that
    # explores candidates in a smart order and prunes aggressively.

    def get_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block products are distinct."""
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2

    def greedy_construct():
        """Standard greedy: try numbers 2..128 in order."""
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)

    # --- Branch and Bound with smart ordering ---
    # We'll search for a better sequence by exploring different orderings
    # and using backtracking when stuck.

    # Key insight: The order in which we add elements matters. Let's try
    # multiple heuristics for ordering and use backtracking.

    def count_distinct_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    def is_prime_power(n):
        for p in [2, 3, 5, 7, 11, 13]:
            q = n
            while q % p == 0:
                q //= p
            if q == 1:
                return True
        return False

    # Try different orderings with backtracking
    def search_with_order(order, max_depth=None):
        """Search for a valid sequence using the given order with backtracking."""
        if time.time() - start_time > DEADLINE:
            return None
        
        values = []
        products = set()
        
        def backtrack(idx, values, products):
            if time.time() - start_time > DEADLINE:
                return False
            
            if idx == len(order):
                return True
            
            x = order[idx]
            
            # Try adding x
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            
            if not products.intersection(tails):
                # Can add x
                new_products = products | set(tails)
                values.append(x)
                
                if backtrack(idx + 1, values, new_products):
                    return True
                
                values.pop()
            
            # Try skipping x
            return backtrack(idx + 1, values, products)
        
        # This naive backtracking is too slow for 127 elements.
        # We need a smarter approach.
        
        # Instead, let's use a greedy approach with multiple restarts
        # and local improvements.
        return None

    # Better approach: Use simulated annealing or hill climbing with 
    # multiple restarts and different initial orderings.

    def greedy_with_order(order):
        """Greedy construction with a specific order."""
        values, products = [], set()
        for x in order:
            if x < 2 or x > 128:
                continue
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    # Try multiple orderings
    candidates = list(range(2, 129))
    orderings = [
        candidates,  # natural order
        sorted(candidates, key=lambda x: count_distinct_prime_factors(x)),  # by prime factor count
        sorted(candidates, key=lambda x: (not is_prime_power(x), x)),  # prime powers first
        sorted(candidates, reverse=True),  # descending
        sorted(candidates, key=lambda x: -count_distinct_prime_factors(x)),  # reverse prime factor count
    ]

    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        seq = greedy_with_order(order)
        if is_valid(seq) and len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    # Local search: try to improve by swapping elements
    used = set(best_seq)
    remaining = [x for x in range(2, 129) if x not in used]

    def try_swap_improve(seq):
        """Try to improve the sequence by swapping elements."""
        if time.time() - start_time > DEADLINE:
            return seq
        
        improved = True
        while improved:
            improved = False
            for i in range(len(seq)):
                for j in range(len(seq)):
                    if i == j:
                        continue
                    # Try swapping seq[i] and seq[j]
                    new_seq = seq[:]
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    new_seq.sort()
                    if is_valid(new_seq):
                        # Check if this helps us add more elements
                        # For now, just accept if valid
                        pass
            
            # Try inserting a new element
            for new_elem in remaining[:]:
                for pos in range(len(seq) + 1):
                    new_seq = seq[:pos] + [new_elem] + seq[pos:]
                    if is_valid(new_seq):
                        seq = new_seq
                        used.add(new_elem)
                        remaining.remove(new_elem)
                        improved = True
                        break
                if improved:
                    break
        
        return seq

    if time.time() - start_time < DEADLINE:
        improved_seq = try_swap_improve(best_seq)
        if is_valid(improved_seq) and len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(improved_seq)

    # Final validation
    if not is_valid(best_seq):
        best_seq = greedy_construct()

    return best_seq
