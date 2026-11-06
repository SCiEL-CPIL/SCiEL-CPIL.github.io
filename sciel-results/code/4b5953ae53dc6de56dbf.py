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
    
    def get_score(seq):
        """
        Returns a score where higher is better.
        Penalizes duplicates heavily.
        """
        products = set()
        n = len(seq)
        num_dups = 0
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    num_dups += 1
                else:
                    products.add(prod)
        
        if num_dups == 0:
            return n
        else:
            return -10000 * num_dups + n

    def compute_products(seq):
        """Compute set of all subarray products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    primes = get_primes(128)
    all_nums = list(range(2, 129))
    
    # Initialize with primes
    current_seq = primes[:]
    current_products = compute_products(current_seq)
    current_score = len(current_seq) # Since primes are unique and distinct products, it's valid
    
    # Greedy initial construction: try adding composites
    composites = [x for x in all_nums if x not in current_seq]
    
    # Sort composites: prefer those with fewer prime factors? 
    # Or just try all and see which fit.
    # To make it faster, we can try to add in a specific order.
    # Let's try adding in increasing order, but check validity.
    
    def try_add(x, seq, products):
        """
        Check if x can be added to seq.
        Returns (new_seq, new_products) if valid, else (None, None).
        """
        # Find insertion point
        # Since seq is sorted, we can binary search or linear scan
        # Linear scan is fine for n~30
        k = 0
        while k < len(seq) and seq[k] < x:
            k += 1
        
        # New subarrays containing x:
        # seq[i..j] in new_seq where i <= k <= j
        # These are formed by:
        # suffix of seq[0..k-1] * x * prefix of seq[k..n-1]
        # Wait, indices in new_seq:
        # x is at index k.
        # Left part: seq[0..k-1]
        # Right part: seq[k..n-1]
        
        # Products of subarrays ending at k-1 in old seq (suffixes)
        left_prods = [1]
        prod = 1
        for i in range(k-1, -1, -1):
            prod *= seq[i]
            left_prods.append(prod)
            
        # Products of subarrays starting at k in old seq (prefixes)
        right_prods = [1]
        prod = 1
        for i in range(k, len(seq)):
            prod *= seq[i]
            right_prods.append(prod)
            
        new_prods = set()
        for lp in left_prods:
            for rp in right_prods:
                p = lp * x * rp
                # Check if this product already exists in old products
                if p in products:
                    return None, None
                # Check if this product collides with other new products
                if p in new_prods:
                    return None, None
                new_prods.add(p)
                
        # If we reach here, no conflicts
        new_seq = seq[:k] + [x] + seq[k:]
        new_products = products.union(new_prods)
        return new_seq, new_products

    # Greedy Add Phase
    # Try to add composites. Order matters.
    # Let's sort by value, or by "complexity"
    # A number with more small prime factors is more likely to collide?
    # Let's just try all composites in increasing order.
    for x in composites:
        if time.time() > deadline - 5:
            break
        new_seq, new_products = try_add(x, current_seq, current_products)
        if new_seq is not None:
            current_seq = new_seq
            current_products = new_products
            current_score = len(current_seq)
            
    best_seq = current_seq[:]
    best_score = current_score
    best_products = current_products.copy()
    
    # Local Search / Simulated Annealing
    # We need a fast validity check for swaps/removals.
    # For removal, we can recompute products if it's fast enough, or use a heuristic.
    # Given n is small (~30-40), recomputing O(n^2) products takes ~1000-1600 ops.
    # 60 seconds allows for ~10^6-10^7 ops. So ~1000-10000 recomputations.
    # This is tight. We should use incremental checks where possible.
    
    # Let's do a randomized local search.
    # Move: Add, Remove, Swap.
    
    T = 50.0
    cooling = 0.999
    
    while time.time() < deadline - 2:
        # Pick a move
        # Weight towards Add if we have room, Swap otherwise
        # We can't exceed 127 elements, but we are limited by uniqueness.
        
        # Generate candidate
        move_type = random.choices(['add', 'remove', 'swap'], weights=[0.3, 0.1, 0.6])[0]
        
        new_seq = None
        new_products = None
        new_score = 0
        
        if move_type == 'add':
            candidates = [x for x in all_nums if x not in current_seq]
            if candidates:
                x = random.choice(candidates)
                new_seq, new_products = try_add(x, current_seq, current_products)
                if new_seq:
                    new_score = len(new_seq)
        
        elif move_type == 'remove':
            if len(current_seq) > 2:
                idx = random.randint(0, len(current_seq)-1)
                x = current_seq[idx]
                new_seq = current_seq[:idx] + current_seq[idx+1:]
                # Recompute products for removal? 
                # It's safer to just recompute if the set is small.
                # Or we can note that removing an element removes products containing it.
                # But we don't have a structure to easily remove them from the set.
                # So recompute.
                if time.time() < deadline - 2:
                    new_products = compute_products(new_seq)
                    new_score = len(new_seq)
                    # Check validity? compute_products doesn't check for duplicates.
                    # We need to check if new_products has duplicates? 
                    # No, a set by definition has no duplicates. 
                    # But wait, the condition is "Products ... must be distinct".
                    # If we compute the set, we lose the count of duplicates.
                    # We need to know if the *list* of products has duplicates.
                    # My compute_products returns a set. This is wrong for scoring if there are dups.
                    # However, if we maintain the set correctly, we know it's valid.
                    # For removal, the resulting sequence is a subsequence of a valid sequence?
                    # No, removing an element from a valid sequence yields a valid sequence?
                    # Yes! If all subarray products were distinct in S, then any subsequence of S (formed by removing elements) will also have distinct subarray products?
                    # Let's verify.
                    # S = [a, b, c]. Prods: a, b, c, ab, bc, abc.
                    # Remove b -> [a, c]. Prods: a, c, ac.
                    # Are any of these equal to each other? No, because they were distinct in S.
                    # Is it possible that ac == a? No, c != 1.
                    # Is it possible that ac == c? No, a != 1.
                    # Is it possible that ac == ab? No, c != b.
                    # So yes, any subsequence of a valid sequence is valid.
                    # So removal is always valid!
                    # Score decreases by 1.
        
        elif move_type == 'swap':
            if len(current_seq) > 0:
                idx_out = random.randint(0, len(current_seq)-1)
                x_out = current_seq[idx_out]
                candidates = [x for x in all_nums if x not in current_seq]
                if candidates:
                    x_in = random.choice(candidates)
                    # Construct new seq
                    # Remove x_out, add x_in, sort
                    temp = [x for x in current_seq if x != x_out]
                    # Insert x_in in sorted order
                    k = 0
                    while k < len(temp) and temp[k] < x_in:
                        k += 1
                    new_seq = temp[:k] + [x_in] + temp[k:]
                    
                    # Check validity of new_seq
                    # We can use a fast check.
                    # Since swap is expensive to check incrementally, let's do a fast full check.
                    # But we need to be careful with time.
                    # Let's only do full check if we have time.
                    if time.time() < deadline - 2:
                        # Fast check: compute set and see if len == n*(n+1)/2
                        prods = compute_products(new_seq)
                        n = len(new_seq)
                        if len(prods) == n*(n+1)//2:
                            new_products = prods
                            new_score = n
                        else:
                            new_score = -10000 # Invalid
        
        # Acceptance
        if new_seq is not None:
            delta = new_score - current_score
            if delta > 0:
                current_seq = new_seq
                if new_products is None: # e.g. removal
                    current_products = compute_products(current_seq)
                else:
                    current_products = new_products
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_seq = current_seq[:]
                    best_products = current_products.copy()
            else:
                if T > 0.1:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_seq = new_seq
                        if new_products is None:
                            current_products = compute_products(current_seq)
                        else:
                            current_products = new_products
                        current_score = new_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_seq
