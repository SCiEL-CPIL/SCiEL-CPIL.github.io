import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    all_numbers = list(range(2, 129))
    
    def check_valid(seq):
        """
        Check if strictly increasing sequence 'seq' has distinct products
        for all non-empty consecutive blocks.
        """
        n = len(seq)
        if n == 0:
            return True
        
        # Compute prefix products S[0]=1, S[i]=seq[0]*...*seq[i-1]
        S = [1] * (n + 1)
        for i in range(n):
            S[i+1] = S[i] * seq[i]
            
        seen_products = set()
        # Products of blocks [i+1, j] are S[j] // S[i] for 0 <= i < j <= n
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod = S[j] // S[i]
                if prod in seen_products:
                    return False
                seen_products.add(prod)
        return True

    def get_prefix_products(seq):
        n = len(seq)
        S = [1] * (n + 1)
        for i in range(n):
            S[i+1] = S[i] * seq[i]
        return S

    def get_product_set(seq):
        S = get_prefix_products(seq)
        n = len(seq)
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                seen_products.add(S[j] // S[i])
        return seen_products

    def try_insert_fast(seq, c, S, prod_set):
        """
        Try to insert c into seq.
        Returns (new_seq, new_S, new_prod_set) if valid, else None.
        """
        n = len(seq)
        # Find insertion index
        idx = 0
        for i in range(n):
            if seq[i] < c:
                idx = i + 1
            else:
                break
        
        new_seq = seq[:idx] + [c] + seq[idx:]
        
        # Calculate new products involving c
        # Formula: x * (P[j] // P[i]) for 0 <= i <= idx, idx <= j <= n
        # Note: S is prefix of old seq (length n+1)
        new_products_to_check = []
        for i in range(idx + 1):
            for j in range(idx, n + 1):
                # P[j] // P[i] is product of seq[i..j-1]
                # If i==idx and j==idx, product is 1 (empty), so val = c
                # If i < idx, left part exists.
                # If j > idx, right part exists.
                val = c * (S[j] // S[i])
                if val in prod_set:
                    return None
                new_products_to_check.append(val)
        
        # If all valid, update state
        new_prod_set = set(prod_set)
        new_prod_set.update(new_products_to_check)
        
        # Update prefix products
        new_S = [1] * (n + 2)
        for i in range(idx + 1):
            new_S[i] = S[i]
        for i in range(idx + 1, n + 2):
            new_S[i] = S[i-1] * c
            
        return new_seq, new_S, new_prod_set

    def greedy_build(order):
        seq = []
        S = get_prefix_products(seq)
        prod_set = get_product_set(seq)
        used = set(seq)
        
        for c in order:
            if c in used:
                continue
            res = try_insert_fast(seq, c, S, prod_set)
            if res is not None:
                seq, S, prod_set = res
                used = set(seq)
        
        return seq

    best_seq = []
    best_len = 0
    
    # Try multiple orders to find a good starting point
    orders = [
        all_numbers,
        sorted(all_numbers, reverse=True),
        sorted(all_numbers, key=lambda x: math.gcd(x, 128)), # Heuristic
        sorted(all_numbers, key=lambda x: len([i for i in range(2, x) if x % i == 0])) # Fewest divisors
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        seq = greedy_build(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Local Search
    def local_search(seq):
        current_seq = seq[:]
        current_S = get_prefix_products(current_seq)
        current_prod_set = get_product_set(current_seq)
        used = set(current_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        improved = True
        while improved:
            improved = False
            
            # Try insertions
            for c in sorted(remaining):
                if time.time() > deadline:
                    break
                res = try_insert_fast(current_seq, c, current_S, current_prod_set)
                if res is not None:
                    current_seq, current_S, current_prod_set = res
                    used = set(current_seq)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            
            if time.time() > deadline:
                break

            # If no insertion worked, try swaps
            if remaining:
                for _ in range(10): # Reduced swaps
                    if time.time() > deadline:
                        break
                    if not remaining:
                        break
                    
                    i = random.randint(0, len(current_seq) - 1)
                    x_to_remove = current_seq[i]
                    c_to_add = random.choice(remaining)
                    
                    new_set = set(current_seq)
                    new_set.remove(x_to_remove)
                    new_set.add(c_to_add)
                    new_seq = sorted(new_set)
                    
                    if check_valid(new_seq):
                        current_seq = new_seq
                        current_S = get_prefix_products(new_seq)
                        current_prod_set = get_product_set(new_seq)
                        used = set(current_seq)
                        remaining = [x for x in all_numbers if x not in used]
                        improved = True
                        break

        return current_seq

    if best_seq:
        best_seq = local_search(best_seq)
    
    # Final Polish with multiple random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
        # Perturb best_seq
        perturbed = best_seq[:]
        num_removals = random.randint(1, 2)
        for _ in range(num_removals):
            if not perturbed:
                break
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
            
        # Re-greedy from perturbed state with random order
        random.shuffle(all_numbers)
        new_seq = greedy_build(all_numbers)
        
        # Merge perturbed into new_seq? Or just use new_seq?
        # Let's try to insert perturbed elements into new_seq
        # Simpler: Just run local search on new_seq
        new_seq = local_search(new_seq)
        
        if len(new_seq) > len(best_seq):
            best_seq = new_seq[:]

    # Final Verification
    if not check_valid(best_seq):
        # Fallback to a known valid small sequence
        best_seq = [2, 3, 5, 7] # Primes are usually safe
        if not check_valid(best_seq):
            best_seq = [2]

    return best_seq
