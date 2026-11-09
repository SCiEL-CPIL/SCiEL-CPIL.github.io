import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    all_numbers = list(range(2, 129))
    
    def check_valid(seq):
        """Check if strictly increasing sequence has distinct products for all blocks."""
        n = len(seq)
        if n == 0:
            return True
        
        prefix = [1] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] * seq[i]
        
        seen_products = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod = prefix[j] // prefix[i]
                if prod in seen_products:
                    return False
                seen_products.add(prod)
        return True

    def get_state(seq):
        """Compute prefix products and set of all block products."""
        n = len(seq)
        prefix = [1] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] * seq[i]
        
        prod_set = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                prod_set.add(prefix[j] // prefix[i])
                
        return prefix, prod_set

    def try_insert(seq, c, prefix, prod_set):
        """
        Try to insert c into seq.
        Returns (new_seq, new_prefix, new_prod_set) if valid, else None.
        """
        n = len(seq)
        # Find insertion index
        idx = 0
        for i in range(n):
            if seq[i] < c:
                idx = i + 1
            else:
                break
        
        # Check if c is already in seq
        if idx < n and seq[idx] == c:
            return None
        
        new_seq = seq[:idx] + [c] + seq[idx:]
        
        # Calculate new prefix products
        new_prefix = [1] * (n + 2)
        # Copy old prefixes up to idx
        for i in range(idx + 1):
            new_prefix[i] = prefix[i]
        
        # Compute new prefixes
        new_prefix[idx + 1] = prefix[idx] * c
        for i in range(idx + 1, n + 1):
            new_prefix[i + 1] = new_prefix[i] * seq[i - 1] # seq[i-1] is the element at index i in new_seq? 
            # Let's trace:
            # new_seq = [s0...s_{idx-1}, c, s_idx...s_{n-1}]
            # new_prefix[0] = 1
            # new_prefix[idx+1] = prefix[idx] * c
            # new_prefix[idx+2] = new_prefix[idx+1] * s_idx
            # new_prefix[i+1] = new_prefix[i] * new_seq[i]
            # new_seq[i] for i > idx is seq[i-1]
            # So for i from idx+1 to n:
            # new_prefix[i+1] = new_prefix[i] * seq[i-1] is correct for i >= idx+1?
            # Let i = idx+1. new_prefix[idx+2] = new_prefix[idx+1] * seq[idx]. Correct.
            # Let i = n. new_prefix[n+1] = new_prefix[n] * seq[n-1]. Correct.
            
        # Check new products
        new_prod_set = set(prod_set)
        
        # New blocks are those that include c.
        # A block is [i, j) in new_seq.
        # It includes c (at index idx) iff i <= idx < j.
        # So i in [0, idx], j in [idx+1, n+1].
        
        for i in range(idx + 1):
            for j in range(idx + 1, n + 2):
                prod = new_prefix[j] // new_prefix[i]
                if prod in new_prod_set:
                    return None
                new_prod_set.add(prod)
                
        return new_seq, new_prefix, new_prod_set

    def local_search(seq, max_iters=500):
        """
        Improve sequence by insertions.
        """
        current_seq = seq[:]
        current_prefix, current_prod_set = get_state(current_seq)
        used = set(current_seq)
        remaining = [x for x in all_numbers if x not in used]
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try insertions
            # Sort remaining by value to try smaller numbers first? Or random?
            # Let's try all and pick the one that allows the most future insertions?
            # That's expensive. Just try all valid insertions and pick one.
            # Heuristic: Pick the one that minimizes the "gap" or something?
            # For now, just pick the first valid one in sorted order.
            
            candidates = []
            for c in remaining:
                if time.time() > deadline:
                    break
                res = try_insert(current_seq, c, current_prefix, current_prod_set)
                if res is not None:
                    candidates.append((c, res))
            
            if candidates:
                # Strategy: Pick the candidate that results in the smallest new products?
                # Or just pick the smallest c?
                # Smallest c is a good heuristic to keep numbers small.
                candidates.sort(key=lambda x: x[0])
                c, (new_seq, new_prefix, new_prod_set) = candidates[0]
                
                current_seq = new_seq
                current_prefix = new_prefix
                current_prod_set = new_prod_set
                used = set(current_seq)
                remaining = [x for x in all_numbers if x not in used]
                improved = True
            
            # If no insertion, try a swap?
            # Swaps are harder to implement efficiently with the prefix product trick.
            # But we can do: remove one, try insert another.
            if not improved and remaining:
                # Try to swap: remove x from seq, insert y from remaining
                # This is expensive: O(N * R * (N+R))
                # Let's limit this.
                # Try removing the largest element? Or random?
                # Let's try a few random swaps.
                swap_found = False
                for _ in range(10): # Limit swap attempts
                    if time.time() > deadline:
                        break
                    if not current_seq:
                        break
                    idx_to_remove = random.randint(0, len(current_seq) - 1)
                    x_to_remove = current_seq[idx_to_remove]
                    
                    temp_seq = current_seq[:idx_to_remove] + current_seq[idx_to_remove+1:]
                    temp_prefix, temp_prod_set = get_state(temp_seq)
                    
                    for y in remaining:
                        res = try_insert(temp_seq, y, temp_prefix, temp_prod_set)
                        if res is not None:
                            # Accept swap
                            current_seq, current_prefix, current_prod_set = res
                            used = set(current_seq)
                            remaining = [x for x in all_numbers if x not in used]
                            improved = True
                            swap_found = True
                            break
                    if swap_found:
                        break
                        
        return current_seq

    # Main Search
    best_seq = []
    best_len = 0
    
    # 1. Multiple greedy starts
    orders = [
        all_numbers,
        sorted(all_numbers, reverse=True),
        sorted(all_numbers, key=lambda x: (x % 2, x)), # Evens then odds?
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        seq = []
        prefix, prod_set = get_state(seq)
        used = set()
        
        for c in order:
            if c in used:
                continue
            res = try_insert(seq, c, prefix, prod_set)
            if res is not None:
                seq, prefix, prod_set = res
                used = set(seq)
        
        seq = local_search(seq)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # 2. Iterated Local Search
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        # Perturb best_seq
        perturbed = best_seq[:]
        num_removals = random.randint(1, 3)
        for _ in range(num_removals):
            if not perturbed:
                break
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
        
        # Local search from perturbed
        candidate = local_search(perturbed)
        
        if len(candidate) > best_len:
            best_len = len(candidate)
            best_seq = candidate[:]
            
    # Final verification
    if not check_valid(best_seq):
        best_seq = [2]

    return best_seq
