import time
import random

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

def is_valid(seq):
    if not seq:
        return True
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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [p for p in range(2, 129) if is_prime(p)]
    composites = [c for c in range(2, 129) if not is_prime(c)]
    
    best_set = set(primes)
    best_len = len(primes)
    
    # Helper to compute all products of a sequence
    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    # Helper to check if adding c to seq is valid, returns (is_valid, new_products)
    def try_add(seq, c, current_prods):
        # seq is sorted
        # Find insertion point
        idx = 0
        while idx < len(seq) and seq[idx] < c:
            idx += 1
        
        left = seq[:idx]
        right = seq[idx:]
        
        new_prods = set()
        # Singleton
        new_prods.add(c)
        
        # Blocks ending at c (from left)
        prod = c
        for x in reversed(left):
            prod *= x
            new_prods.add(prod)
            
        # Blocks starting at c (to right)
        prod = c
        for x in right:
            prod *= x
            new_prods.add(prod)
            
        # Blocks crossing c
        # We need to combine suffix of left and prefix of right with c
        # Actually, the above loops cover:
        # [.., x_i, c] and [c, y_j, ..]
        # What about [x_i, c, y_j]?
        # My loops:
        # 1. c
        # 2. x_k c, x_{k-1} x_k c, ...
        # 3. c y_l, c y_l y_{l+1}, ...
        # Missing: x_k c y_l, x_{k-1} x_k c y_l, etc.
        
        # Let's generate all blocks containing c properly.
        # A block containing c is defined by start i in left (or c) and end j in right (or c).
        # If start is in left: prod_start = product(seq[i:idx]) * c
        # If end is in right: prod_end = c * product(right[:j+1])
        # Total prod = prod_start * product(right[:j+1]) ? No.
        
        # Let's just generate all new products by iterating start in [0..idx] and end in [idx..n-1]
        # where the block includes index idx (which is c).
        
        new_prods = set()
        n = len(seq)
        # c is at index idx in the new sequence
        # Block [i, j] in new seq corresponds to:
        # if i < idx < j: prod = (prod seq[i:idx]) * c * (prod seq[idx:j+1])
        # if i == idx < j: prod = c * (prod seq[idx:j+1])
        # if i < idx == j: prod = (prod seq[i:idx]) * c
        # if i == idx == j: prod = c
        
        # Precompute suffix products of left
        # left_suffix[k] = product(left[k:])
        left_suffix = [1] * (idx + 1)
        if idx > 0:
            left_suffix[idx-1] = left[-1]
            for k in range(idx-2, -1, -1):
                left_suffix[k] = left[k] * left_suffix[k+1]
        # left_suffix[idx] = 1 (empty)
        
        # Precompute prefix products of right
        right_prefix = [1] * (len(right) + 1)
        if len(right) > 0:
            right_prefix[1] = right[0]
            for k in range(1, len(right)):
                right_prefix[k+1] = right[k] * right_prefix[k]
        # right_prefix[0] = 1 (empty)
        
        # Generate all products
        for i in range(idx + 1): # i is start index in new seq (0 to idx)
            # Product from i to idx-1 in old seq (left part)
            p_left = left_suffix[i] if i < idx else 1
            for j in range(idx, n + 1): # j is end index in new seq (idx to n-1)
                # Product from idx to j-1 in old seq (right part)
                # Note: right part starts at old index idx.
                # In new seq, c is at idx.
                # Right part of block is right[0 : j-idx]
                p_right = right_prefix[j - idx]
                
                prod = p_left * c * p_right
                new_prods.add(prod)
                
        # Check collisions with existing products
        if new_prods & current_prods:
            return False, None
            
        # Check internal collisions
        # Number of blocks containing c is (idx + 1) * (n - idx + 1)
        # Wait, i ranges 0..idx (idx+1 choices), j ranges idx..n-1 (n-idx choices).
        # Total blocks = (idx + 1) * (n - idx)
        if len(new_prods) != (idx + 1) * (n - idx):
            return False, None
            
        return True, new_prods

    # Main search
    current_set = set(primes)
    current_seq = sorted(current_set)
    current_prods = get_products(current_seq)
    
    remaining = composites.copy()
    random.shuffle(remaining)
    
    # Greedy phase
    for c in remaining:
        if time.time() > deadline:
            break
        if c in current_set:
            continue
        is_ok, new_prods = try_add(current_seq, c, current_prods)
        if is_ok:
            current_set.add(c)
            current_seq = sorted(current_set)
            current_prods = new_prods
            if len(current_set) > best_len:
                best_len = len(current_set)
                best_set = set(current_set)
                
    # Local search / Swaps
    # We will try to swap a prime for a composite to allow more additions later
    # Or just random restarts with different orders
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        current_set = set(primes)
        current_seq = sorted(current_set)
        current_prods = get_products(current_seq)
        
        # Randomly shuffle composites
        cands = composites.copy()
        random.shuffle(cands)
        
        for c in cands:
            if time.time() > deadline:
                break
            if c in current_set:
                continue
            is_ok, new_prods = try_add(current_seq, c, current_prods)
            if is_ok:
                current_set.add(c)
                current_seq = sorted(current_set)
                current_prods = new_prods
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = set(current_set)
            
        # Try one swap step
        # Pick a random prime p in current_set
        # Pick a random composite c not in current_set
        # If removing p allows adding c, do it?
        # This is complex. Let's stick to random restarts for now, but with more of them.
        
    return sorted(best_set)
