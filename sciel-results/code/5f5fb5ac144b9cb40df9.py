import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # 1. Identify all primes in [2, 128]
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]
    
    # Start with all primes
    seq = primes[:]
    
    # Function to calculate all block products for a sequence
    def get_products(s):
        prods = set()
        n = len(s)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= s[j]
                prods.add(p)
        return prods
        
    current_products = get_products(seq)
    
    # Try to insert composites
    # Sort composites by value to maintain increasing order insertion logic?
    # We just iterate through composites in increasing order.
    # For each composite, find insertion point.
    
    # To speed up, we can pre-compute insertion points?
    # Since seq is sorted, insertion point is bisect.
    import bisect
    
    for c in composites:
        if time.time() > deadline:
            break
            
        # Find insertion index
        idx = bisect.bisect_left(seq, c)
        
        # Candidate sequence
        new_seq = seq[:idx] + [c] + seq[idx:]
        
        # Calculate new products
        # Optimization: Only calculate products involving c?
        # Products not involving c are the same as before.
        # We need to check:
        # 1. New products involving c are not in current_products.
        # 2. New products involving c are distinct from each other.
        
        new_prods = set()
        # Products ending at c
        # Blocks [i, idx] in new_seq where new_seq[idx] == c
        # In old seq indices, this corresponds to suffixes of seq[:idx]
        
        # Let's just compute all new products for simplicity first, as L is small (< 100)
        # L=31, products ~500. L=50, products ~1300.
        # Computing 1300 products is fast.
        
        # But we can be smarter.
        # The new products are:
        # 1. c itself
        # 2. c * (suffix of left part)
        # 3. (prefix of right part) * c
        # 4. (suffix of left part) * c * (prefix of right part)
        
        # Actually, it's easier to just compute the set of products for new_seq
        # and check if len(new_prods) == len(new_seq)*(len(new_seq)+1)//2
        # AND if new_prods is a superset of current_products? No, just check intersection.
        
        # If we compute all products for new_seq, we can check:
        # if len(new_prods) == len(new_seq) * (len(new_seq) + 1) // 2:
        #     # All distinct
        #     # And since we only added c, any product not involving c is from old seq.
        #     # So we just need to ensure no collision with old products.
        #     # Actually, if all are distinct in new_seq, and old products are a subset of new products?
        #     # No, old products are still there.
        #     # If new_prods has size L_new*(L_new+1)/2, it means all new blocks are distinct.
        #     # Does it guarantee no collision with old blocks?
        #     # Yes, because old blocks are a subset of the blocks in new_seq?
        #     # No! Inserting c splits an old block into two, and creates new ones.
        #     # Old blocks that don't include c are unchanged.
        #     # Old blocks that included the gap are split.
        #     # So the set of products is NOT a superset.
        #     # We must check if any NEW product (involving c) equals any OLD product (not involving c).
        
        # Let's identify products involving c.
        # Left part: seq[:idx]
        # Right part: seq[idx:]
        
        left_suffixes = []
        p = 1
        for i in range(idx - 1, -1, -1):
            p *= seq[i]
            left_suffixes.append(p)
            
        right_prefixes = []
        p = 1
        for i in range(idx, len(seq)):
            p *= seq[i]
            right_prefixes.append(p)
            
        # New products involving c:
        # 1. c
        # 2. c * s (s in left_suffixes)
        # 3. s * c (s in right_prefixes)
        # 4. s1 * c * s2 (s1 in left_suffixes, s2 in right_prefixes)
        
        candidates = set()
        candidates.add(c)
        for s in left_suffixes:
            candidates.add(c * s)
        for s in right_prefixes:
            candidates.add(s * c)
        for s1 in left_suffixes:
            for s2 in right_prefixes:
                candidates.add(s1 * c * s2)
                
        # Check if any candidate is in current_products
        if candidates.intersection(current_products):
            continue
            
        # Check if candidates are distinct from each other
        if len(candidates) != len(left_suffixes) * (len(right_prefixes) + 1) + (len(right_prefixes) + 1):
            # Formula: 1 (c) + len(L) + len(R) + len(L)*len(R)
            # Actually, the set 'candidates' might have duplicates if e.g. c*s1 == c*s2?
            # Since s1, s2 are distinct suffixes/prefixes of distinct numbers?
            # Suffixes are distinct because numbers are distinct and > 1.
            # Prefixes are distinct.
            # So c*s1 are distinct. s2*c are distinct.
            # s1*c*s2 are distinct?
            # Yes, unique factorization.
            # So len(candidates) should be 1 + len(L) + len(R) + len(L)*len(R)
            expected_count = 1 + len(left_suffixes) + len(right_prefixes) + len(left_suffixes) * len(right_prefixes)
            if len(candidates) != expected_count:
                continue
                
        # If we reach here, it's a valid insertion
        seq = new_seq
        current_products = get_products(seq) # Update to full set for next iterations
        
    return seq
