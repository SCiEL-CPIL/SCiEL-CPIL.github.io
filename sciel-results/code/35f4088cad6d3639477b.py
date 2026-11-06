import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for safety
    
    # Precompute primes? Not strictly necessary, but factorization can help 
    # if we were doing more complex number theory. Here, direct multiplication is fine.
    
    def get_all_products(seq):
        """
        Returns a set of all non-empty consecutive subarray products.
        """
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    def is_valid_insertion(seq, x, pos):
        """
        Checks if x can be inserted at pos in seq.
        seq is strictly increasing.
        pos is the index where x would be inserted (0 to len(seq)).
        """
        if pos > 0 and seq[pos-1] >= x:
            return False
        if pos < len(seq) and seq[pos] <= x:
            return False
            
        # If x is already in seq, it's not a valid "new" insertion in the context of increasing length
        # But we check unused numbers, so x is not in seq.
        
        # We need to check if any new product collides with existing products.
        # Existing products are from subarrays not involving x.
        # New products involve x.
        
        # Let L = seq[:pos], R = seq[pos:]
        # Suffixes of L: S_L = [L[-1], L[-2]*L[-1], ...]
        # Prefixes of R: P_R = [R[0], R[0]*R[1], ...]
        
        # New products:
        # 1. x
        # 2. S_L[i] * x  (for all suffixes of L)
        # 3. x * P_R[j]  (for all prefixes of R)
        # 4. S_L[i] * x * P_R[j] (for all combinations)
        
        # We need to verify:
        # a) None of these new products are in the set of products of L or R (excluding those that overlap? No, L and R are disjoint in indices, so their products are distinct sets? 
        #    Wait. Products of L are subarrays of L. Products of R are subarrays of R.
        #    The set of "old" products is Prods(L) U Prods(R).
        #    The new products must not be in Prods(L) U Prods(R).
        #    AND the new products must be distinct among themselves.
        
        # Let's compute Prods(L) and Prods(R)? 
        # That's O(N^2) again if we recompute.
        # However, N is small.
        # Let's just compute the set of all products of the new sequence and check for duplicates?
        # No, that's O(N^2).
        
        # Let's stick to the O(N) check if possible, but it's tricky to maintain the "old products" set efficiently without passing it around.
        # Given N ~ 35, O(N^2) is ~1200 ops.
        # If we do this for 100 candidates, it's 120k ops.
        # This is extremely fast in Python (microseconds to milliseconds).
        # So, a full check of the candidate sequence is likely the safest and simplest approach that is still fast enough.
        
        # Let's verify the speed of full check.
        # 1200 multiplications and set adds.
        # 100 candidates * 1200 = 120,000 operations.
        # Even if each op is 100ns, that's 12ms.
        # We can do ~80 full checks per second.
        # In 50 seconds, we can do ~4000 full checks.
        # Each insertion attempt involves checking ~100 unused numbers.
        # So we can do ~40 full "insertion passes".
        # This seems sufficient.
        
        # To be safe and faster, let's implement the O(N) check properly.
        # We need the set of products of L and R.
        # We can compute them on the fly? No, that defeats the purpose.
        # Let's just do the full check for simplicity and correctness, as N is small.
        
        new_seq = seq[:pos] + [x] + seq[pos:]
        return check_validity(new_seq)

    def check_validity(seq):
        if not seq:
            return True
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        
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

    def greedy_construct(randomize=False):
        values = []
        candidates = list(range(2, 129))
        
        for _ in range(len(candidates)):
            valid_candidates = []
            for x in candidates:
                if values and x <= values[-1]:
                    continue
                # Check if x can be appended
                # New products: x * suffix_prod for all suffixes of values
                # And x itself.
                # We need to check against ALL existing products.
                
                # Efficient check for appending:
                # Existing products set is not passed, so we'd need to recompute or maintain.
                # For greedy, we can just maintain the set.
                
                # Let's rewrite greedy to maintain the product set.
                pass # Placeholder, will implement below

    # Better Greedy
    def greedy_construct_impl(randomize=False):
        values = []
        products = set()
        candidates = list(range(2, 129))
        
        for _ in range(len(candidates)):
            valid_candidates = []
            for x in candidates:
                if values and x <= values[-1]:
                    continue
                if x in products:
                    continue
                
                # Check x * suffix_prods
                prod = 1
                valid = True
                for a in reversed(values):
                    prod *= a
                    if x * prod in products:
                        valid = False
                        break
                    # Also, the new product x*prod itself must not collide with other new products?
                    # x*prod1 == x*prod2 => prod1==prod2. Suffixes are distinct.
                    # So new products are distinct among themselves.
                    # We just need to check against existing `products`.
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if randomize:
                # Pick a random valid candidate
                x = random.choice(valid_candidates)
            else:
                x = min(valid_candidates)
            
            values.append(x)
            
            # Update products
            prod = 1
            for a in reversed(values):
                prod *= a
                products.add(prod)
            
            candidates.remove(x)
        
        return values

    best_seq = greedy_construct_impl(randomize=False)
    best_len = len(best_seq)
    
    # Local Search
    # We will run multiple trials.
    # In each trial, we start with a randomized greedy sequence (or the best one) 
    # and try to insert unused elements.
    
    for trial in range(2000): # High number, limited by time
        if time.time() > deadline:
            break
        
        # Start with a randomized greedy sequence to explore different basins
        # Or start with best_seq? 
        # Starting with best_seq might get stuck. 
        # Let's alternate or just randomize.
        # Let's use a randomized greedy start.
        current_seq = greedy_construct_impl(randomize=True)
        
        # If this start is already better, update best
        if len(current_seq) > best_len:
            best_seq = current_seq
            best_len = len(current_seq)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            used = set(current_seq)
            unused = [x for x in range(2, 129) if x not in used]
            random.shuffle(unused)
            
            for u in unused:
                if time.time() > deadline:
                    break
                
                # Find insertion position
                pos = bisect.bisect_left(current_seq, u)
                
                # Check bounds
                if pos < len(current_seq) and current_seq[pos] == u:
                    continue # Should not happen as u is unused
                
                # Try inserting at pos
                # Optimization: Only try if it's strictly between neighbors
                if pos > 0 and current_seq[pos-1] >= u:
                    continue
                if pos < len(current_seq) and current_seq[pos] <= u:
                    continue
                
                new_seq = current_seq[:pos] + [u] + current_seq[pos:]
                
                if check_validity(new_seq):
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_seq = current_seq
                        best_len = len(current_seq)
                    improved = True
                    # Restart the while loop to try other insertions with the new length
                    break
        
        # If we found a better sequence, we might want to try to improve it further
        # But the loop structure already does this: if improved, it loops back.
        # If not improved, it exits the while loop and starts a new trial.

    return best_seq
