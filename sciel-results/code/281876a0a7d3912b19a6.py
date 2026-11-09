import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Helper to check if inserting x at index i into seq is valid
    # given the current set of products 'cur_prods'
    def can_insert(seq, x, i, cur_prods):
        n = len(seq)
        # New subarrays must include x
        # x is at index i in the new sequence
        # Start s from 0 to i
        # End e from i to n (in new sequence, which is len(seq)+1)
        # Note: indices in seq are 0..n-1
        # If i < n, seq[i] is the element after x
        # If i == n, x is at the end
        
        new_prods = set()
        
        # Precompute prefix products of seq up to i
        # and suffix products from i to n
        # Actually, just compute on the fly
        
        for s in range(0, i + 1):
            left_prod = 1
            for k in range(s, i):
                left_prod *= seq[k]
            
            # Now extend to the right
            # The subarray is seq[s:i] + [x] + seq[i:e]
            # e goes from i to n
            right_prod = 1
            for e in range(i, n + 1):
                if e > i:
                    right_prod *= seq[e-1] # seq[i] is at index i in new seq? 
                    # Wait. 
                    # New sequence: seq[0]...seq[i-1], x, seq[i]...seq[n-1]
                    # If we pick end index e in NEW sequence:
                    # If e == i: subarray is seq[s:i] + [x]
                    # If e > i: subarray is seq[s:i] + [x] + seq[i:e-1] (using old seq indices)
                    
                # Let's re-index carefully.
                # New index i holds x.
                # New indices 0..i-1 map to seq 0..i-1.
                # New indices i+1..n map to seq i..n-1.
                
                # Subarray [s, e] in new sequence.
                # s <= i <= e.
                
                # Product = (prod seq[s:i]) * x * (prod seq[i:e-1])
                
                # Let's just calculate the product directly for simplicity, N is small.
                pass
        
        # Simpler implementation:
        # Generate all new subarrays products
        # Check for duplicates in new_prods
        # Check for intersection with cur_prods
        
        # Since N is small, we can just build the new sequence and run is_valid?
        # No, is_valid is O(N^3).
        # But can_insert with O(N^2) is better.
        
        # Let's implement the O(N^2) check properly.
        
        # Prefix products of seq
        # Suffix products of seq
        
        # Actually, let's just use the full is_valid for the first version to ensure correctness,
        # and optimize if time permits. Given the 85s limit and fast O(N^2) check,
        # let's stick to a robust implementation.
        
        # We'll use a helper that takes the full new sequence and checks validity.
        # But we can optimize by only checking the new subarrays.
        
        # Let's implement the efficient check.
        
        # 1. Compute products of all subarrays in the new sequence that contain x.
        # 2. Check if any are in cur_prods.
        # 3. Check if any are duplicates in the new set.
        
        new_prods_list = []
        
        # s: start index in new seq (0 to i)
        # e: end index in new seq (i to n)
        
        for s in range(0, i + 1):
            # Compute product of seq[s:i]
            left_p = 1
            for k in range(s, i):
                left_p *= seq[k]
            
            # Now extend right
            # The part after x is seq[i:e-1]
            # If e == i, the part is empty (prod 1)
            # If e == i+1, the part is seq[i]
            
            right_p = 1
            for e in range(i, n + 1):
                # e is the end index in the NEW sequence
                # The elements after x are new_seq[i+1...e]
                # Which correspond to seq[i ... e-1]
                
                # We need prod of seq[i ... e-1]
                # Let's compute this incrementally
                
                # Wait, the loop structure above is slightly off for incremental right_p.
                # Let's reset right_p for each s? No, right_p depends on e.
                
                # Let's just compute the product of the segment seq[i:e-1]
                # This is O(N) per e. Total O(N^2) per s. Total O(N^3).
                # Back to O(N^3).
                
                pass
        
        # Given the complexity of optimization and the small N,
        # let's just use is_valid(new_seq) which is O(N^3) ~ 27000 ops.
        # 100 candidates * 30 positions * 27000 = 8.1 * 10^7 ops.
        # This is ~0.1s in C, but Python is slower. Maybe 1-2s per round.
        # We have 85s. So ~40 rounds.
        # This is sufficient.
        
        return True # Placeholder

    # 1. Greedy
    values = []
    products = set()
    for x in range(2, 129):
        if time.time() > deadline: break
        new_prods = []
        p = 1
        for a in reversed(values):
            p *= a
            new_prods.append(x * p)
        new_prods.append(x)
        if products.intersection(new_prods):
            continue
        values.append(x)
        products.update(new_prods)
        
    best_seq = list(values)
    
    # 2. Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in best_seq]
        for x in candidates:
            if time.time() > deadline: break
            for i in range(len(best_seq) + 1):
                new_seq = best_seq[:i] + [x] + best_seq[i:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    improved = True
                    break
            if improved: break
            
    return best_seq

def is_valid(seq):
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
