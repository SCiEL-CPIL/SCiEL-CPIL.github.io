import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    def time_left():
        return deadline - time.time()

    def compute_all_products(seq):
        """Compute all non-empty consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid_sequence(seq):
        """Check if all subarray products are distinct."""
        if not seq:
            return True
        products = compute_all_products(seq)
        expected_count = len(seq) * (len(seq) + 1) // 2
        return len(products) == expected_count

    def greedy_construct():
        """Greedy construction to get an initial sequence."""
        seq = []
        products_set = set()
        
        for x in range(2, 129):
            if time_left() < 5.0:
                break
            # Check if x can be appended
            new_prods = []
            prod = x
            for a in reversed(seq):
                prod *= a
                new_prods.append(prod)
            
            if not any(p in products_set for p in new_prods):
                seq.append(x)
                products_set.update(new_prods)
        
        return seq

    def backtrack_search(start_idx, seq, products_set, best_seq_ref):
        """
        Backtracking search to maximize length.
        """
        if time_left() < 2.0:
            return
        
        if len(seq) > len(best_seq_ref):
            best_seq_ref[:] = seq[:]  # Update in place
        
        # Pruning: if we can't beat the best length, stop
        if len(seq) + (128 - start_idx + 1) <= len(best_seq_ref):
            return
        
        for x in range(start_idx, 129):
            if time_left() < 2.0:
                return
            
            # Check validity
            new_prods = []
            prod = x
            for a in reversed(seq):
                prod *= a
                new_prods.append(prod)
            
            valid = True
            for p in new_prods:
                if p in products_set:
                    valid = False
                    break
            
            if valid:
                new_products_set = products_set | set(new_prods)
                seq.append(x)
                backtrack_search(x + 1, seq, new_products_set, best_seq_ref)
                seq.pop()

    def insertion_search(seq):
        """
        Try to insert elements into gaps to increase length.
        """
        improved = True
        while improved and time_left() > 10.0:
            improved = False
            n = len(seq)
            # Try inserting in each gap
            for i in range(n + 1):
                if time_left() < 10.0:
                    break
                
                # Determine valid range for x
                # If i == 0, x > 1 (and < seq[0])
                # If i == n, x > seq[n-1]
                # Else, x > seq[i-1] and x < seq[i]
                
                low = seq[i-1] + 1 if i > 0 else 2
                high = seq[i] - 1 if i < n else 128
                
                for x in range(low, high + 1):
                    if time_left() < 10.0:
                        break
                    
                    # Create new sequence
                    new_seq = seq[:i] + [x] + seq[i:]
                    
                    # Check validity
                    if is_valid_sequence(new_seq):
                        seq = new_seq
                        improved = True
                        break # Restart loop with new seq
        
        return seq

    # Phase 1: Greedy
    best_seq = greedy_construct()
    
    # Phase 2: Backtracking
    # We use a list to allow in-place update in recursive function
    best_seq_ref = list(best_seq)
    
    # To make backtracking more effective, we can run it from scratch
    # or from the greedy result. Running from scratch with pruning is safer.
    backtrack_search(2, [], set(), best_seq_ref)
    
    if len(best_seq_ref) > len(best_seq):
        best_seq = best_seq_ref
    
    # Phase 3: Insertion Search
    if time_left() > 15.0:
        best_seq = insertion_search(best_seq)
        
    # Final validation
    if best_seq:
        if not is_valid_sequence(best_seq):
            # Fallback to greedy if something went wrong
            best_seq = greedy_construct()
            
    return best_seq
