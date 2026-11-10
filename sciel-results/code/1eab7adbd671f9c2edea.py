"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _compute_products(values):
    """Compute all block products for a given sequence."""
    products = set()
    n = len(values)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            products.add(prod)
    return products

def _is_valid_insertion(values, pos, x):
    """Check if inserting x at pos keeps all block products distinct."""
    n = len(values)
    # We need to check all blocks that include x
    # Blocks ending at pos (new position of x): values[i..pos] for i in [0, pos]
    # Blocks starting at pos+1 (new position of x): values[pos+1..j] for j in [pos+1, n]
    # But actually, when we insert x at pos, the new sequence is values[:pos] + [x] + values[pos:]
    # We need to check all blocks that include x, and verify they don't collide with existing blocks
    
    # First, compute all existing block products of the original sequence
    # Then compute new block products that include x
    # A new block product is valid if it's not in the existing set
    
    # But wait, we also need to make sure the new blocks don't collide with each other
    # Since all blocks include x, and x is new, two different blocks including x 
    # could have the same product only if... actually they could.
    # E.g., values = [2, 3], insert 6 at pos 1: blocks are [2,6] and [6,3] with products 12 and 18
    # Or values = [2, 3], insert 6 at pos 0: blocks are [6], [6,2], [6,2,3] with products 6, 12, 36
    
    # Let me just compute all new block products and check against existing + each other
    
    existing = _compute_products(values)
    
    new_products = set()
    # Blocks that include x at position pos in the new sequence
    # The new sequence is: values[0..pos-1], x, values[pos..n-1]
    # A block including x is values[i..pos-1] + [x] + values[pos..j] for some i, j
    # where 0 <= i <= pos and pos <= j < n
    
    for i in range(pos + 1):  # i from 0 to pos (i is the start index, pos is where x goes)
        left_prod = 1
        for k in range(i, pos):
            left_prod *= values[k]
        for j in range(pos, n):
            right_prod = 1
            for k in range(pos, j + 1):
                right_prod *= values[k]
            block_prod = left_prod * x * right_prod
            if block_prod in existing or block_prod in new_products:
                return False
            new_products.add(block_prod)
    
    return True

import random

def _fast_local_search(values, max_time=50):
    """
    Fast local search using limited 1-out/1-in swaps and random restarts.
    This avoids the O(N^4) explosion of the previous 1-out/2-in search.
    """
    start_time = time.time()
    best_values = values[:]
    best_len = len(values)
    
    # Helper to check if a sequence is valid
    def _check_valid(seq):
        if not seq:
            return True
        if seq[0] < 2 or seq[-1] > 128:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        prods = _compute_products(seq)
        return len(prods) == len(seq) * (len(seq) + 1) // 2

    # Try a number of random restarts / local improvements
    # Strategy: 
    # 1. Try to swap one element with another to enable a longer sequence.
    # 2. If that fails, try a random greedy restart with different tie-breaking.
    
    # Set of used numbers in best_values
    used = set(best_values)
    available = [x for x in range(2, 129) if x not in used]
    
    attempts = 0
    while time.time() - start_time < max_time:
        attempts += 1
        if attempts > 500: # Hard cap on attempts to prevent infinite loop if time check is slow
            break
            
        # Randomly pick an element to remove
        if not best_values:
            break
        rem_idx = random.randint(0, len(best_values) - 1)
        rem_val = best_values[rem_idx]
        
        # Try replacing it with a random available number
        if not available:
            break
        new_val = random.choice(available)
        
        # Construct candidate: remove rem_val, insert new_val at a valid position
        # Since the sequence is strictly increasing, we need to find the correct spot for new_val
        # in the sequence without rem_val.
        seq_without = best_values[:rem_idx] + best_values[rem_idx+1:]
        
        # Find insertion position for new_val in seq_without
        # Use binary search or linear scan
        insert_pos = -1
        for k in range(len(seq_without)):
            if seq_without[k] > new_val:
                insert_pos = k
                break
        if insert_pos == -1:
            insert_pos = len(seq_without)
            
        # Check if inserting at insert_pos is valid (strictly increasing)
        # We must ensure new_val fits between neighbors
        valid_insert = True
        if insert_pos > 0 and seq_without[insert_pos-1] >= new_val:
            valid_insert = False
        if insert_pos < len(seq_without) and seq_without[insert_pos] <= new_val:
            valid_insert = False
            
        if not valid_insert:
            continue
            
        candidate = seq_without[:insert_pos] + [new_val] + seq_without[insert_pos:]
        
        if _check_valid(candidate):
            # If valid, we have the same length. 
            # To improve length, we would need to insert *another* number.
            # Let's try to insert any number from available (excluding new_val) into this candidate.
            used_in_cand = set(candidate)
            avail_in_cand = [x for x in range(2, 129) if x not in used_in_cand]
            
            # Try inserting one more number
            for x in avail_in_cand:
                # Find position for x in candidate
                pos_x = -1
                for k in range(len(candidate)):
                    if candidate[k] > x:
                        pos_x = k
                        break
                if pos_x == -1:
                    pos_x = len(candidate)
                    
                if pos_x > 0 and candidate[pos_x-1] >= x:
                    continue
                if pos_x < len(candidate) and candidate[pos_x] <= x:
                    continue
                    
                final_candidate = candidate[:pos_x] + [x] + candidate[pos_x:]
                if _check_valid(final_candidate):
                    if len(final_candidate) > best_len:
                        best_values = final_candidate
                        best_len = len(final_candidate)
                        used = set(best_values)
                        available = [x for x in range(2, 129) if x not in used]
                        break
            # Even if we didn't extend, if the swap itself was valid, 
            # it might open up future swaps, but since we are maximizing length,
            # we only update best if we got a longer sequence.
            # However, sometimes a same-length swap is necessary to enable a longer one later.
            # To keep it simple and fast, we only update if length increases.
            # This might get stuck in local optima, but it's fast.
            pass

    return best_values

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Phase 1: Greedy construction
    values, products = [], set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    greedy_len = len(values)
    
    # Phase 2: Fast local search
    if time.time() < deadline - 5:
        improved = _fast_local_search(values, max_time=deadline - time.time() - 5)
        if len(improved) > greedy_len:
            values = improved
    
    return values
# EVOLVE-BLOCK-END
