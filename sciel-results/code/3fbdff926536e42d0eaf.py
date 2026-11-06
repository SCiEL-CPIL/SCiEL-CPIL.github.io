# EVOLVE-BLOCK-START
import heapq
import time

def construct():
    """
    Returns a strictly increasing sequence of integers in [2, 128] 
    maximizing length such that all contiguous block products are distinct.
    Uses Best-First Search (A* variant) with Branch and Bound.
    """
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    factor_vecs = {}
    for n in range(2, 129):
        n_tmp = n
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n_tmp % p == 0:
                vec[i] += 1
                n_tmp //= p
            if n_tmp == 1:
                break
        factor_vecs[n] = tuple(vec)
    
    zero_vec = tuple([0] * num_primes)
    
    # State structure for the heap:
    # (-potential, -length, last_val, prefix_vecs, product_set, sequence)
    # We use negative values for max-heap behavior using min-heap.
    
    def get_potential(seq_len, last_val):
        # Upper bound: current length + max possible additional numbers
        # The max additional numbers is 128 - last_val
        # But we can refine this.
        return seq_len + (128 - last_val)
    
    # Initial state: empty sequence
    # last_val = 1 (so next can be 2)
    # prefix_vecs = [zero_vec]
    # product_set = set()
    # sequence = []
    
    initial_state = (
        -get_potential(0, 1), 
        -0, 
        1, 
        [zero_vec], 
        set(), 
        []
    )
    
    heap = [initial_state]
    best_seq = []
    best_len = 0
    start_time = time.time()
    timeout = 50 # Leave 10s buffer
    
    while heap:
        # Check time
        if time.time() - start_time > timeout:
            break
        
        neg_potential, neg_len, last_val, prefix_vecs, product_set, seq = heapq.heappop(heap)
        current_len = -neg_len
        current_potential = -neg_potential
        
        # If this state's potential is not better than the best found, and we've already found a solution,
        # we might prune? 
        # Actually, if current_potential <= best_len, we can skip expanding this state?
        # Yes, because even if we add all remaining, we won't beat best_len.
        if current_potential <= best_len:
            continue
            
        # Update best if this state is longer
        if current_len > best_len:
            best_len = current_len
            best_seq = seq[:]
            
        # Expand: try adding next number x > last_val
        # To maximize length, we should try smaller x first? 
        # Smaller x leaves more room for larger x later? 
        # Actually, the "room" is just the count of numbers > last_val.
        # Picking a smaller x increases the count of remaining numbers.
        # So we should prioritize smaller x to keep the potential high.
        
        for x in range(last_val + 1, 129):
            # Check time again in inner loop? Maybe too slow.
            # Just check every 100 iterations?
            
            # Calculate new prefix
            last_prefix = prefix_vecs[-1]
            x_vec = factor_vecs[x]
            new_prefix = tuple(last_prefix[i] + x_vec[i] for i in range(num_primes))
            
            # Check for collisions with new blocks ending at x
            valid = True
            new_products = []
            for i in range(len(seq) + 1):
                old_prefix = prefix_vecs[i]
                block_vec = tuple(new_prefix[j] - old_prefix[j] for j in range(num_primes))
                if block_vec in product_set:
                    valid = False
                    break
                new_products.append(block_vec)
            
            if not valid:
                continue
            
            # If valid, create new state
            new_len = current_len + 1
            new_last_val = x
            new_potential = get_potential(new_len, new_last_val)
            
            # Pruning: If new_potential <= best_len, don't push
            if new_potential <= best_len:
                continue
                
            new_prefix_vecs = prefix_vecs + [new_prefix]
            new_product_set = product_set.copy()
            new_product_set.update(new_products)
            new_seq = seq + [x]
            
            # Push to heap
            # We use -new_potential to prioritize higher potential
            heapq.heappush(heap, (
                -new_potential, 
                -new_len, 
                new_last_val, 
                new_prefix_vecs, 
                new_product_set, 
                new_seq
            ))
            
    return best_seq

# EVOLVE-BLOCK-END
