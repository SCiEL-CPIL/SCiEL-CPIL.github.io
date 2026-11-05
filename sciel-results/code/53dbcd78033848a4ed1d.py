import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Helper to check validity
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

    # 1. Greedy Initialization
    values = []
    products = set()
    for x in range(2, 129):
        # Compute tails for x appended to values
        # Tails are: x, x*last, x*last*second_last, ...
        tails = []
        prod = x
        for a in reversed(values):
            tails.append(prod)
            prod *= a
        # The tail for the full sequence is prod * first? No.
        # If values = [v0, v1, ..., vk], and we add x at end:
        # New blocks ending at x:
        # [x], [vk, x], [vk-1, vk, x], ...
        # prod starts as x.
        # tails = [x]
        # next: prod = x * vk. tails.append(prod)
        # ...
        # This logic in the seed is:
        # tails = [x]
        # prod = x
        # for a in reversed(values):
        #   prod *= a
        #   tails.append(prod)
        # This generates [x, x*v_k, x*v_k*v_{k-1}, ...]. Correct.
        
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
            
    best_seq = values
    best_len = len(best_seq)
    
    # Helper to try inserting a value into a sorted sequence
    def try_insert(seq, val):
        # Find position to maintain sorted order
        # Since seq is sorted, we can binary search or linear scan
        # Linear scan is fine for small N
        pos = -1
        for i, v in enumerate(seq):
            if v > val:
                pos = i
                break
        if pos == -1:
            pos = len(seq)
            
        # Check if val is already in seq
        if pos < len(seq) and seq[pos] == val:
            return None # Duplicate
            
        # Construct new seq
        new_seq = seq[:pos] + [val] + seq[pos:]
        
        # Check validity
        if is_valid(new_seq):
            return new_seq
        return None

    # Helper to try swapping an element
    def try_swap(seq, index, new_val):
        # Remove seq[index], insert new_val
        # Must maintain sorted order
        remaining = seq[:index] + seq[index+1:]
        return try_insert(remaining, new_val)

    # Local Search Loop
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        
        # Attempt 1: Direct insertion of any missing number
        # This is fast if we just check all missing numbers
        missing = [x for x in range(2, 129) if x not in best_seq]
        
        # Optimization: If we can insert any missing number, we improve length.
        # We should prioritize finding a valid insertion.
        inserted = False
        for m in missing:
            new_seq = try_insert(best_seq, m)
            if new_seq:
                best_seq = new_seq
                best_len += 1
                improved = True
                inserted = True
                break # Restart search with new best_seq
                
        if improved:
            continue
            
        # Attempt 2: Swap to enable insertion
        # If we can't insert directly, try swapping an element in best_seq 
        # with a missing element, then try to insert a (different) missing element.
        
        # To limit search space, we can iterate over swaps.
        # Number of swaps: len(best_seq) * len(missing)
        # For each swap, we check validity (O(N^2)) and then try all missing inserts (O(M * N^2))
        # This might be too slow if done naively.
        
        # Heuristic: Only try swaps that "open up" space? 
        # Or just iterate. N~40, M~80. 40*80 = 3200 swaps.
        # Each swap check: is_valid (40^2 = 1600 ops). 3200 * 1600 = 5M ops. Fast in Python? Maybe 1-2s.
        # Then for each valid swap, try 80 inserts. 80 * 1600 = 128k ops.
        # Total per full sweep: ~5M + 3200*128k (if all valid) -> too slow.
        
        # Refinement: 
        # 1. Generate all valid swaps (same length).
        # 2. For each valid swap, try to insert any missing number.
        # 3. If any insertion works, accept the new state (length+1).
        
        valid_swaps = []
        for i in range(len(best_seq)):
            old_val = best_seq[i]
            for m in missing:
                # Skip if m == old_val (not possible since m is missing)
                # Check if swapping best_seq[i] with m yields a valid sequence
                # We can construct the candidate sequence and check validity
                candidate = best_seq[:i] + [m] + best_seq[i+1:]
                # Check sorted order
                # Since best_seq is sorted, candidate is sorted iff:
                # (i==0 or best_seq[i-1] < m) and (i==len-1 or m < best_seq[i+1])
                if i > 0 and not (best_seq[i-1] < m):
                    continue
                if i < len(best_seq)-1 and not (m < best_seq[i+1]):
                    continue
                
                if is_valid(candidate):
                    valid_swaps.append((i, m, candidate))
        
        # Now try to extend any valid swap
        for i, m, cand in valid_swaps:
            # Try to insert any missing number into cand
            # Note: 'missing' list is based on best_seq. 
            # cand has 'm' instead of best_seq[i].
            # So the set of elements in cand is best_seq - {best_seq[i]} + {m}.
            # The missing numbers for cand are (original_missing - {m}) U {best_seq[i]}.
            # We want to add a number z such that z not in cand.
            
            # Let's just iterate through all numbers 2..128 not in cand
            cand_set = set(cand)
            for z in range(2, 129):
                if z in cand_set:
                    continue
                new_seq = try_insert(cand, z)
                if new_seq:
                    best_seq = new_seq
                    best_len += 1
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            # If no improvement found in a full sweep, we might be in a local optimum.
            # We can try randomized restarts or perturbations, but given the time limit 
            # and the fact that greedy is already decent, and swaps are expensive,
            # we might just stop if we can't improve length.
            # However, the prompt says "Maximize length". 
            # If we can't increase length, we are done?
            # The score is based on length. If we can't increase length, we can't improve score.
            break

    return best_seq

# Run initial check to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(len(res))
