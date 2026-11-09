import time
import random
import itertools
import sys

# Precompute all triples and their masks
ALL_TRIPLES = []
TRIPLE_MASKS = []
PAIR_TO_TRIPLES = {}

for i in range(16):
    for j in range(i+1, 16):
        for k in range(j+1, 16):
            t = (i, j, k)
            idx = len(ALL_TRIPLES)
            ALL_TRIPLES.append(t)
            mask = (1 << i) | (1 << j) | (1 << k)
            TRIPLE_MASKS.append(mask)
            # Map pairs to triple indices
            for p in [(i, j), (i, k), (j, k)]:
                if p not in PAIR_TO_TRIPLES:
                    PAIR_TO_TRIPLES[p] = []
                PAIR_TO_TRIPLES[p].append(idx)

def calculate_alpha(triples_idx):
    """
    Calculate the independence number of the system defined by triples_idx.
    triples_idx: list of indices into ALL_TRIPLES.
    """
    if not triples_idx:
        return 16
    
    masks = [TRIPLE_MASKS[i] for i in triples_idx]
    
    # Check subsets from largest size down to 1
    # We can skip sizes > 16 obviously
    # To speed up, we can stop if we find an independent set
    # But we want the MAX size.
    
    # Optimization: If the number of triples is small, alpha is large.
    # If the number of triples is large, alpha is small.
    
    # Iterate k from 16 down to 1
    for k in range(16, 0, -1):
        # Generate all subsets of size k
        # Using itertools.combinations is slow.
        # Bit trick:
        # We can iterate all masks and check popcount?
        # 2^16 = 65536 masks.
        # For each mask, if popcount(mask) == k, check if independent.
        # This is 65536 checks per k. Total 65536 * 16 = 1M checks.
        # Too slow if done for every k.
        
        # Better: Iterate all masks once, group by popcount?
        # Or just iterate all masks, and keep track of max independent size.
        
        max_ind_size = 0
        for mask in range(1, 1 << 16):
            # Quick check: if popcount(mask) <= max_ind_size, skip?
            # No, we want to find if there is ANY independent set of size > max_ind_size.
            # So we can just check all masks, and if independent, update max.
            # But checking all 65535 masks for each evaluation is 65535 * M ops.
            
            # Let's stick to the k-loop but optimize the subset generation.
            pass
            
    # Alternative: Use the property that if S is independent, all subsets of S are independent.
    # We want the largest S.
    
    # Let's use a simple loop over all masks, but break early if possible?
    # No, we need the max.
    
    # Let's try the ILP approach if available, else fallback.
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        # Variables y_v for v in 0..15
        # Maximize sum(y_v)
        # Minimize -sum(y_v)
        c = [-1] * 16
        
        # Constraints: For each triple, sum(y_a + y_b + y_c) <= 2
        # A matrix of shape (num_triples, 16)
        A = []
        for idx in triples_idx:
            row = [0] * 16
            i, j, k = ALL_TRIPLES[idx]
            row[i] = 1
            row[j] = 1
            row[k] = 1
            A.append(row)
            
        if not A:
            return 16
            
        A = __import__('numpy').array(A)
        b = [2] * len(A)
        
        constraints = LinearConstraint(A, -__import__('numpy').inf, b)
        integrality = [1] * 16
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            # res.x contains the solution
            # sum of y_v
            return int(round(sum(res.x)))
        else:
            # Fallback
            pass
    except Exception:
        pass
        
    # Fallback: Bitmask check
    # To make it faster, we only check masks with popcount > current_best_alpha?
    # But we don't have a global current_best_alpha easily accessible in this function.
    # We can pass it or just calculate exactly.
    
    # Optimized bitmask check:
    # Precompute popcount for all masks?
    # Or just iterate.
    
    # Since n=16 is small, let's just do the k-loop with combinations if M is small?
    # No, 65536 is small.
    
    max_alpha = 0
    # Iterate all masks
    for mask in range(1, 1 << 16):
        # Check if independent
        is_ind = True
        for m in masks:
            if (mask & m) == m:
                is_ind = False
                break
        if is_ind:
            # Count bits
            # bin(mask).count("1") is slow.
            # Use lookup or int.bit_count() (Python 3.10+)
            try:
                cnt = mask.bit_count()
            except AttributeError:
                cnt = bin(mask).count("1")
            if cnt > max_alpha:
                max_alpha = cnt
    return max_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_alpha = 17
    best_system = []
    
    # Helper to check if a triple can be added
    def is_valid_triple(triple_idx, current_triples, pair_to_triple):
        t = ALL_TRIPLES[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p in pair_to_triple:
                return False
        return True

    def add_triple(triple_idx, current_triples, pair_to_triple):
        t = ALL_TRIPLES[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            pair_to_triple[p] = triple_idx
        current_triples.append(triple_idx)

    def remove_triple(triple_idx, current_triples, pair_to_triple):
        t = ALL_TRIPLES[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            del pair_to_triple[p]
        current_triples.remove(triple_idx)

    # Main search loop
    while time.time() < deadline:
        # 1. Generate random maximal packing
        current_triples = []
        pair_to_triple = {}
        
        # Shuffle all triples
        shuffled_indices = list(range(len(ALL_TRIPLES)))
        random.shuffle(shuffled_indices)
        
        for idx in shuffled_indices:
            if is_valid_triple(idx, current_triples, pair_to_triple):
                add_triple(idx, current_triples, pair_to_triple)
                
        # 2. Simulated Annealing
        current_alpha = calculate_alpha(current_triples)
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = list(current_triples)
            
        # SA Parameters
        T = 5.0
        cooling = 0.999
        steps_per_sec = 1000  # Adjust based on speed
        
        sa_start = time.time()
        step_count = 0
        
        while time.time() < sa_start + 2.0:  # 2s per restart
            # Generate move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random valid triple
                # Find valid triples
                valid_indices = []
                for idx in range(len(ALL_TRIPLES)):
                    if is_valid_triple(idx, current_triples, pair_to_triple):
                        valid_indices.append(idx)
                
                if valid_indices:
                    new_idx = random.choice(valid_indices)
                    add_triple(new_idx, current_triples, pair_to_triple)
                    
                    new_alpha = calculate_alpha(current_triples)
                    delta = new_alpha - current_alpha
                    
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = list(current_triples)
                    else:
                        remove_triple(new_idx, current_triples, pair_to_triple)
                        
            elif move_type < 0.7:
                # Remove a random triple
                if current_triples:
                    out_idx = random.choice(current_triples)
                    remove_triple(out_idx, current_triples, pair_to_triple)
                    
                    new_alpha = calculate_alpha(current_triples)
                    delta = new_alpha - current_alpha
                    
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = list(current_triples)
                    else:
                        add_triple(out_idx, current_triples, pair_to_triple)
                        
            else:
                # Swap
                if current_triples:
                    out_idx = random.choice(current_triples)
                    remove_triple(out_idx, current_triples, pair_to_triple)
                    
                    # Find a valid triple to add
                    valid_indices = []
                    for idx in range(len(ALL_TRIPLES)):
                        if idx != out_idx and is_valid_triple(idx, current_triples, pair_to_triple):
                            valid_indices.append(idx)
                            
                    if valid_indices:
                        in_idx = random.choice(valid_indices)
                        add_triple(in_idx, current_triples, pair_to_triple)
                        
                        new_alpha = calculate_alpha(current_triples)
                        delta = new_alpha - current_alpha
                        
                        if delta < 0 or random.random() < pow(2.718, -delta / T):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = list(current_triples)
                        else:
                            # Undo
                            remove_triple(in_idx, current_triples, pair_to_triple)
                            add_triple(out_idx, current_triples, pair_to_triple)
                    else:
                        # No valid triple, undo removal
                        add_triple(out_idx, current_triples, pair_to_triple)
                        
            T *= cooling
            step_count += 1
            
            # Time check inside loop
            if time.time() > deadline:
                break

    # Convert best_system indices to triples
    return [list(ALL_TRIPLES[idx]) for idx in best_system]

if __name__ == "__main__":
    # Run once to test
    res = construct()
    print(res)
