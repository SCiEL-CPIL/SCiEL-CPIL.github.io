import random
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n
    num_targets = target_hi - target_lo + 1
    
    # We will maintain counts for all sums 0..256
    # But only target_lo..target_hi matter for the score
    
    def compute_cost(counts):
        """
        Compute scalar cost.
        Lower is better.
        """
        target_counts = counts[target_lo:target_hi+1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            # High penalty for uncovered
            return 1000.0 * uncovered + np.max(target_counts) + 0.1 * np.sum(target_counts**2)
        
        peak = np.max(target_counts)
        # We want to minimize peak. 
        # To encourage balancing, we also penalize variance or sum of squares.
        # If peak is minimized, sum of squares is a good secondary metric.
        return peak + 0.01 * np.sum(target_counts**2)

    def init_counts_numpy(A):
        """
        Initialize counts array using numpy convolution or loops.
        Given A is a set of ints, we can create a binary vector and use convolution?
        Or just loop. Loop is O(|A|^2). |A| ~ 50-100. 10000 ops is fine.
        But we can be faster:
        For each a in A, add 1 to counts[a+a], and for b>a, add 1 to counts[a+b].
        """
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        A_list = sorted(A)
        for i in range(len(A_list)):
            a = A_list[i]
            # Self pair
            counts[2*a] += 1
            # Pairs with j > i
            if i < len(A_list) - 1:
                # Vectorized update for j > i
                # b_vals = np.array(A_list[i+1:])
                # sums = a + b_vals
                # np.add.at(counts, sums, 1)
                # Since A_list is unique, sums are unique? No, a+b1 == a+b2 implies b1==b2.
                # So sums are unique.
                b_vals = np.array(A_list[i+1:], dtype=np.int32)
                sums = a + b_vals
                counts[sums] += 1
        return counts

    def add_element_numpy(A, counts, y):
        """
        Add y to A and update counts.
        """
        A.add(y)
        # Pairs (y, z) for all z in A.
        # This includes (y, y).
        # If z < y, pair (z, y) was not counted before? 
        # Wait, the previous logic in the seed program:
        # add_element(A, counts, y):
        #   A.add(y)
        #   for z in A: counts[y+z] += 1
        # This adds (y,y) once, and (y,z) for all z != y.
        # Since the set is unordered, (y,z) is the same as (z,y).
        # If we iterate over the *new* A, we add (y,z) for all z in A_new.
        # This covers (y,y) and all (y,z) where z != y.
        # This is correct for the total count of unordered pairs.
        
        A_list = np.array(list(A), dtype=np.int32)
        sums = y + A_list
        # Ensure sums are within bounds
        valid = sums <= max_sum
        if np.any(valid):
            counts[sums[valid]] += 1

    def remove_element_numpy(A, counts, x):
        """
        Remove x from A and update counts.
        """
        A.remove(x)
        # Pairs involving x were {x, z} for z in A_old.
        # A_old = A_new U {x}.
        # So we need to subtract 1 for {x, z} for all z in A_new, and {x, x}.
        
        A_list = np.array(list(A), dtype=np.int32)
        sums = x + A_list
        valid = sums <= max_sum
        if np.any(valid):
            counts[sums[valid]] -= 1
        
        s_xx = 2 * x
        if s_xx <= max_sum:
            counts[s_xx] -= 1

    def swap_element_numpy(A, counts, x, y):
        """
        Swap x with y. x in A, y not in A.
        """
        # Remove x
        A.remove(x)
        A_list_after_rem = np.array(list(A), dtype=np.int32)
        sums_x = x + A_list_after_rem
        valid_x = sums_x <= max_sum
        if np.any(valid_x):
            counts[sums_x[valid_x]] -= 1
        s_xx = 2 * x
        if s_xx <= max_sum:
            counts[s_xx] -= 1
            
        # Add y
        A.add(y)
        A_list_after_add = np.array(list(A), dtype=np.int32)
        sums_y = y + A_list_after_add
        valid_y = sums_y <= max_sum
        if np.any(valid_y):
            counts[sums_y[valid_y]] += 1

    # --- Greedy Initialization ---
    # Start with a set that covers the range well.
    # We want to cover [128, 256].
    # A simple heuristic: Start with A = {0, 128} or similar?
    # Better: Random set, then repair.
    
    # Let's try a few different initializations
    best_A = None
    best_cost = float('inf')
    
    # 1. Random Sets
    for _ in range(10):
        k = random.randint(40, 70)
        A = set(random.sample(range(n+1), k))
        counts = init_counts_numpy(A)
        cost = compute_cost(counts)
        if cost < best_cost:
            best_cost = cost
            best_A = set(A)
            
    # 2. Structured Sets
    for step in [2, 3]:
        for start in range(step):
            A = set(range(start, n+1, step))
            counts = init_counts_numpy(A)
            cost = compute_cost(counts)
            if cost < best_cost:
                best_cost = cost
                best_A = set(A)
                
    # 3. Greedy Coverage
    # Start with empty? No, hard to cover.
    # Start with a medium set and greedily add elements that reduce cost most.
    A_greedy = set(random.sample(range(n+1), 30))
    counts_greedy = init_counts_numpy(A_greedy)
    cost_greedy = compute_cost(counts_greedy)
    
    # Try to improve greedy init by adding elements
    for _ in range(50):
        if time.time() > deadline - 5.0: break
        best_add = None
        best_add_cost = cost_greedy
        # Sample candidates
        candidates = [i for i in range(n+1) if i not in A_greedy]
        random.shuffle(candidates)
        for y in candidates[:20]:
            # Simulate add
            A_greedy.add(y)
            A_list = np.array(list(A_greedy), dtype=np.int32)
            sums = y + A_list
            valid = sums <= max_sum
            if np.any(valid):
                counts_greedy[sums[valid]] += 1
            new_cost = compute_cost(counts_greedy)
            # Revert
            A_greedy.remove(y)
            if np.any(valid):
                counts_greedy[sums[valid]] -= 1
                
            if new_cost < best_add_cost:
                best_add_cost = new_cost
                best_add = y
                # Store the counts state? No, just re-apply later.
        
        if best_add is not None:
            add_element_numpy(A_greedy, counts_greedy, best_add)
            cost_greedy = best_add_cost
        else:
            break
            
    if cost_greedy < best_cost:
        best_cost = cost_greedy
        best_A = set(A_greedy)

    # --- Simulated Annealing ---
    A = set(best_A)
    counts = init_counts_numpy(A)
    curr_cost = compute_cost(counts)
    
    T = 50.0
    T_min = 0.1
    alpha = 0.999
    steps = 0
    
    while time.time() < deadline - 2.0:
        steps += 1
        # Move generation
        # Probabilities:
        # If uncovered, try to add or swap.
        # If covered, try to swap or remove to reduce peak/variance.
        
        target_counts = counts[target_lo:target_hi+1]
        uncovered = np.sum(target_counts == 0)
        
        if uncovered > 0:
            p_add = 0.5
            p_swap = 0.4
            p_remove = 0.1
        else:
            p_add = 0.1
            p_swap = 0.5
            p_remove = 0.4
            
        r = random.random()
        
        if r < p_add:
            if len(A) < n + 1:
                y = random.randint(0, n)
                if y not in A:
                    # Apply
                    add_element_numpy(A, counts, y)
                    new_cost = compute_cost(counts)
                    delta = new_cost - curr_cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        curr_cost = new_cost
                    else:
                        remove_element_numpy(A, counts, y)
                        
        elif r < p_add + p_swap:
            if len(A) > 0:
                x = random.choice(list(A))
                y = random.randint(0, n)
                if y not in A:
                    # Apply
                    swap_element_numpy(A, counts, x, y)
                    new_cost = compute_cost(counts)
                    delta = new_cost - curr_cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        curr_cost = new_cost
                    else:
                        swap_element_numpy(A, counts, y, x)
                        
        else:
            if len(A) > 1:
                x = random.choice(list(A))
                # Apply
                remove_element_numpy(A, counts, x)
                new_cost = compute_cost(counts)
                delta = new_cost - curr_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    curr_cost = new_cost
                else:
                    add_element_numpy(A, counts, x)
                    
        T *= alpha
        if T < T_min:
            T = T_min

    return list(A)
