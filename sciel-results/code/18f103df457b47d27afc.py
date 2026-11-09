import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy or tetracode extension
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage masks (bitsets) for fast OR operations if needed, 
    # but for incremental counting, we need the list of neighbors.
    # neighbor_list[i] = list of indices covered by words[i] (including i)
    neighbor_list = []
    for i in range(N):
        w_i = words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                j = word_to_idx[new_w]
                neighbors.append(j)
        neighbor_list.append(neighbors)
        
    # We will use an array to track coverage counts.
    # coverage_counts[j] = number of codewords in current_code that cover word j.
    
    def get_initial_tetracode():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                w = tuple(h + list(tail))
                init_indices.append(word_to_idx[w])
        return init_indices

    best_code = set(get_initial_tetracode())
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # State variables
    current_code = set(best_code)
    coverage_counts = np.zeros(N, dtype=np.int32)
    
    # Initialize coverage counts
    for idx in current_code:
        for j in neighbor_list[idx]:
            coverage_counts[j] += 1
            
    current_uncovered = np.sum(coverage_counts == 0)
    
    # Helper to check if a codeword is redundant
    # A codeword c is redundant if for all j in neighbor_list[c], coverage_counts[j] > 1
    # (since c itself contributes 1 to all j in its neighborhood)
    def is_redundant(idx):
        # If removing idx makes any j uncovered, it's not redundant.
        # j is uncovered after removal if coverage_counts[j] == 1 (only idx covers it).
        for j in neighbor_list[idx]:
            if coverage_counts[j] == 1:
                return False
        return True

    # Phase 1: Deterministic Greedy Reduction
    # Try to remove redundant codewords until no more can be removed.
    # This is fast and improves the starting point for SA.
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        # Iterate in random order to avoid bias
        indices_list = list(current_code)
        random.shuffle(indices_list)
        for idx in indices_list:
            if is_redundant(idx):
                # Remove it
                for j in neighbor_list[idx]:
                    coverage_counts[j] -= 1
                current_code.remove(idx)
                current_uncovered = np.sum(coverage_counts == 0)
                changed = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = set(current_code)

    # Phase 2: Simulated Annealing
    # We use the incremental coverage counts for fast delta calculation.
    # Cost function: size + large_penalty * uncovered
    # We want to minimize size. Uncovered must be 0 for validity.
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    all_indices = list(range(N))
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        
        new_code = current_code
        new_uncovered = current_uncovered
        
        # We need to calculate the delta in cost.
        # cost = |C| + 1000 * uncovered
        # delta_cost = delta_size + 1000 * delta_uncovered
        
        delta_size = 0
        delta_uncovered = 0
        
        if move_type < 0.4:
            # Remove a random codeword
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            
            # Check impact
            # If we remove to_remove, coverage_counts[j] decreases by 1 for j in neighbor_list[to_remove]
            # New uncovered count = count of j in neighbor_list[to_remove] where coverage_counts[j] == 1
            # (Because if count was 1, it becomes 0. If count > 1, it stays > 0. If count was 0, impossible as it's in current_code)
            
            newly_uncovered = 0
            for j in neighbor_list[to_remove]:
                if coverage_counts[j] == 1:
                    newly_uncovered += 1
            
            delta_size = -1
            delta_uncovered = newly_uncovered
            
            # Acceptance check
            delta_cost = delta_size + 1000 * delta_uncovered
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Accept
                for j in neighbor_list[to_remove]:
                    coverage_counts[j] -= 1
                current_code.remove(to_remove)
                current_uncovered += newly_uncovered
                
                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)
                        
        elif move_type < 0.7:
            # Add a random codeword
            to_add = random.choice(all_indices)
            if to_add in current_code:
                continue
                
            # Check impact
            # Coverage_counts[j] increases by 1 for j in neighbor_list[to_add]
            # New covered count = count of j in neighbor_list[to_add] where coverage_counts[j] == 0
            newly_covered = 0
            for j in neighbor_list[to_add]:
                if coverage_counts[j] == 0:
                    newly_covered += 1
            
            delta_size = 1
            delta_uncovered = -newly_covered
            
            delta_cost = delta_size + 1000 * delta_uncovered
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Accept
                for j in neighbor_list[to_add]:
                    coverage_counts[j] += 1
                current_code.add(to_add)
                current_uncovered -= newly_covered
                
                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)
                        
        else:
            # Swap: remove one, add another
            if not current_code:
                continue
            to_remove = random.choice(list(current_code))
            to_add = random.choice(all_indices)
            if to_add == to_remove or to_add in current_code:
                continue
            
            # Calculate combined impact
            # 1. Remove to_remove: counts decrease
            # 2. Add to_add: counts increase
            
            # We need to be careful with overlapping neighborhoods.
            # Let's simulate the counts change on a temporary copy or just calculate deltas carefully.
            # Given N=729, copying the array is fast.
            temp_counts = coverage_counts.copy()
            
            for j in neighbor_list[to_remove]:
                temp_counts[j] -= 1
            
            for j in neighbor_list[to_add]:
                temp_counts[j] += 1
            
            new_uncovered_val = np.sum(temp_counts == 0)
            
            delta_size = 0 # Remove 1, Add 1
            delta_uncovered = new_uncovered_val - current_uncovered
            
            delta_cost = delta_size + 1000 * delta_uncovered
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Accept
                coverage_counts = temp_counts
                current_code.remove(to_remove)
                current_code.add(to_add)
                current_uncovered = new_uncovered_val
                
                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)
                        
        T *= cooling
        
        # Periodic deterministic polish
        # Every 1000 steps, try to remove redundant words again
        # This helps escape local minima where a random removal might fail but a specific one works
        if random.random() < 0.01:
             indices_list = list(current_code)
             random.shuffle(indices_list)
             for idx in indices_list:
                 if is_redundant(idx):
                     for j in neighbor_list[idx]:
                         coverage_counts[j] -= 1
                     current_code.remove(idx)
                     current_uncovered = np.sum(coverage_counts == 0)
                     if current_uncovered == 0:
                         if len(current_code) < best_size:
                             best_size = len(current_code)
                             best_code = set(current_code)

    # Final check: ensure best_code is valid
    # We trust the tracking, but let's do a final verification of best_code just in case.
    # Reconstruct coverage for best_code
    final_cov = np.zeros(N, dtype=np.int32)
    for idx in best_code:
        for j in neighbor_list[idx]:
            final_cov[j] += 1
    if np.any(final_cov == 0):
        # Fallback to the initial tetracode if something went wrong (should not happen)
        best_code = set(get_initial_tetracode())

    return [list(words[i]) for i in best_code]
