import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129
    
    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using numpy convolution-like approach"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=int)
        # Compute all pair sums (unordered, with self-pairs)
        # For efficiency, use broadcasting for small k
        if k <= 50:
            A_i = A_arr[:, None]
            A_j = A_arr[None, :]
            sums = A_i + A_j
            # Count only upper triangle + diagonal (unordered pairs with self)
            # Actually, for unordered pairs with self: count (i,j) for i<=j
            # Use triu indices
            mask = np.triu(np.ones((k, k), dtype=bool), k=0)
            valid_sums = sums[mask]
            valid_sums = valid_sums[valid_sums <= max_sum]
            if len(valid_sums) > 0:
                counts = np.bincount(valid_sums, minlength=max_sum + 1)
            else:
                counts = np.zeros(max_sum + 1, dtype=int)
            return counts
        else:
            # Fallback for large sets
            counts = np.zeros(max_sum + 1, dtype=int)
            for i in range(k):
                ai = A_arr[i]
                s = ai + ai
                if s <= max_sum:
                    counts[s] += 1
                for j in range(i + 1, k):
                    aj = A_arr[j]
                    s = ai + aj
                    if s <= max_sum:
                        counts[s] += 1
            return counts
    
    def get_cost(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        if np.any(target_counts == 0):
            uncovered = np.sum(target_counts == 0)
            return 1000 + uncovered
        return int(np.max(target_counts))
    
    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, counts)
        return (int(np.max(target_counts)), 0, counts)
    
    def incremental_swap_cost(A, x_in, x_out, current_counts):
        """
        Compute new counts after removing x_in and adding x_out.
        Returns (new_counts, new_cost, new_uncovered)
        """
        A_sorted = sorted(A)
        k = len(A_sorted)
        
        # Remove x_in from consideration
        # For each other element y in A (y != x_in), the pair (x_in, y) contributes to sum x_in+y
        # Also (x_in, x_in) contributes to 2*x_in
        # After removal, these contributions are gone
        
        # Add x_out to consideration
        # For each other element y in A (y != x_in), the pair (x_out, y) contributes to sum x_out+y
        # Also (x_out, x_out) contributes to 2*x_out
        
        new_counts = current_counts.copy()
        
        # Subtract contributions involving x_in
        # Self-pair
        s = 2 * x_in
        if s <= max_sum:
            new_counts[s] -= 1
        
        # Pairs with other elements
        for y in A:
            if y == x_in:
                continue
            s = x_in + y
            if s <= max_sum:
                new_counts[s] -= 1
        
        # Add contributions involving x_out
        # Self-pair
        s = 2 * x_out
        if s <= max_sum:
            new_counts[s] += 1
        
        # Pairs with other elements (excluding x_in which is being removed)
        for y in A:
            if y == x_in:
                continue
            s = x_out + y
            if s <= max_sum:
                new_counts[s] += 1
        
        # Compute new cost
        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))
        
        return new_counts, new_cost, uncovered
    
    def incremental_add_cost(A, x_add, current_counts):
        """
        Compute new counts after adding x_add to A.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()
        
        # Self-pair
        s = 2 * x_add
        if s <= max_sum:
            new_counts[s] += 1
        
        # Pairs with existing elements
        for y in A:
            s = x_add + y
            if s <= max_sum:
                new_counts[s] += 1
        
        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))
        
        return new_counts, new_cost, uncovered
    
    def incremental_remove_cost(A, x_rem, current_counts):
        """
        Compute new counts after removing x_rem from A.
        Returns (new_counts, new_cost, new_uncovered)
        """
        new_counts = current_counts.copy()
        
        # Self-pair
        s = 2 * x_rem
        if s <= max_sum:
            new_counts[s] -= 1
        
        # Pairs with other elements
        for y in A:
            if y == x_rem:
                continue
            s = x_rem + y
            if s <= max_sum:
                new_counts[s] -= 1
        
        target_counts = new_counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            new_cost = 1000 + uncovered
        else:
            new_cost = int(np.max(target_counts))
        
        return new_counts, new_cost, uncovered
    
    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')
    
    # Try various structured initial sets
    candidates = []
    
    # Candidate 1: k=16, spread out
    for k in [16, 17, 18, 19, 20]:
        # Spread elements evenly from 0 to 128
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) < k:
            base_set = set()
            used = set()
            for i in range(k):
                x = round(i * 128 / (k - 1))
                while x in used:
                    x += 1
                    if x > 128:
                        x = 0
                base_set.add(x)
                used.add(x)
            base_set = set(x for x in base_set if 0 <= x <= 128)
        candidates.append(set(base_set))
        
        # Also try with slight perturbations
        for offset in range(-3, 4):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128 / (k - 1))) + offset
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 2:
                candidates.append(perturbed)
    
    # Candidate: Two clusters approach
    for a in range(5, 15):
        for b in range(113, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)
    
    # Candidate: Mix of small and large
    for m in range(6, 12):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid = set(range(60 - m // 2, 60 + m // 2 + 1))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)
    
    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost, uncovered, _ = get_cost_detailed(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)
    
    # Now optimize the best candidate with local search using incremental updates
    if best_A is not None:
        A = set(best_A)
        current_counts = get_counts_fast(A)
        current_cost, _, _ = get_cost_detailed(A)
        
        # Local search: try swaps and additions/removals with incremental cost
        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)
        
        for iteration in range(50000):
            if time.time() > deadline - 5.0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6 and len(outside) > 0:
                # Swap
                x_in = rng.choice(A_list)
                x_out = rng.choice(outside)
                
                # Try the swap
                new_counts, new_cost, _ = incremental_swap_cost(A, x_in, x_out, current_counts)
                
                if new_cost <= current_cost:
                    # Accept the swap
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    A_list.remove(x_in)
                    A_list.append(x_out)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                # else: reject, no changes needed
            
            elif move_type < 0.8 and len(A) > 14:
                # Removal
                x = rng.choice(A_list)
                
                new_counts, new_cost, _ = incremental_remove_cost(A, x, current_counts)
                
                if new_cost <= current_cost:
                    A.remove(x)
                    outside.append(x)
                    A_list.remove(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                # else: reject
            
            elif move_type < 1.0 and len(A) < 30:
                # Addition
                x = rng.choice(outside)
                
                new_counts, new_cost, _ = incremental_add_cost(A, x, current_counts)
                
                if new_cost <= current_cost:
                    A.add(x)
                    outside.remove(x)
                    A_list.append(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                # else: reject
            
            # Early exit if we reach a very good cost
            if best_cost <= 2:
                break
    
    # Final verification and cleanup
    if best_A is None:
        # Fallback: return a valid set
        best_A = set(range(0, 129, 8))
    
    return list(best_A)
