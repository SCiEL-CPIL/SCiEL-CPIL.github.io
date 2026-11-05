import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = np.arange(min_sum, max_sum + 1)
    num_targets = len(target_sums)  # 129

    def compute_counts_from_binary(binary_vec):
        """
        Compute pair-sum counts using FFT convolution.
        binary_vec: array of length 129 with 1 where element is in A.
        Returns: counts array of length 257 where counts[s] = number of unordered pairs (i,j) with i<=j, i+j=s
        """
        # Convolution: (binary * binary)[s] = sum_i binary[i]*binary[s-i] = number of ordered pairs (i, s-i) both in A
        # For unordered with self: count = (ordered_count + self_pairs) / 2
        # where self_pairs[s] = binary[s//2] if s is even else 0
        conv = np.fft.irfft(np.fft.rfft(binary_vec) ** 2, n=256)
        conv = np.round(conv).astype(np.int64)
        conv = conv[:257]
        
        # Self pairs: for even s, if s//2 is in A, add 1
        self_pairs = np.zeros(257, dtype=np.int64)
        for s in range(0, 257, 2):
            idx = s // 2
            if idx <= 128:
                self_pairs[s] = binary_vec[idx]
        
        # Unordered count = (ordered + self) / 2
        counts = (conv + self_pairs) // 2
        return counts

    def get_binary(A):
        """Convert set A to binary indicator vector of length 129."""
        bv = np.zeros(129, dtype=np.int64)
        for x in A:
            bv[x] = 1
        return bv

    def get_cost_from_counts(counts):
        """Compute cost from counts array."""
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return 1000 + uncovered
        return int(np.max(target_counts))

    def get_cost(A):
        bv = get_binary(A)
        counts = compute_counts_from_binary(bv)
        return get_cost_from_counts(counts)

    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')
    
    candidates = []
    
    # Candidate 1: k elements spread evenly
    for k in range(10, 25):
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) >= k - 2:
            candidates.append(set(base_set))
        
        # Perturbations
        for offset in range(-4, 5):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128 / max(k - 1, 1))) + offset
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 2:
                candidates.append(perturbed)
    
    # Candidate: Two clusters
    for a in range(3, 20):
        for b in range(109, 129):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)
    
    # Candidate: Mix of small and large
    for m in range(4, 15):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid_start = 60 - m // 2
        mid_end = 60 + m // 2
        mid = set(range(max(0, mid_start), min(129, mid_end + 1)))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)
    
    # Random candidates
    rng0 = random.Random(42)
    for _ in range(200):
        k = rng0.randint(12, 28)
        cand = set(rng0.sample(range(129), k))
        candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost = get_cost(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)
    
    # Local search with FFT-based count updates
    if best_A is not None:
        A = set(best_A)
        bv = get_binary(A)
        current_counts = compute_counts_from_binary(bv)
        current_cost = get_cost_from_counts(current_counts)
        
        rng = random.Random(12345)
        outside = list(set(range(129)) - A)
        A_list = list(A)
        
        # Precompute for faster incremental updates
        # When we swap x_in -> x_out, the change in counts is:
        # Remove contributions of x_in, add contributions of x_out
        # For a set A, the contribution of element x to count[s] is:
        #   1 if x + x = s (self-pair, counted once)
        #   1 if there exists y in A, y != x, x + y = s
        # Actually for unordered: element x contributes to sum s if:
        #   s = 2x (self) -> contributes 1
        #   s = x + y for some y in A, y != x -> contributes 1 for each such y
        # But we need to be careful: the total count[s] = number of unordered pairs {i,j} with i+j=s
        # When we remove x, we lose pairs {x,x} and {x,y} for all y in A\{x}
        # When we add x, we gain pairs {x,x} and {x,y} for all y in A\{x} (the new A)
        
        # Faster approach: just recompute with FFT (it's very fast for n=129)
        # But let's try incremental for the swap case
        
        def try_swap(A, x_in, x_out, current_counts):
            """Try swapping x_in for x_out, return (new_counts, new_cost) or None if invalid."""
            # New A = (A - {x_in}) + {x_out}
            # Counts change:
            # Remove x_in: lose pairs {x_in, x_in} and {x_in, y} for y in A, y != x_in
            # Add x_out: gain pairs {x_out, x_out} and {x_out, y} for y in A\{x_in}, y != x_out
            
            new_counts = current_counts.copy()
            
            # Remove x_in contributions
            s_self = 2 * x_in
            if s_self <= max_sum:
                new_counts[s_self] -= 1
            
            for y in A:
                if y == x_in:
                    continue
                s = x_in + y
                if s <= max_sum:
                    new_counts[s] -= 1
            
            # Add x_out contributions (with A' = A - {x_in} + {x_out})
            s_self = 2 * x_out
            if s_self <= max_sum:
                new_counts[s_self] += 1
            
            for y in A:
                if y == x_in:
                    continue
                if y == x_out:
                    continue
                s = x_out + y
                if s <= max_sum:
                    new_counts[s] += 1
            
            # Also need to handle: if x_out was not in A, the pair {x_out, x_in} 
            # Wait, x_in is being removed, so we don't add {x_out, x_in}
            # And x_out wasn't in A before, so no existing pairs with x_out to worry about
            
            target_counts = new_counts[min_sum:max_sum + 1]
            uncovered = np.sum(target_counts == 0)
            if uncovered > 0:
                new_cost = 1000 + uncovered
            else:
                new_cost = int(np.max(target_counts))
            
            return new_counts, new_cost
        
        def try_add(A, x_add, current_counts):
            """Try adding x_add to A."""
            new_counts = current_counts.copy()
            
            s_self = 2 * x_add
            if s_self <= max_sum:
                new_counts[s_self] += 1
            
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
            
            return new_counts, new_cost
        
        def try_remove(A, x_rem, current_counts):
            """Try removing x_rem from A."""
            new_counts = current_counts.copy()
            
            s_self = 2 * x_rem
            if s_self <= max_sum:
                new_counts[s_self] -= 1
            
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
            
            return new_counts, new_cost
        
        no_improve_count = 0
        max_no_improve = 3000
        
        for iteration in range(100000):
            if time.time() > deadline - 5.0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.65 and len(outside) > 0:
                # Swap
                x_in = rng.choice(A_list)
                x_out = rng.choice(outside)
                
                new_counts, new_cost = try_swap(A, x_in, x_out, current_counts)
                
                if new_cost <= current_cost:
                    A.remove(x_in)
                    A.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    A_list.remove(x_in)
                    A_list.append(x_out)
                    current_counts = new_counts
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.85 and len(A) > 10:
                # Removal
                x = rng.choice(A_list)
                
                new_counts, new_cost = try_remove(A, x, current_counts)
                
                if new_cost <= current_cost:
                    A.remove(x)
                    outside.append(x)
                    A_list.remove(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    no_improve_count += 1
                    
            elif move_type < 1.0 and len(A) < 35:
                # Addition
                x = rng.choice(outside)
                
                new_counts, new_cost = try_add(A, x, current_counts)
                
                if new_cost <= current_cost:
                    A.add(x)
                    outside.remove(x)
                    A_list.append(x)
                    current_counts = new_counts
                    current_cost = new_cost
                    no_improve_count = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    no_improve_count += 1
            
            # Restart if stuck
            if no_improve_count > max_no_improve:
                # Perturb: do a few random swaps to escape local optimum
                for _ in range(5):
                    if len(outside) > 0:
                        x_in = rng.choice(A_list)
                        x_out = rng.choice(outside)
                        A.remove(x_in)
                        A.add(x_out)
                        outside.remove(x_out)
                        outside.append(x_in)
                        A_list.remove(x_in)
                        A_list.append(x_out)
                bv = get_binary(A)
                current_counts = compute_counts_from_binary(bv)
                current_cost = get_cost_from_counts(current_counts)
                no_improve_count = 0
            
            # Early exit if optimal
            if best_cost <= 2:
                break
    
    # Final verification
    if best_A is None:
        best_A = set(range(0, 129, 8))
    
    # Verify the result is valid
    final_counts = compute_counts_from_binary(get_binary(best_A))
    final_target = final_counts[min_sum:max_sum + 1]
    if np.any(final_target == 0):
        # Fallback to a known valid set
        best_A = set(range(0, 129, 4))
        final_counts = compute_counts_from_binary(get_binary(best_A))
        final_target = final_counts[min_sum:max_sum + 1]
        if np.any(final_target == 0):
            best_A = set(range(0, 129, 2))
    
    return list(best_A)
