import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subsets)
    
    # For each t-subset, store the list of k-subsets that contain it
    covering_blocks = [[] for _ in range(num_t)]
    for b_idx, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            covering_blocks[idx].append(b_idx)
    
    # Build incidence: for each k-subset, the set of t-subsets it covers
    # Use a numpy array of shape (num_k, num_t) with 0/1 - but that's 50388 * 3876 ≈ 195M entries, too big
    # Instead, use a list of frozensets or bitmasks
    # num_t = 3876, so bitmask is 3876 bits ≈ 485 bytes per block. 50388 * 485 ≈ 24MB. OK.
    # Actually, let's use Python ints as bitmasks.
    block_mask = [0] * num_k
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_mask[b_idx] = mask
    
    # For fast gain computation, we can use the bitmask approach
    # uncovered_mask: int with bits set for uncovered t-subsets
    # gain of block b = (block_mask[b] & uncovered_mask).bit_count()
    
    def greedy_construct(noise=0.0, seed=None):
        rng = random.Random(seed)
        active_set = set()
        counts = [0] * num_t
        uncovered_mask = (1 << num_t) - 1
        
        while uncovered_mask:
            best_block = -1
            second_best_block = -1
            max_gain = 0
            second_max_gain = 0
            
            for b_idx in range(num_k):
                gain = (block_mask[b_idx] & uncovered_mask).bit_count()
                if gain > max_gain:
                    second_max_gain = max_gain
                    second_best_block = best_block
                    max_gain = gain
                    best_block = b_idx
                elif gain > second_max_gain:
                    second_max_gain = gain
                    second_best_block = b_idx
            
            if best_block == -1:
                break
                
            chosen_block = best_block
            if noise > 0 and rng.random() < noise and second_best_block != -1:
                chosen_block = second_best_block
            
            active_set.add(chosen_block)
            bm = block_mask[chosen_block]
            for i in range(num_t):
                if (bm >> i) & 1:
                    counts[i] += 1
            
            uncovered_mask &= ~bm
            
        return active_set, counts
    
    def get_unique_mask(counts):
        um = 0
        for i in range(num_t):
            if counts[i] == 1:
                um |= (1 << i)
        return um
    
    def minimize(active_set, counts, unique_mask=None):
        if unique_mask is None:
            unique_mask = get_unique_mask(counts)
        
        changed = True
        while changed:
            changed = False
            for b_idx in list(active_set):
                if b_idx not in active_set:
                    continue
                if block_mask[b_idx] & unique_mask:
                    continue
                active_set.remove(b_idx)
                bm = block_mask[b_idx]
                for i in range(num_t):
                    if (bm >> i) & 1:
                        old_count = counts[i]
                        counts[i] = old_count - 1
                        if old_count == 2:
                            unique_mask |= (1 << i)
                changed = True
        return active_set, counts, unique_mask
    
    def local_search(active_set, counts, unique_mask, rng, time_budget=30.0):
        improved = True
        ls_start = time.time()
        while improved and (time.time() - ls_start) < time_budget:
            improved = False
            items = list(active_set)
            rng.shuffle(items)
            
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                if (time.time() - ls_start) > time_budget:
                    break
                
                # Check if removable
                if not (block_mask[b_idx] & unique_mask):
                    active_set.remove(b_idx)
                    bm = block_mask[b_idx]
                    for i in range(num_t):
                        if (bm >> i) & 1:
                            old_count = counts[i]
                            counts[i] = old_count - 1
                            if old_count == 2:
                                unique_mask |= (1 << i)
                    improved = True
                    active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
                    continue
                
                # Find swap candidates
                u_b = block_mask[b_idx] & unique_mask
                if u_b == 0:
                    continue
                    
                # Get the unique subsets covered by b_idx
                bits = []
                temp = u_b
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                
                if not bits:
                    continue
                
                # Sort by rarity
                bits.sort(key=lambda i: len(covering_blocks[i]))
                rarest_idx = bits[0]
                
                found_swap = False
                for cand_idx in covering_blocks[rarest_idx]:
                    if cand_idx == b_idx or cand_idx in active_set:
                        continue
                    if (block_mask[cand_idx] & u_b) == u_b:
                        active_set.remove(b_idx)
                        active_set.add(cand_idx)
                        bm_out = block_mask[b_idx]
                        bm_in = block_mask[cand_idx]
                        for i in range(num_t):
                            if (bm_out >> i) & 1:
                                old_count = counts[i]
                                counts[i] = old_count - 1
                                if old_count == 2:
                                    unique_mask |= (1 << i)
                                elif old_count == 1:
                                    unique_mask &= ~(1 << i)
                        for i in range(num_t):
                            if (bm_in >> i) & 1:
                                old_count = counts[i]
                                counts[i] = old_count + 1
                                if old_count == 0:
                                    unique_mask |= (1 << i)
                                elif old_count == 1:
                                    unique_mask &= ~(1 << i)
                        active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
                        improved = True
                        found_swap = True
                        break
                
                if not found_swap:
                    # Try 2-for-1 swap: remove b_idx, add two blocks
                    # Find the most constrained unique subset
                    # For each unique subset i in u_b, find candidates covering it
                    # Then try pairs of candidates
                    if len(bits) >= 2:
                        # Pick the two rarest
                        cand_sets = []
                        for bi in bits[:3]:  # top 3 rarest
                            cands = [c for c in covering_blocks[bi] if c != b_idx and c not in active_set]
                            if cands:
                                cand_sets.append(cands)
                        
                        if len(cand_sets) >= 2:
                            # Try pairs from the two rarest
                            best_pair_gain = -1
                            best_pair = None
                            for c1 in cand_sets[0][:20]:  # limit
                                for c2 in cand_sets[1][:20]:
                                    if c1 == c2:
                                        continue
                                    # Check if {c1, c2} covers all of u_b
                                    combined = block_mask[c1] | block_mask[c2]
                                    if (combined & u_b) == u_b:
                                        # Compute how many extra unique subsets this removes
                                        # (i.e., how many currently-unique subsets become non-unique)
                                        extra = (combined & unique_mask).bit_count() - u_b.bit_count()
                                        if extra > best_pair_gain:
                                            best_pair_gain = extra
                                            best_pair = (c1, c2)
                            
                            if best_pair:
                                c1, c2 = best_pair
                                active_set.remove(b_idx)
                                active_set.add(c1)
                                active_set.add(c2)
                                bm_out = block_mask[b_idx]
                                bm_in1 = block_mask[c1]
                                bm_in2 = block_mask[c2]
                                for i in range(num_t):
                                    if (bm_out >> i) & 1:
                                        old_count = counts[i]
                                        counts[i] = old_count - 1
                                        if old_count == 2:
                                            unique_mask |= (1 << i)
                                        elif old_count == 1:
                                            unique_mask &= ~(1 << i)
                                for i in range(num_t):
                                    if (bm_in1 >> i) & 1:
                                        old_count = counts[i]
                                        counts[i] = old_count + 1
                                        if old_count == 0:
                                            unique_mask |= (1 << i)
                                        elif old_count == 1:
                                            unique_mask &= ~(1 << i)
                                for i in range(num_t):
                                    if (bm_in2 >> i) & 1:
                                        old_count = counts[i]
                                        counts[i] = old_count + 1
                                        if old_count == 0:
                                            unique_mask |= (1 << i)
                                        elif old_count == 1:
                                            unique_mask &= ~(1 << i)
                                active_set, counts, unique_mask = minimize(active_set, counts, unique_mask)
                                improved = True
    
    # Simulated annealing on the solution
    def simulated_annealing(active_set, counts, unique_mask, rng, time_budget=25.0):
        sa_start = time.time()
        current_set = set(active_set)
        current_counts = list(counts)
        current_um = unique_mask
        best_set = set(current_set)
        best_counts = list(current_counts)
        best_um = current_um
        best_size = len(current_set)
        
        T = 50.0
        T_min = 0.5
        cooling = 0.95
        
        while T > T_min and (time.time() - sa_start) < time_budget:
            if not current_set:
                break
            
            # Pick a random block from current_set
            b_idx = rng.choice(list(current_set))
            action = rng.random()
            
            if action < 0.4:
                # Remove b_idx
                u_b = block_mask[b_idx] & current_um
                if u_b:
                    # Would create uncovered subsets - accept with probability exp(-delta/T)
                    delta = u_b.bit_count()
                    if rng.random() < np.exp(-delta / T):
                        current_set.remove(b_idx)
                        bm = block_mask[b_idx]
                        for i in range(num_t):
                            if (bm >> i) & 1:
                                old_count = current_counts[i]
                                current_counts[i] = old_count - 1
                                if old_count == 2:
                                    current_um |= (1 << i)
                                elif old_count == 1:
                                    current_um &= ~(1 << i)
                        # Check validity - if invalid, revert
                        # Actually, we should track uncovered
                        # For simplicity, only accept if still valid
                        # Recompute uncovered
                        covered = 0
                        for b in current_set:
                            covered |= block_mask[b]
                        if covered != (1 << num_t) - 1:
                            # Revert
                            current_set.add(b_idx)
                            for i in range(num_t):
                                if (bm >> i) & 1:
                                    current_counts[i] += 1
                                    if current_counts[i] == 1:
                                        current_um |= (1 << i)
                                    elif current_counts[i] == 2:
                                        current_um &= ~(1 << i)
                        else:
                            current_set, current_counts, current_um = minimize(current_set, current_counts, current_um)
                            if len(current_set) < best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                                best_counts = list(current_counts)
                                best_um = current_um
                else:
                    # Safe removal
                    current_set.remove(b_idx)
                    bm = block_mask[b_idx]
                    for i in range(num_t):
                        if (bm >> i) & 1:
                            old_count = current_counts[i]
                            current_counts[i] = old_count - 1
                            if old_count == 2:
                                current_um |= (1 << i)
                    current_set, current_counts, current_um = minimize(current_set, current_counts, current_um)
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                        best_counts = list(current_counts)
                        best_um = current_um
            elif action < 0.8:
                # Swap: remove b_idx, add a random block from covering_blocks of a random unique subset
                u_b = block_mask[b_idx] & current_um
                if u_b:
                    temp = u_b
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    cands = [c for c in covering_blocks[idx] if c != b_idx and c not in current_set]
                    if cands:
                        cand = rng.choice(cands)
                        # Check if cand covers all of u_b
                        if (block_mask[cand] & u_b) == u_b:
                            current_set.remove(b_idx)
                            current_set.add(cand)
                            bm_out = block_mask[b_idx]
                            bm_in = block_mask[cand]
                            for i in range(num_t):
                                if (bm_out >> i) & 1:
                                    old_count = current_counts[i]
                                    current_counts[i] = old_count - 1
                                    if old_count == 2:
                                        current_um |= (1 << i)
                                    elif old_count == 1:
                                        current_um &= ~(1 << i)
                            for i in range(num_t):
                                if (bm_in >> i) & 1:
                                    old_count = current_counts[i]
                                    current_counts[i] = old_count + 1
                                    if old_count == 0:
                                        current_um |= (1 << i)
                                    elif old_count == 1:
                                        current_um &= ~(1 << i)
                            current_set, current_counts, current_um = minimize(current_set, current_counts, current_um)
                            if len(current_set) < best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                                best_counts = list(current_counts)
                                best_um = current_um
            else:
                # Add a random block, then minimize
                cand = rng.randrange(num_k)
                if cand not in current_set:
                    current_set.add(cand)
                    bm = block_mask[cand]
                    for i in range(num_t):
                        if (bm >> i) & 1:
                            old_count = current_counts[i]
                            current_counts[i] = old_count + 1
                            if old_count == 0:
                                current_um |= (1 << i)
                            elif old_count == 1:
                                current_um &= ~(1 << i)
                    current_set, current_counts, current_um = minimize(current_set, current_counts, current_um)
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                        best_counts = list(current_counts)
                        best_um = current_um
            
            T *= cooling
        
        return best_set, best_counts, best_um
    
    # Main logic
    best_solution = None
    best_count = float('inf')
    
    # Multiple restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        noise = 0.0 if r == 0 else 0.15
        seed = r * 12345 + 67890
        active_set, counts = greedy_construct(noise, seed)
        active_set, counts, unique_mask = minimize(active_set, counts)
        
        # Local search with time budget
        ls_budget = min(15.0, (deadline - time.time()) * 0.3)
        if ls_budget > 2:
            rng = random.Random(seed + 1)
            active_set, counts, unique_mask = local_search(active_set, counts, unique_mask, rng, ls_budget)
        
        # Simulated annealing
        sa_budget = min(20.0, (deadline - time.time()) * 0.4)
        if sa_budget > 3:
            rng = random.Random(seed + 2)
            active_set, counts, unique_mask = simulated_annealing(active_set, counts, unique_mask, rng, sa_budget)
        
        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_solution = set(active_set)
    
    if best_solution is None:
        active_set, counts = greedy_construct(0.0, 0)
        active_set, counts, unique_mask = minimize(active_set, counts)
        best_solution = active_set
    
    result = [list(all_k_subsets[b_idx]) for b_idx in best_solution]
    return result
