import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed, strategy="gain"):
        """
        Greedy set cover with different strategies.
        strategy: "gain" = max total uncovered covered, "hardest" = prioritize rare subsets,
                  "conflict" = maximize new/overlap ratio to penalize redundancy
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        # For "hardest" strategy: track how many blocks cover each t-subset
        t_cover_count = np.zeros(n_t, dtype=np.int32)
        for i in range(n_k):
            for t_idx in k_sub_t_idxs[i]:
                t_cover_count[t_idx] += 1
        
        # For "conflict" strategy: track overlap with selected blocks
        overlap_count = np.zeros(n_t, dtype=np.int32)
        
        selected = []
        
        for _ in range(n_k):  # Safety bound
            if not uncovered.any():
                break
            
            if strategy == "gain":
                scores = gains.astype(np.float64).copy()
                if seed > 0:
                    scores += rng.random() * 0.01
                best_idx = int(np.argmax(scores))
                if scores[best_idx] < 0.5:
                    break
            elif strategy == "hardest":
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    break
                min_cov = np.min(t_cover_count[unc_idx])
                hard_ts = [i for i in unc_idx if t_cover_count[i] == min_cov]
                target_t = rng.choice(hard_ts)
                candidates = [b for b in t_sub_to_k_idxs[target_t] if gains[b] > 0]
                if not candidates:
                    break
                best_idx = max(candidates, key=lambda b: gains[b] + rng.random() * 0.01)
            else:  # "conflict" strategy
                # Score = newly_covered / (1 + overlap_with_selected)
                best_idx = -1
                best_score = -1.0
                # Sample a subset of candidates for speed
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    break
                # Pick a random uncovered t-subset to guide candidate selection
                target_t = rng.choice(unc_idx.tolist())
                candidates = [b for b in t_sub_to_k_idxs[target_t] if gains[b] > 0]
                if not candidates:
                    break
                # Score each candidate
                for b in candidates:
                    newly = 0
                    total_overlap = 0
                    for t_idx in k_sub_t_idxs[b]:
                        if uncovered[t_idx]:
                            newly += 1
                        total_overlap += 1
                    # total_overlap here is just len(k_sub_t_idxs[b]), but we want
                    # how many of its t-subsets are already covered by selected blocks
                    already_covered = 0
                    for t_idx in k_sub_t_idxs[b]:
                        if not uncovered[t_idx]:
                            already_covered += 1
                    score = newly / (1.0 + already_covered * 0.5)
                    if score > best_score:
                        best_score = score
                        best_idx = b
                if best_idx < 0:
                    break
            
            selected.append(best_idx)
            
            # Update gains, t_cover_count, and overlap_count
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    t_cover_count[t_idx] -= 1
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                # Track overlap for conflict strategy
                overlap_count[t_idx] += 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass with random ordering for better results."""
        if not blocks_idx:
            return blocks_idx

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        rng = random.Random(99)
        changed = True
        max_rounds = 50
        round_num = 0
        while changed and round_num < max_rounds:
            changed = False
            round_num += 1
            new_blocks = []
            # Shuffle for different removal order each round
            shuffled = list(blocks_idx)
            rng.shuffle(shuffled)
            for b_idx in shuffled:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def simulated_annealing(initial_blocks, max_time):
        """
        Simulated Annealing for covering design optimization.
        State: list of block indices
        Energy: len(blocks) + penalty * num_uncovered
        Moves: remove, swap, add
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        num_uncovered = int(np.sum(cover_count == 0))
        penalty = 500.0
        current_energy = len(current_blocks) + penalty * num_uncovered
        
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        rng = random.Random(42)
        T = 30.0
        cooling_rate = 0.995
        min_T = 0.05
        
        while time.time() < max_time and T > min_T:
            move_type = rng.random()
            
            if move_type < 0.40 and len(current_blocks) > 0:
                # Try removal
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                new_uncovered = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        new_uncovered += 1
                
                if new_uncovered == 0:
                    # Valid removal
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    # Invalid removal - accept with probability
                    delta_E = 1.0 + penalty * new_uncovered
                    if rng.random() < np.exp(-delta_E / T):
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E
            
            elif move_type < 0.85 and len(current_blocks) > 0:
                # Try swap
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                # Find unique t-subsets (covered only by old_b)
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    # Candidate must cover all unique t-subsets
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    # No unique subsets, try random candidates
                    candidates = set()
                    for _ in range(10):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                
                if candidates:
                    new_b = rng.choice(list(candidates))
                    
                    # Calculate energy change
                    delta_size = 0  # swap keeps size same
                    delta_uncovered = 0
                    
                    # Temporarily remove old_b to see what becomes uncovered
                    temp_cover = cover_count.copy()
                    for t_idx in k_sub_t_idxs[old_b]:
                        temp_cover[t_idx] -= 1
                    newly_uncovered = np.sum(temp_cover == 0)
                    
                    # Add new_b back
                    for t_idx in k_sub_t_idxs[new_b]:
                        temp_cover[t_idx] += 1
                    after_add_uncovered = np.sum(temp_cover == 0)
                    
                    delta_E = delta_size + penalty * (after_add_uncovered - num_uncovered)
                    
                    if delta_E <= 0:
                        # Accept (improvement or neutral)
                        for t_idx in k_sub_t_idxs[old_b]:
                            cover_count[t_idx] -= 1
                        for t_idx in k_sub_t_idxs[new_b]:
                            cover_count[t_idx] += 1
                        current_blocks[i] = new_b
                        current_energy += delta_E
                    else:
                        # Accept with probability
                        if rng.random() < np.exp(-delta_E / T):
                            for t_idx in k_sub_t_idxs[old_b]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[new_b]:
                                cover_count[t_idx] += 1
                            current_blocks[i] = new_b
                            current_energy += delta_E
            
            else:
                # Try add (only if we have uncovered subsets)
                if num_uncovered > 0:
                    # Find an uncovered t-subset
                    unc_idx = np.where(cover_count == 0)[0]
                    if len(unc_idx) > 0:
                        target_t = rng.choice(unc_idx.tolist())
                        candidates = [b for b in t_sub_to_k_idxs[target_t] if b not in current_blocks]
                        if candidates:
                            new_b = rng.choice(candidates)
                            
                            # Calculate energy change
                            delta_size = 1
                            # Count how many newly covered
                            newly_covered = 0
                            for t_idx in k_sub_t_idxs[new_b]:
                                if cover_count[t_idx] == 0:
                                    newly_covered += 1
                            
                            delta_E = delta_size - penalty * newly_covered
                            
                            if delta_E <= 0:
                                # Accept
                                for t_idx in k_sub_t_idxs[new_b]:
                                    cover_count[t_idx] += 1
                                current_blocks.append(new_b)
                                current_energy += delta_E
                            else:
                                if rng.random() < np.exp(-delta_E / T):
                                    for t_idx in k_sub_t_idxs[new_b]:
                                        cover_count[t_idx] += 1
                                    current_blocks.append(new_b)
                                    current_energy += delta_E
            
            # Update num_uncovered
            num_uncovered = int(np.sum(cover_count == 0))
            
            # Track best
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
            
            # Cool down
            T *= cooling_rate
            
            # Periodic reduction
            if int((T - min_T) / 0.5) != int((T / cooling_rate - min_T) / 0.5):
                reduced = reduce_blocks(current_blocks)
                if len(reduced) < len(current_blocks):
                    # Rebuild cover count
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx in reduced:
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] += 1
                    current_blocks = reduced
                    num_uncovered = int(np.sum(cover_count == 0))
                    current_energy = len(current_blocks) + penalty * num_uncovered
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(current_blocks)
        
        return best_blocks

    def local_search_swap(initial_blocks, max_time):
        """
        Local search focused on swaps and removals.
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        if np.any(cover_count == 0):
            return current_blocks

        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(123)
        no_improve_count = 0
        max_no_improve = 600
        
        while time.time() < max_time and no_improve_count < max_no_improve:
            move_type = rng.random()
            
            if move_type < 0.40:
                # Try removal
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                removable = True
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        removable = False
                        break
                
                if removable:
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    no_improve_count = 0
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    no_improve_count += 1

            elif move_type < 0.90:
                # Try swap
                if len(current_blocks) == 0:
                    no_improve_count += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    candidates = set()
                    for _ in range(12):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                            break
                
                if not candidates:
                    no_improve_count += 1
                    continue
                
                new_b = rng.choice(list(candidates))
                
                for t_idx in k_sub_t_idxs[old_b]:
                    cover_count[t_idx] -= 1
                for t_idx in k_sub_t_idxs[new_b]:
                    cover_count[t_idx] += 1
                
                current_blocks[i] = new_b
                no_improve_count = 0
                
                # Aggressive reduction after swap
                changed = True
                limit = 6
                while changed and limit > 0:
                    changed = False
                    limit -= 1
                    rng.shuffle(current_blocks)
                    for j in range(len(current_blocks)):
                        b_idx = current_blocks[j]
                        is_essential = False
                        for t_idx in k_sub_t_idxs[b_idx]:
                            if cover_count[t_idx] == 1:
                                is_essential = True
                                break
                        if not is_essential:
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            current_blocks.pop(j)
                            changed = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = list(current_blocks)
                            break

            else:
                no_improve_count += 1

        return best_blocks

    # Run initial greedy with "gain" strategy
    blocks_idx = greedy_cover(0, "gain")
    blocks_idx = reduce_blocks(blocks_idx)
    
    best_blocks = blocks_idx
    best_len = len(blocks_idx)
    
    # Try "conflict-aware" greedy
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        b_idx = greedy_cover(0, "conflict")
        b_idx = reduce_blocks(b_idx)
        if b_idx and len(b_idx) < best_len:
            best_blocks = b_idx
            best_len = len(b_idx)
    
    # Try multiple strategies and seeds if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Try "hardest" strategy
        b_idx = greedy_cover(42, "hardest")
        b_idx = reduce_blocks(b_idx)
        if b_idx and len(b_idx) < best_len:
            best_blocks = b_idx
            best_len = len(b_idx)
        
        # Try randomized greedy with different seeds
        remaining_time = deadline - time.time()
        if remaining_time > 8.0:
            seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
            for seed in seeds:
                if time.time() >= deadline - 6.0:
                    break
                b_idx = greedy_cover(seed, "gain")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)
        
        # Try "conflict" with different seeds
        remaining_time = deadline - time.time()
        if remaining_time > 6.0:
            seeds = [5, 15, 25, 35, 45]
            for seed in seeds:
                if time.time() >= deadline - 5.0:
                    break
                b_idx = greedy_cover(seed, "conflict")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)
        
        # Try "hardest" with different seeds
        remaining_time = deadline - time.time()
        if remaining_time > 6.0:
            seeds = [10, 20, 30, 50, 70]
            for seed in seeds:
                if time.time() >= deadline - 5.0:
                    break
                b_idx = greedy_cover(seed, "hardest")
                b_idx = reduce_blocks(b_idx)
                if b_idx and len(b_idx) < best_len:
                    best_blocks = b_idx
                    best_len = len(b_idx)

    # Run Simulated Annealing on the best solution found
    if best_blocks and time.time() < deadline - 3.0:
        sa_time = min(25.0, deadline - time.time() - 3.0)
        if sa_time > 5.0:
            sa_result = simulated_annealing(best_blocks, time.time() + sa_time)
            if len(sa_result) < best_len:
                best_blocks = sa_result
                best_len = len(best_blocks)
    
    # Run Local Search on the best solution found
    if best_blocks and time.time() < deadline - 2.0:
        ls_time = min(12.0, deadline - time.time() - 2.0)
        if ls_time > 3.0:
            ls_result = local_search_swap(best_blocks, time.time() + ls_time)
            if len(ls_result) < best_len:
                best_blocks = ls_result
                best_len = len(best_blocks)
    
    # Run a second local search pass with different seed if time allows
    if best_blocks and time.time() < deadline - 2.0:
        ls_time2 = min(8.0, deadline - time.time() - 2.0)
        if ls_time2 > 2.0:
            # Modify the local search to use a different seed by calling again
            ls_result2 = local_search_swap(best_blocks, time.time() + ls_time2)
            if len(ls_result2) < best_len:
                best_blocks = ls_result2
                best_len = len(best_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 1.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
