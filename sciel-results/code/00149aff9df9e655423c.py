import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # 1. Precompute all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # 3. For each k-subset, compute the list of t-subset indices it covers
    # k_sub_t_idxs[i] is a list of t-subset indices covered by k-subset i
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)

    # 4. Build reverse mapping: for each t-subset, which k-subsets cover it
    # t_sub_to_k_idxs[j] is a list of k-subset indices covering t-subset j
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """
        Efficient greedy set cover using NumPy for gain updates.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # Initial gains: number of t-subsets covered by each block
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        selected = []
        
        for _ in range(n_k): # Safety bound
            if not uncovered.any():
                break
            if time.time() > deadline - 5.0:
                break
            
            # Find best block with positive gain
            # Use a copy to avoid modifying the original gains array during argmax
            active_gains = gains.copy()
            
            # Add small random noise to break ties when seed > 0
            if seed > 0:
                noise = rng.random() * 0.01
                active_gains += noise
            
            best_idx = int(np.argmax(active_gains))
            
            if active_gains[best_idx] < 0.5: # Gain is integer, so < 0.5 means 0
                break
                
            selected.append(best_idx)
            
            # Update gains: for each t-subset covered by best_idx, decrement gain of all blocks covering it
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
                        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
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

    def is_valid_covering(blocks_idx):
        """Check if the given blocks form a valid covering."""
        if not blocks_idx:
            return n_t == 0
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        return np.all(cover_count > 0)

    def greedy_repair(blocks_idx):
        """Add blocks to make the covering valid if it isn't."""
        if not blocks_idx:
            return greedy_cover(0)
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        uncovered = (cover_count == 0)
        if not uncovered.any():
            return blocks_idx
        
        # Greedily add blocks to cover uncovered t-subsets
        selected = list(blocks_idx)
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        
        # Recompute gains based on current uncovered set
        for i in range(n_k):
            g = 0
            for t_idx in k_sub_t_idxs[i]:
                if uncovered[t_idx]:
                    g += 1
            gains[i] = g
        
        while uncovered.any():
            if time.time() > deadline - 3.0:
                break
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
        
        return selected

    def simulated_annealing(initial_blocks, max_time):
        """
        Simulated Annealing on block selection.
        State: List of block indices.
        Energy: len(blocks) + penalty * num_uncovered
        Moves: 
          1. Remove a block (if valid, always accept; else prob accept)
          2. Swap a block with another that covers its unique subsets (if valid, always accept; else prob accept)
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1
        
        current_energy = len(current_blocks) + 1000.0 * np.sum(cover_count == 0)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        # SA Parameters
        T = 50.0  # Initial temperature
        cooling_rate = 0.99
        min_T = 0.1
        penalty = 1000.0
        
        rng = random.Random(42)
        
        while time.time() < max_time and T > min_T:
            move_type = rng.choice(['remove', 'swap'])
            
            if move_type == 'remove' and len(current_blocks) > 0:
                # Pick random block
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                # Check if removal is valid
                valid_removal = True
                new_uncovered = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        valid_removal = False
                        new_uncovered += 1
                
                if valid_removal:
                    # Always accept
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    # Energy change: +1 (size) + penalty * new_uncovered
                    delta_E = 1.0 + penalty * new_uncovered
                    # Accept with probability exp(-delta_E / T)
                    if rng.random() < np.exp(-delta_E / T):
                        # Accept invalid move
                        for t_idx in k_sub_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E
                T *= cooling_rate
            elif move_type == 'swap' and len(current_blocks) > 0:
                # Pick a random block to remove
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                # Find blocks that could replace it (cover its unique subsets)
                unique_t_idxs = [t_idx for t_idx in k_sub_t_idxs[b_idx] if cover_count[t_idx] == 1]
                
                if not unique_t_idxs:
                    # No unique subsets, just remove
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                    T *= cooling_rate
                else:
                    # Find candidate replacement blocks
                    candidates = []
                    for t_idx in unique_t_idxs:
                        for cand in t_sub_to_k_idxs[t_idx]:
                            if cand != b_idx and cand not in current_blocks:
                                candidates.append(cand)
                    
                    if candidates:
                        # Pick a random candidate
                        new_b_idx = rng.choice(candidates)
                        
                        # Compute energy change
                        old_covered = 0
                        for t_idx in k_sub_t_idxs[b_idx]:
                            if cover_count[t_idx] > 1:
                                old_covered += 1
                        new_covered = 0
                        for t_idx in k_sub_t_idxs[new_b_idx]:
                            if cover_count[t_idx] == 0:
                                new_covered += 1
                            elif cover_count[t_idx] > 1:
                                new_covered += 1
                        # Actually, let's just compute the new uncovered count
                        # Remove b_idx, add new_b_idx
                        new_uncovered_count = 0
                        for t_idx in k_sub_t_idxs[b_idx]:
                            if cover_count[t_idx] == 1:
                                # This t_idx would become uncovered unless new_b_idx covers it
                                if new_b_idx in t_sub_to_k_idxs[t_idx]:
                                    pass  # Still covered
                                else:
                                    new_uncovered_count += 1
                        # But we also need to check if new_b_idx covers any currently uncovered t_idxs
                        # Let's just compute the actual new state
                        new_uncovered = 0
                        for t_idx in range(n_t):
                            cc = cover_count[t_idx]
                            if t_idx in k_sub_t_idxs[b_idx]:
                                cc -= 1
                            if t_idx in k_sub_t_idxs[new_b_idx]:
                                cc += 1
                            if cc == 0:
                                new_uncovered += 1
                        
                        delta_E = (new_uncovered - np.sum(cover_count == 0)) * penalty
                        if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                            # Perform swap
                            for t_idx in k_sub_t_idxs[b_idx]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_t_idxs[new_b_idx]:
                                cover_count[t_idx] += 1
                            current_blocks[i] = new_b_idx
                            current_energy += delta_E
                        T *= cooling_rate
            else:
                # No valid move, cool down
                T *= cooling_rate
            
            # Check if we have a better solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
                
            # Break if we've reached a perfect solution and can't improve
            if np.sum(cover_count == 0) == 0 and current_energy == len(current_blocks):
                # We have a valid covering with no uncovered
                # Continue trying to reduce size
                pass
            else:
                if np.sum(cover_count == 0) == 0:
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(current_blocks)

        return best_blocks

    # --- Main Logic ---
    
    # 1. Get an initial greedy solution
    initial_blocks = greedy_cover(0)
    
    # 2. Reduce redundant blocks
    reduced_blocks = reduce_blocks(initial_blocks)
    
    # 3. Try to improve with Simulated Annealing
    # We have a time budget. Let's reserve 15s for the final repair and return.
    sa_end_time = start_time + 115.0
    improved_blocks = simulated_annealing(reduced_blocks, sa_end_time)
    
    # 4. Ensure validity with greedy repair if needed
    if not is_valid_covering(improved_blocks):
        improved_blocks = greedy_repair(improved_blocks)
    
    # 5. Final reduction
    final_blocks = reduce_blocks(improved_blocks)
    
    # 6. Final validity check
    if not is_valid_covering(final_blocks):
        final_blocks = greedy_repair(final_blocks)
        final_blocks = reduce_blocks(final_blocks)
    
    # 7. Convert indices to blocks
    result_blocks = [list(all_k_subs[i]) for i in final_blocks]
    
    return result_blocks
