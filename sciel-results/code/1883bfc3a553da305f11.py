import itertools
import time
import numpy as np
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: return by 145s to ensure we don't hit the 150s kill switch
    deadline = start_time + 145.0
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute coverage of each block as a set of t-subset indices
    # We use a list of sets for intersection logic, and a numpy array for fast counting if needed
    # But for SA, we need fast "is this t-subset covered?" checks.
    # Bitmask approach:
    # num_t_subsets = 3876. This fits in 61 uint64 words.
    # This might be overkill if we just use boolean arrays.
    # Let's stick to boolean arrays for simplicity and speed in Python/NumPy, 
    # but optimize the update logic.
    
    # block_coverages: list of numpy arrays (int32 indices of t-subsets)
    block_coverages = []
    for block in all_k_blocks:
        covers = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Reverse index: t_subset_idx -> list of block indices covering it
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverages):
        for t_idx in cov:
            t_sub_to_blocks[t_idx].append(b_idx)
    
    # Helper to initialize a valid cover using Greedy Set Union (GSU)
    def greedy_set_union():
        covered = np.zeros(num_t_subsets, dtype=bool)
        current_blocks = []
        
        # Lazy Queue: (negative_gain, block_idx)
        pq = []
        for b_idx in range(num_k_blocks):
            gain = len(block_coverages[b_idx])
            heapq.heappush(pq, (-gain, b_idx))
            
        while True:
            if time.time() > deadline - 20.0:
                break
            if np.all(covered):
                break
                
            neg_gain, b_idx = heapq.heappop(pq)
            
            cov = block_coverages[b_idx]
            new_covered = ~covered[cov]
            current_gain = np.count_nonzero(new_covered)
            
            if current_gain == 0:
                continue
                
            if current_gain < -neg_gain:
                heapq.heappush(pq, (-current_gain, b_idx))
                continue
            
            covered[cov] = True
            current_blocks.append(b_idx)
            
        return current_blocks, covered

    # Simulated Annealing
    def simulated_annealing(initial_blocks, initial_covered):
        # State:
        # blocks: list of block indices
        # counts: numpy array of counts for each t-subset
        # covered: derived from counts > 0
        
        current_blocks = list(initial_blocks)
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b in current_blocks:
            counts[block_coverages[b]] += 1
        
        # Objective function:
        # Primary: Minimize number of blocks
        # Secondary: Minimize "slack" (sum of counts - num_t_subsets) to encourage tighter packing
        # This helps escape local minima where size is fixed but structure is bad.
        
        def get_objective(blocks, counts):
            size = len(blocks)
            # Slack is how much "extra" coverage we have.
            # Total coverage = sum(counts). 
            # Ideal total coverage = num_t_subsets (each covered exactly once).
            # Slack = sum(counts) - num_t_subsets.
            slack = np.sum(counts) - num_t_subsets
            # Weight: We want to reduce size primarily. 
            # But reducing size is hard. Reducing slack makes it easier to swap.
            # Let's use a composite score.
            # Score = size * 1000 + slack
            # If we can find a swap that keeps size but reduces slack, we accept it.
            # If we find a swap that reduces size, we always accept it.
            
            return size * 10000 + slack

        current_score = get_objective(current_blocks, counts)
        best_blocks = list(current_blocks)
        best_counts = counts.copy()
        best_score = current_score
        
        # Temperature schedule
        T_start = 100.0
        T_end = 0.1
        # Number of steps depends on time budget
        # We will run until deadline - 5s
        
        T = T_start
        cooling_rate = 0.9995 # Adjust based on expected steps
        
        while time.time() < deadline - 5.0:
            if np.random.random() < 0.01: # Check time more frequently? No, loop is fast.
                pass
            
            # Move Generation:
            # 1. Swap: Remove one block, add one block.
            # 2. Delete: Remove one block (if valid).
            # 3. Insert: Add one block (if it helps slack? No, usually bad for size, but might help future swaps? Rarely useful for minimization directly).
            
            # Let's focus on Swap and Delete.
            
            move_type = random.random()
            
            if move_type < 0.8: # Swap
                # Pick a random block to remove
                b_remove = random.choice(current_blocks)
                cov_remove = block_coverages[b_remove]
                
                # Find candidate blocks to add.
                # To be a valid swap (size constant), the new block must cover all t-subsets
                # that become uncovered by removing b_remove.
                # Uncovered if counts[t] == 1 and t in b_remove.
                
                critical_ts = cov_remove[counts[cov_remove] == 1]
                
                if len(critical_ts) == 0:
                    # This block is redundant (can be deleted).
                    # Try deletion instead.
                    # For now, treat as a swap with any block? No, let's force deletion logic.
                    # Actually, if critical_ts is empty, we can just delete it.
                    # But SA needs a "new state". 
                    # Let's pick a random block to add to keep size constant, 
                    # which might increase slack but allow exploration?
                    # Or better: Just try to find a block that covers critical_ts.
                    # If critical_ts is empty, ANY block is a valid "swap" in terms of coverage,
                    # but we want to reduce slack. 
                    # So we pick a block that maximizes overlap with existing coverage?
                    # Or just pick a random block.
                    
                    # Let's pick a block that covers the most already covered t-subsets?
                    # That reduces slack.
                    # Actually, let's just pick a random block from all_k_blocks 
                    # that is not b_remove.
                    b_add = random.randint(0, num_k_blocks - 1)
                    if b_add == b_remove:
                        b_add = (b_add + 1) % num_k_blocks
                else:
                    # Need to cover critical_ts
                    # Find intersection of t_sub_to_blocks for each t in critical_ts
                    # Sort critical_ts by length of candidate list to speed up intersection
                    
                    sorted_critical = sorted(critical_ts, key=lambda t: len(t_sub_to_blocks[t]))
                    
                    if len(sorted_critical) == 0:
                        # Should not happen if critical_ts not empty
                        continue
                        
                    candidates = set(t_sub_to_blocks[sorted_critical[0]])
                    candidates.discard(b_remove)
                    
                    for t_idx in sorted_critical[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(t_sub_to_blocks[t_idx])
                        candidates.discard(b_remove)
                        
                    if not candidates:
                        # No valid swap partner. Try a different removal?
                        # For now, skip this move.
                        continue
                        
                    b_add = random.choice(list(candidates))
                
                # Apply Swap
                cov_add = block_coverages[b_add]
                
                # Check if b_add is already in current_blocks
                # If so, it's a "no-op" or duplicate. We should avoid duplicates.
                if b_add in current_blocks:
                    # If b_add is already there, swapping b_remove with b_add just removes b_remove.
                    # This is effectively a deletion.
                    # Let's handle this as a deletion move.
                    # But for now, let's just skip if it leads to duplicate, 
                    # unless we want to allow duplicates? No, blocks must be distinct.
                    # So if b_add is in current_blocks, we cannot add it.
                    # We must pick a different candidate.
                    # Let's re-pick from candidates minus current_blocks.
                    # This is expensive. 
                    # Alternative: Allow the "swap" to just remove b_remove if b_add is present?
                    # No, that changes size.
                    
                    # Let's filter candidates to exclude current_blocks.
                    # This is O(N) per move. N ~ 150. Acceptable.
                    valid_candidates = [c for c in candidates if c not in current_blocks]
                    if not valid_candidates:
                        continue
                    b_add = random.choice(valid_candidates)
                    cov_add = block_coverages[b_add]
                    
                # Update counts
                counts[cov_remove] -= 1
                counts[cov_add] += 1
                
                # Update blocks
                current_blocks.remove(b_remove)
                current_blocks.append(b_add)
                
                new_score = get_objective(current_blocks, counts)
                
                # Acceptance
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_blocks = list(current_blocks)
                        best_counts = counts.copy()
                else:
                    # Revert
                    counts[cov_remove] += 1
                    counts[cov_add] -= 1
                    current_blocks.remove(b_add)
                    current_blocks.append(b_remove)
                    
            else: # Deletion (20% chance)
                # Find a block that can be removed
                # A block can be removed if it has no critical t-subsets (counts==1)
                # We pick a random block and check.
                b_remove = random.choice(current_blocks)
                cov_remove = block_coverages[b_remove]
                
                if np.any(counts[cov_remove] == 1):
                    # Cannot remove
                    continue
                
                # Remove
                counts[cov_remove] -= 1
                current_blocks.remove(b_remove)
                
                new_score = get_objective(current_blocks, counts)
                
                # Always accept deletion if valid (score decreases)
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_blocks = list(current_blocks)
                    best_counts = counts.copy()
                
            # Cooling
            T *= cooling_rate
            if T < T_end:
                T = T_end
        
        return best_blocks

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # We do multiple restarts of SA with different initial GSU solutions
    # GSU is fast, so we can do many.
    
    num_restarts = 50 # Max, limited by time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Greedy Construction
        init_blocks, init_covered = greedy_set_union()
        
        # 2. Simulated Annealing
        # SA takes significant time. We give it a slice of the remaining time.
        # Or just let it run until its internal deadline.
        # Since SA has its own time check, it will stop when needed.
        # But we want to ensure we try multiple restarts.
        # Let's limit SA time per restart.
        sa_deadline = time.time() + (deadline - 10.0 - time.time()) / (num_restarts - i)
        # Modify SA to respect this? 
        # Easier: Just run SA. If it takes too long, we lose restarts.
        # Let's add a time limit to SA explicitly.
        
        # To do this cleanly, I'll pass a deadline to SA.
        # But I defined SA without a deadline param. 
        # Let's just rely on the global deadline in SA.
        # If SA runs for 140s, we get 1 restart.
        # If SA is fast, we get more.
        
        refined_blocks = simulated_annealing(init_blocks, init_covered)
        
        if len(refined_blocks) < best_size:
            best_size = len(refined_blocks)
            best_solution = refined_blocks.copy()
            
            # Early stop if we hit a very good score
            if best_size <= 135:
                break

    if best_solution is None:
        # Fallback
        init_blocks, _ = greedy_set_union()
        best_solution = init_blocks
        
    return [list(all_k_blocks[b_idx]) for b_idx in best_solution]
