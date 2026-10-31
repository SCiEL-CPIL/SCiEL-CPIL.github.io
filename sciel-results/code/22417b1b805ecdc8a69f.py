import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsubs = {}
    tsub_to_blocks = {i: [] for i in range(n_tsubsets)}
    
    for b in all_blocks:
        covered_tsubs = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered_tsubs.append(idx)
            tsub_to_blocks[idx].append(b)
        block_to_tsubs[b] = covered_tsubs

    def greedy_init():
        # Standard Greedy Set Cover
        # uncovered_count[ts_idx] = 1 if uncovered, 0 if covered
        # We want to pick block maximizing sum(uncovered_count[ts] for ts in block)
        
        uncovered = [1] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        
        while n_uncovered > 0:
            best_block = None
            best_gain = -1
            
            # Optimization: We can't just scan all blocks every time if it's too slow.
            # C(19,7) = 50388.
            # Steps ~ 150.
            # 50388 * 150 * 35 (avg tsubs per block) = ~260M operations.
            # In Python, this might be too slow for the greedy phase alone.
            # Let's use the random-uncovered heuristic but make it smarter?
            # Or just use the standard greedy and hope it's fast enough?
            # 260M simple integer additions/comparisons in Python is ~2-5 seconds.
            # This is acceptable within 115s budget, leaving plenty for SA.
            
            for b in all_blocks:
                if b in block_set:
                    continue
                gain = 0
                for ts_idx in block_to_tsubs[b]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None or best_gain == 0:
                # Should not happen for valid covering parameters
                break
                
            blocks.append(best_block)
            block_set.add(best_block)
            
            for ts_idx in block_to_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                    n_uncovered -= 1
                    
        return blocks

    def calculate_energy(blocks):
        # Returns number of uncovered t-subsets
        coverage = [0] * n_tsubsets
        for b in blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        return sum(1 for c in coverage if c == 0)

    def simulated_annealing(init_blocks):
        # Initialize state
        current_blocks = list(init_blocks)
        current_block_set = set(current_blocks)
        
        # Coverage counts
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        # Current Energy (number of uncovered)
        current_energy = sum(1 for c in coverage if c == 0)
        
        best_blocks = current_blocks[:]
        best_energy = current_energy
        best_size = len(current_blocks)
        
        # SA Parameters
        T = 10.0
        T_min = 0.01
        alpha = 0.995 # Cooling rate
        max_iterations = 50000
        
        current_time = time.time()
        
        # Helper to calculate delta energy for a move
        # Move: Remove block b
        def get_remove_delta(b):
            delta = 0
            for ts_idx in block_to_tsubs[b]:
                if coverage[ts_idx] == 1:
                    delta += 1
            return delta

        # Move: Swap b_old with b_new
        def get_swap_delta(b_old, b_new):
            # Remove b_old, Add b_new
            # Delta = NewUncovered - OldUncovered
            # We can compute this by checking the symmetric difference of their t-subsets
            # Or simply:
            # 1. Decrement coverage for b_old
            # 2. Increment coverage for b_new
            # 3. Count changes in zero-status
            # But we need to revert if rejected.
            # Let's just compute the new energy directly for the swap?
            # That's O(k) to update coverage, then O(N) to count zeros? No.
            
            # Efficient delta:
            # Let S_old be tsubs of b_old, S_new be tsubs of b_new.
            # For ts in S_old: if coverage[ts] == 1, it becomes 0 (increase energy by 1) unless it's in S_new and coverage[ts] was 0?
            # This is tricky.
            
            # Simpler:
            # Just apply the move to a temporary copy of coverage?
            # No, that's O(N).
            
            # Let's use the property:
            # Energy is count of zeros.
            # If we remove b_old:
            #   ts in S_old where coverage[ts]==1 -> becomes 0 (+1 energy)
            # If we add b_new:
            #   ts in S_new where coverage[ts]==0 -> becomes 1 (-1 energy)
            #   BUT if ts is in both S_old and S_new, and coverage[ts]==1:
            #      Remove: 1->0 (+1)
            #      Add: 0->1 (-1)
            #      Net 0.
            #   If ts is in both S_old and S_new, and coverage[ts]==0:
            #      (Impossible, coverage[ts] is global count. If it's 0, b_old doesn't cover it? No, b_old covers it, so coverage[ts] >= 1. So this case doesn't exist for b_old's tsubs).
            #      Wait, coverage[ts] is the count in the *current* solution.
            #      If b_old is in solution, coverage[ts] >= 1 for all ts in S_old.
            
            # So:
            # Delta from removing b_old:
            #   +1 for each ts in S_old where coverage[ts] == 1.
            # Delta from adding b_new:
            #   -1 for each ts in S_new where (coverage[ts] - (1 if ts in S_old else 0)) == 0.
            #   i.e., ts in S_new such that coverage[ts] == 1 AND ts NOT in S_old, OR coverage[ts] == 0 AND ts in S_old?
            #   Let's just calculate the new coverage for the affected tsubs.
            
            affected = set(block_to_tsubs[b_old])
            affected.update(block_to_tsubs[b_new])
            
            old_energy_part = 0
            new_energy_part = 0
            
            for ts in affected:
                old_c = coverage[ts]
                new_c = old_c
                if ts in block_to_tsubs[b_old]:
                    new_c -= 1
                if ts in block_to_tsubs[b_new]:
                    new_c += 1
                
                if old_c == 0:
                    old_energy_part += 1
                if new_c == 0:
                    new_energy_part += 1
                    
            return new_energy_part - old_energy_part

        for _ in range(max_iterations):
            if (time.time() - start_time) > time_limit:
                break
                
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove
                if len(current_blocks) == 0:
                    continue
                idx = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[idx]
                delta = get_remove_delta(b_old)
                
                # Acceptance
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Apply
                    current_blocks.pop(idx)
                    current_block_set.remove(b_old)
                    for ts_idx in block_to_tsubs[b_old]:
                        coverage[ts_idx] -= 1
                    current_energy += delta
                    
                    if current_energy == 0 and len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_energy = 0
                        best_blocks = current_blocks[:]
                        
            elif move_type < 0.8:
                # Swap
                if len(current_blocks) == 0:
                    continue
                idx = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[idx]
                
                # Pick random b_new not in set
                # To avoid O(N) search for random not-in-set, we can pick random from all_blocks
                # and check if it's in set. If it is, retry a few times?
                # Or just pick random index in all_blocks.
                b_new = random.choice(all_blocks)
                if b_new in current_block_set:
                    continue
                
                delta = get_swap_delta(b_old, b_new)
                
                # Acceptance
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Apply
                    current_blocks[idx] = b_new
                    current_block_set.remove(b_old)
                    current_block_set.add(b_new)
                    
                    for ts_idx in block_to_tsubs[b_old]:
                        coverage[ts_idx] -= 1
                    for ts_idx in block_to_tsubs[b_new]:
                        coverage[ts_idx] += 1
                        
                    current_energy += delta
                    
                    if current_energy == 0 and len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_energy = 0
                        best_blocks = current_blocks[:]
                        
            else:
                # Add (rarely, to escape? No, minimization. Maybe skip Add?)
                # Actually, adding a block increases size. It only makes sense if it allows
                # removing 2+ blocks later? SA is single-step.
                # So Add is generally bad for size. We'll skip it or make it very rare.
                pass

            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_blocks

    # Run
    init_blocks = greedy_init()
    result_blocks = simulated_annealing(init_blocks)
    
    # Ensure validity
    # (SA should maintain validity of best_blocks, but let's double check)
    # If best_blocks is invalid, we should return init_blocks?
    # The code only updates best_blocks if current_energy == 0.
    # So best_blocks is always valid.
    
    return [list(b) for b in result_blocks]
