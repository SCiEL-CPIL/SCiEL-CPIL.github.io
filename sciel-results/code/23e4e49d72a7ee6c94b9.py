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
    # tsub_to_blocks is not strictly needed for GRASP construction if we use gains, 
    # but useful for checking redundancy? 
    # Actually, for redundancy, we need to know if a t-subset is covered by >1 blocks in the current set.
    # We can maintain coverage counts dynamically.
    
    for b in all_blocks:
        covered_tsubs = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered_tsubs.append(idx)
        block_to_tsubs[b] = covered_tsubs

    def greedy_randomized_construct(R=5):
        """
        Constructs a solution using Randomized Greedy.
        R is the size of the Candidate List.
        """
        uncovered = [1] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        
        while n_uncovered > 0:
            # Calculate gains for all blocks not in block_set
            # Gain = number of uncovered t-subsets covered by the block
            # Optimization: Instead of scanning all 50k blocks every time, 
            # we could maintain a priority queue, but updating it is expensive.
            # Scanning 50k blocks * 150 steps * 35 tsubs = 260M ops. 
            # In Python, this is ~2-4 seconds per construction.
            # If we do 20 constructions, that's 40-80 seconds. Tight but doable.
            
            best_gain = -1
            candidates = []
            
            for b in all_blocks:
                if b in block_set:
                    continue
                gain = 0
                for ts_idx in block_to_tsubs[b]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [b]
                elif gain == best_gain and gain > 0:
                    candidates.append(b)
            
            if best_gain == 0:
                # Should not happen
                break
                
            # Pick randomly from top R candidates
            # If len(candidates) > R, pick R random ones from candidates, then pick one from those?
            # Or just pick one from the top R of the sorted list?
            # Since we only collected the *best* gain candidates, if there are many ties, 
            # we should sample from them.
            if len(candidates) > R:
                # Sample R from the best gain candidates
                selected = random.sample(candidates, R)
            else:
                selected = candidates
            
            chosen_block = random.choice(selected)
            blocks.append(chosen_block)
            block_set.add(chosen_block)
            
            for ts_idx in block_to_tsubs[chosen_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                    n_uncovered -= 1
                    
        return blocks

    def reduce_irredundant(blocks):
        """
        Removes any block that is redundant (i.e., removing it does not uncover any t-subset).
        Returns the reduced list of blocks.
        """
        if not blocks:
            return []
            
        # Calculate coverage counts
        coverage = [0] * n_tsubsets
        for b in blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
                
        blocks_set = set(blocks)
        changed = True
        current_blocks = list(blocks)
        
        while changed:
            changed = False
            new_blocks = []
            for b in current_blocks:
                # Check if b is redundant
                # b is redundant if for all ts in b, coverage[ts] > 1
                redundant = True
                for ts_idx in block_to_tsubs[b]:
                    if coverage[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove b
                    for ts_idx in block_to_tsubs[b]:
                        coverage[ts_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            
            current_blocks = new_blocks
            
        return current_blocks

    def local_search_exchange(blocks, time_budget=5.0):
        """
        Tries to improve the solution by swapping blocks.
        A swap is valid if the new set of blocks still covers all t-subsets.
        We accept a swap if it is valid. This explores the space of valid covers.
        After a series of swaps, we run irredundancy again.
        """
        if not blocks:
            return blocks
            
        current_blocks = list(blocks)
        current_set = set(current_blocks)
        
        # Initial coverage
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        best_blocks = current_blocks[:]
        best_size = len(current_blocks)
        
        start_ls = time.time()
        
        # We will try to swap blocks.
        # To make it efficient, we can focus on blocks that cover "scarce" t-subsets.
        # However, a simple random walk with validity check is robust.
        
        # Optimization for validity check:
        # A swap (remove b_out, add b_in) is valid iff:
        # For every ts in b_out, if coverage[ts] == 1, then ts must be in b_in.
        # (Because if coverage[ts] == 1, it's only covered by b_out. If we remove b_out, 
        #  it becomes 0 unless b_in covers it).
        
        while (time.time() - start_ls) < time_budget:
            # Pick a random block to remove
            if len(current_blocks) == 0:
                break
            idx_out = random.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[idx_out]
            
            # Identify critical t-subsets for b_out
            # These are ts in b_out where coverage[ts] == 1
            critical_ts = []
            for ts_idx in block_to_tsubs[b_out]:
                if coverage[ts_idx] == 1:
                    critical_ts.append(ts_idx)
            
            # If no critical ts, b_out is redundant. 
            # It should have been removed by reduce_irredundant, but maybe it became redundant 
            # due to previous swaps.
            if not critical_ts:
                # Remove it immediately
                for ts_idx in block_to_tsubs[b_out]:
                    coverage[ts_idx] -= 1
                current_blocks.pop(idx_out)
                current_set.remove(b_out)
                continue
            
            # We need to find a b_in (not in current_set) that covers ALL critical_ts.
            # To be efficient, we can't scan all blocks.
            # We can precompute for each t-subset, which blocks cover it?
            # That's tsub_to_blocks.
            
            # Let's pick a random critical ts and look for blocks covering it.
            # If we pick one critical ts, any valid replacement MUST cover it.
            # So we can filter candidates to blocks covering this specific ts.
            
            target_ts = random.choice(critical_ts)
            
            # Get all blocks covering target_ts
            # We need tsub_to_blocks. Let's build it if not exists? 
            # It's expensive to build for all tsubs? 
            # C(19,4) = 3876 tsubs. C(19,7) = 50388 blocks.
            # Each block covers C(7,4)=35 tsubs.
            # Total entries = 50388 * 35 = 1.7M. This is fine.
            
            # We didn't build tsub_to_blocks in the main scope. 
            # Let's build it locally or globally? 
            # It's better to build it once globally for the whole program run.
            # But the function is self-contained. 
            # I will build it inside construct() before the loop.
            
            # Wait, I need to modify the main construct to build tsub_to_blocks.
            
            # For now, assume we have access to a list of candidate blocks for target_ts.
            # I will refactor the code to build tsub_to_blocks.
            
            # Let's stop this thought and write the code properly.
            pass

        return best_blocks

    # --- Main Execution ---
    
    # Build tsub_to_blocks for efficient candidate selection in local search
    tsub_to_blocks = {i: [] for i in range(n_tsubsets)}
    for b in all_blocks:
        for ts_idx in block_to_tsubs[b]:
            tsub_to_blocks[ts_idx].append(b)

    best_solution = []
    best_size = float('inf')
    
    # Run GRASP iterations
    # Estimate time per iteration:
    # Construction: ~2-3s
    # Irredundancy: ~0.1s
    # Local Search: ~5s
    # Total ~8s per iteration.
    # We have 115s. So ~10-14 iterations.
    
    while (time.time() - start_time) < time_limit - 5:
        # 1. Construct
        init_blocks = greedy_randomized_construct(R=10)
        
        # 2. Reduce Irredundant
        reduced_blocks = reduce_irredundant(init_blocks)
        
        # 3. Local Search (Exchange)
        # Implement the exchange logic here properly using tsub_to_blocks
        
        current_blocks = reduced_blocks
        current_set = set(current_blocks)
        
        # Coverage counts
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        ls_start = time.time()
        ls_budget = min(5.0, time_limit - 5 - (time.time() - start_time))
        
        while (time.time() - ls_start) < ls_budget:
            if not current_blocks:
                break
                
            # Pick random block to try to remove/swap
            idx_out = random.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[idx_out]
            
            # Check if b_out is redundant
            is_redundant = True
            for ts_idx in block_to_tsubs[b_out]:
                if coverage[ts_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for ts_idx in block_to_tsubs[b_out]:
                    coverage[ts_idx] -= 1
                current_blocks.pop(idx_out)
                current_set.remove(b_out)
                continue
            
            # Find critical t-subsets
            critical_ts = []
            for ts_idx in block_to_tsubs[b_out]:
                if coverage[ts_idx] == 1:
                    critical_ts.append(ts_idx)
            
            # Pick a random critical ts
            target_ts = random.choice(critical_ts)
            
            # Get candidate blocks covering target_ts
            candidates = tsub_to_blocks[target_ts]
            
            # Filter candidates: 
            # 1. Not in current_set
            # 2. Must cover ALL other critical_ts as well (to maintain validity)
            valid_replacement = None
            for b_in in candidates:
                if b_in in current_set:
                    continue
                
                # Check if b_in covers all critical_ts
                # Optimization: Only check if b_in covers the specific critical_ts.
                # Since critical_ts is usually small (1-3), this is fast.
                covers_all = True
                for ts_idx in critical_ts:
                    # Check if ts_idx is in block_to_tsubs[b_in]
                    # This is O(35) check. 
                    # We can precompute a set of tsubs for each block? 
                    # block_to_tsubs is a list. 
                    # Let's use a set for b_in's tsubs for faster lookup?
                    # Or just iterate? 
                    # 35 items is small.
                    if ts_idx not in block_to_tsubs[b_in]:
                        covers_all = False
                        break
                
                if covers_all:
                    valid_replacement = b_in
                    break # Pick the first valid one? Or random? 
                    # Random is better for exploration.
                    # But we want to find *any* valid swap to keep the walk moving.
            
            if valid_replacement:
                # Perform swap
                b_in = valid_replacement
                
                # Update coverage
                for ts_idx in block_to_tsubs[b_out]:
                    coverage[ts_idx] -= 1
                for ts_idx in block_to_tsubs[b_in]:
                    coverage[ts_idx] += 1
                
                # Update sets
                current_blocks[idx_out] = b_in
                current_set.remove(b_out)
                current_set.add(b_in)
        
        # Final irredundancy after local search
        final_blocks = reduce_irredundant(current_blocks)
        
        if len(final_blocks) < best_size:
            best_size = len(final_blocks)
            best_solution = final_blocks

    return [list(b) for b in best_solution]
