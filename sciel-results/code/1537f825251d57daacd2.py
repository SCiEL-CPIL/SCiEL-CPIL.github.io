import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    
    # We will use a greedy set cover approach on a pool of candidate blocks.
    # The pool size and number of restarts will be adjusted based on time.
    
    best_blocks = None
    best_size = float('inf')
    
    # Helper to run one greedy attempt with a specific candidate pool
    def run_greedy(pool):
        """
        pool: list of (block_tuple, set_of_t_subsets_in_block)
        """
        uncovered = set(all_t_subsets)
        selected = []
        candidates = pool[:]
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            best_block_tuple = None
            best_block_subsets = None
            
            # Find the block with max gain
            for i, (bt, bsubs) in enumerate(candidates):
                gain = len(uncovered.intersection(bsubs))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_block_tuple = bt
                    best_block_subsets = bsubs
            
            if best_gain == 0:
                # No candidate covers any uncovered subset. 
                # This implies the pool was insufficient. 
                # In a robust implementation, we might generate more, 
                # but for this task, we assume the pool is large enough 
                # or we break and return invalid/incomplete.
                # Given v=19, k=8, t=5, a random pool of 5000 is likely sufficient 
                # to cover everything if we are lucky, but greedy might get stuck 
                # if the best remaining block has gain 0 but uncovered is not empty.
                # This can happen if the specific combination of remaining uncovered 
                # subsets is not covered by any single remaining candidate.
                # To be safe, if this happens, we return a large penalty or just the current selection.
                # However, the problem statement says "Valid when every t-subset... lies in at least one block".
                # If we return an incomplete cover, it's invalid (score 0).
                # So we must ensure we find a cover.
                # If we get stuck, we can try to construct a block on the fly for the first uncovered subset.
                
                # Fallback: Construct a block containing min(uncovered)
                # This reverts to the seed logic for the specific gap.
                target = min(uncovered)
                block = list(target)
                # Add random points until size k
                while len(block) < k:
                    # Pick random point not in block
                    cand = random.randint(0, v-1)
                    if cand not in block:
                        block.append(cand)
                block.sort()
                block_tuple = tuple(block)
                # Calculate its subsets
                bsubs = set(itertools.combinations(block_tuple, t))
                # Add to selected
                selected.append(block_tuple)
                uncovered.difference_update(bsubs)
                # We don't add this to candidates as it's a one-off fix
                # But we must ensure we don't loop infinitely.
                # Since we add k-t points, we cover some new subsets.
                # If we still have uncovered, we loop again.
                # This fallback is expensive but rare if pool is good.
                continue

            # Add best block to selected
            selected.append(best_block_tuple)
            uncovered.difference_update(best_block_subsets)
            
            # Remove from candidates to avoid re-selection
            # Using a set of indices or just removing from list is O(N).
            # For N=5000, this is fine.
            candidates.pop(best_idx)
            
            # Time check
            if time.time() - start_time > timeout:
                return selected, False # Incomplete
        
        # Redundancy Removal
        # Iterate through selected blocks and try to remove them
        # We do this multiple passes or a single pass is often enough.
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(selected):
                bt = selected[i]
                # Check if bt is essential
                # A block is essential if there exists a t-subset S in bt 
                # such that S is not covered by any other block in selected.
                
                # To check this efficiently:
                # Let S_bt be the set of t-subsets in bt.
                # Let S_others be the union of t-subsets of all other blocks in selected.
                # If S_bt - S_others is not empty, bt is essential.
                
                # This calculation is expensive if done naively for each block.
                # Optimization: 
                # For each t-subset in bt, check if it is covered by another block.
                # We can maintain a count of how many times each t-subset is covered.
                
                # Let's re-structure redundancy removal:
                # 1. Count coverage for all t-subsets from the current 'selected' set.
                # 2. For each block in selected, check if it contains any t-subset with count 1.
                #    If yes, it's essential.
                #    If no, it can be removed.
                # 3. If removed, update counts.
                
                # This is O(N_blocks * C(k,t)) per pass.
                # With N_blocks ~ 300 and C(8,5)=56, this is ~17k operations per block check.
                # Total ~ 5 million ops per pass. Very fast.
                
                pass # Placeholder for logic below
        
        return selected, True

    # Better Redundancy Removal Implementation
    def remove_redundancies(blocks):
        if not blocks:
            return blocks
        
        # Count coverage of each t-subset
        coverage_count = {}
        for b in blocks:
            for s in itertools.combinations(b, t):
                coverage_count[s] = coverage_count.get(s, 0) + 1
        
        new_blocks = []
        for b in blocks:
            essential = False
            for s in itertools.combinations(b, t):
                if coverage_count[s] == 1:
                    essential = True
                    break
            if essential:
                new_blocks.append(b)
        
        # If we removed any, we should check again? 
        # Removing a block might make another block essential or non-essential.
        # Actually, removing a block only *decreases* counts.
        # If a block was non-essential (all its subsets had count > 1), 
        # removing another block might drop a count to 1, making it essential.
        # So we need to iterate.
        
        changed = True
        while changed:
            changed = False
            coverage_count = {}
            for b in new_blocks:
                for s in itertools.combinations(b, t):
                    coverage_count[s] = coverage_count.get(s, 0) + 1
            
            next_blocks = []
            for b in new_blocks:
                essential = False
                for s in itertools.combinations(b, t):
                    if coverage_count[s] == 1:
                        essential = True
                        break
                if essential:
                    next_blocks.append(b)
                else:
                    changed = True
            
            new_blocks = next_blocks
        
        return new_blocks

    # Modified run_greedy to use remove_redundancies
    def run_greedy_v2(pool):
        uncovered = set(all_t_subsets)
        selected = []
        candidates = pool[:]
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            best_block_tuple = None
            best_block_subsets = None
            
            for i, (bt, bsubs) in enumerate(candidates):
                gain = len(uncovered.intersection(bsubs))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_block_tuple = bt
                    best_block_subsets = bsubs
            
            if best_gain == 0:
                # Fallback: construct a block for min(uncovered)
                target = min(uncovered)
                block = list(target)
                while len(block) < k:
                    cand = random.randint(0, v-1)
                    if cand not in block:
                        block.append(cand)
                block.sort()
                block_tuple = tuple(block)
                bsubs = set(itertools.combinations(block_tuple, t))
                selected.append(block_tuple)
                uncovered.difference_update(bsubs)
                continue

            selected.append(best_block_tuple)
            uncovered.difference_update(best_block_subsets)
            candidates.pop(best_idx)
            
            if time.time() - start_time > timeout:
                return selected, False
        
        # Apply redundancy removal
        selected = remove_redundancies(selected)
        return selected, True

    # Main Loop: Multiple restarts with different random pools
    pool_size = 3000
    restarts = 0
    
    while time.time() - start_time < timeout:
        # Generate random pool
        # To ensure diversity, we can mix random blocks with some structured ones?
        # Random is usually fine for covering designs if pool is large enough.
        pool = []
        seen_blocks = set()
        while len(pool) < pool_size:
            b = tuple(sorted(random.sample(range(v), k)))
            if b not in seen_blocks:
                seen_blocks.add(b)
                bsubs = set(itertools.combinations(b, t))
                pool.append((b, bsubs))
        
        res, valid = run_greedy_v2(pool)
        
        if valid:
            if len(res) < best_size:
                best_size = len(res)
                best_blocks = res
        
        restarts += 1
        # If we are running out of time, break
        if time.time() - start_time > timeout - 5:
            break

    if best_blocks is None:
        # Fallback to seed-like behavior if we failed to find a valid cover in time
        # This should not happen, but better safe than sorry.
        # Just return a dummy valid cover if possible? 
        # Or raise error? The prompt says "invalid result... scores 0".
        # We must return *something*. 
        # If best_blocks is None, we can try one more simple greedy without pool.
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                # Greedy point addition
                best_x = -1
                max_gain = -1
                for x in range(v):
                    if x not in block:
                        gain = sum(1 for rest in itertools.combinations(block, t-1) 
                                   if tuple(sorted(rest + (x,))) in uncovered)
                        if gain > max_gain:
                            max_gain = gain
                            best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(tuple(block))
            uncovered.difference_update(set(itertools.combinations(tuple(block), t)))
        return [list(b) for b in blocks]

    return [list(b) for b in best_blocks]
