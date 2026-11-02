import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a Multi-Start Destroy and Repair heuristic.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # Pre-compute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Map each block to the set of t-subsets it covers
    block_to_subsets = {}
    for b in all_blocks:
        subs = tuple(sorted(itertools.combinations(b, t)))
        block_to_subsets[b] = subs
        
    # Map each t-subset to the list of blocks covering it
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_blocks = {s: [] for s in all_subsets}
    for b, subs in block_to_subsets.items():
        for s in subs:
            subset_to_blocks[s].append(b)

    def greedy_cover(uncovered_set, existing_blocks=None):
        """
        Greedily add blocks to cover all subsets in uncovered_set.
        Returns the list of added blocks.
        """
        if not uncovered_set:
            return []
        
        added_blocks = []
        current_uncovered = set(uncovered_set)
        
        while current_uncovered:
            # Pick a random uncovered subset to break ties and add diversity
            # Or pick the one with fewest covering blocks? 
            # For repair, we want to cover efficiently.
            # Let's pick the subset with the fewest available blocks (most constrained)
            # But we must ensure we don't pick a block already in existing_blocks if we want distinct blocks?
            # The problem requires distinct blocks.
            # So we must track used blocks.
            
            # To be fast, we just pick a random uncovered subset.
            s = random.choice(list(current_uncovered))
            
            # Find best block covering s that is not in existing_blocks
            # And covers the most other subsets in current_uncovered
            best_block = None
            best_gain = -1
            
            candidates = subset_to_blocks[s]
            for b in candidates:
                if existing_blocks and b in existing_blocks:
                    continue
                gain = 0
                for sub in block_to_subsets[b]:
                    if sub in current_uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    
            if best_block is None:
                # Fallback: pick any block covering s
                for b in candidates:
                    if not existing_blocks or b not in existing_blocks:
                        best_block = b
                        break
                if best_block is None:
                    # Should not happen if valid
                    break

            added_blocks.append(best_block)
            if existing_blocks:
                existing_blocks.add(best_block)
                
            # Update current_uncovered
            for sub in block_to_subsets[best_block]:
                current_uncovered.discard(sub)
                
        return added_blocks

    def local_search_minimize(blocks):
        """
        Remove redundant blocks repeatedly.
        Returns the minimized list of blocks.
        """
        blocks = list(blocks)
        # Calculate coverage counts
        coverage_counts = {}
        for b in blocks:
            for sub in block_to_subsets[b]:
                coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for sub in block_to_subsets[b]:
                    if coverage_counts[sub] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    for sub in block_to_subsets[b]:
                        coverage_counts[sub] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart scan from end after removal
        return blocks

    # Main Loop: Multi-Start Destroy and Repair
    best_covering = None
    best_size = float('inf')
    
    # Initial Greedy Cover
    # Start with empty set
    uncovered = set(all_subsets)
    used_blocks = set()
    initial_blocks = []
    
    while uncovered:
        s = random.choice(list(uncovered))
        best_block = None
        best_gain = -1
        for b in subset_to_blocks[s]:
            if b in used_blocks:
                continue
            gain = 0
            for sub in block_to_subsets[b]:
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
        
        if best_block is None:
            break
            
        initial_blocks.append(best_block)
        used_blocks.add(best_block)
        for sub in block_to_subsets[best_block]:
            uncovered.discard(sub)
            
    current_blocks = local_search_minimize(initial_blocks)
    best_covering = current_blocks
    best_size = len(current_blocks)
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        if not current_blocks:
            break
            
        # Perturb: Select a random block to remove
        # Prefer blocks that are "critical" (not redundant) to force change
        # But removing a redundant block is just a removal, which local search already did.
        # So we must remove a non-redundant block.
        
        # Check for redundant blocks first to ensure we are in a local minimum
        current_blocks = local_search_minimize(current_blocks)
        
        # Find a block that is not redundant
        coverage_counts = {}
        for b in current_blocks:
            for sub in block_to_subsets[b]:
                coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
                
        non_redundant_blocks = []
        for b in current_blocks:
            is_red = True
            for sub in block_to_subsets[b]:
                if coverage_counts[sub] == 1:
                    is_red = False
                    break
            if not is_red:
                non_redundant_blocks.append(b)
                
        if not non_redundant_blocks:
            # All blocks are redundant? That means we can remove all? 
            # No, if all are redundant, we can remove one, then re-check.
            # local_search_minimize should have handled this.
            # If we are here, it means no single block is redundant? 
            # Wait, if no block is redundant, we must pick one to destroy.
            # If all are non-redundant, pick one.
            destroy_block = random.choice(current_blocks)
        else:
            destroy_block = random.choice(non_redundant_blocks)
            
        # Remove destroy_block
        new_blocks = [b for b in current_blocks if b != destroy_block]
        
        # Calculate uncovered subsets
        # Subsets that were only covered by destroy_block are now uncovered.
        # But other blocks might cover them too.
        # We need to find which subsets are NOT covered by new_blocks.
        
        # Efficient way: 
        # Count coverage in new_blocks.
        new_coverage = {}
        for b in new_blocks:
            for sub in block_to_subsets[b]:
                new_coverage[sub] = new_coverage.get(sub, 0) + 1
                
        uncovered_set = set()
        for sub in block_to_subsets[destroy_block]:
            if new_coverage.get(sub, 0) == 0:
                uncovered_set.add(sub)
                
        # Repair: Add blocks to cover uncovered_set
        # We must not add destroy_block again (or any block in new_blocks)
        used_in_new = set(new_blocks)
        added_blocks = greedy_cover(uncovered_set, used_in_new)
        
        candidate_blocks = new_blocks + added_blocks
        
        # Minimize the candidate
        candidate_blocks = local_search_minimize(candidate_blocks)
        
        # Compare
        if len(candidate_blocks) < best_size:
            best_size = len(candidate_blocks)
            best_covering = candidate_blocks
            current_blocks = candidate_blocks
        elif len(candidate_blocks) == best_size:
            # Accept with some probability to escape local minima? 
            # Or just keep the best. 
            # Let's accept 50% of the time to explore.
            if random.random() < 0.5:
                current_blocks = candidate_blocks
                best_covering = candidate_blocks
        else:
            # Revert
            # We don't need to do anything if we just track best_covering
            pass

    return [list(b) for b in best_covering]
