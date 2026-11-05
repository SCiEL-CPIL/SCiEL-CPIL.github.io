"""Covering design C(24,7,3) using Coverage Count Greedy + Local Search."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer for safety

    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute the list of triple indices for every possible block?
    # No, too much memory/time. Compute on the fly but cache if needed?
    # For v=24, k=7, t=3, we have ~346k blocks.
    # Storing 35 ints per block is ~12MB. Feasible.
    # But generating them takes time.
    # Let's compute on the fly in the inner loop, it's fast enough.
    
    def get_block_triple_indices(block):
        """Return list of triple indices covered by this block."""
        return [triple_to_idx[tuple(sorted(c))] for c in itertools.combinations(block, t)]

    # State: coverage_count array
    coverage_count = [0] * num_triples
    blocks = []
    
    def add_block(block):
        blocks.append(block)
        for idx in get_block_triple_indices(block):
            coverage_count[idx] += 1

    def remove_block(block):
        # Find index of block in blocks list
        # This is O(N). N is ~76. Fine.
        idx = blocks.index(block)
        blocks.pop(idx)
        for t_idx in get_block_triple_indices(block):
            coverage_count[t_idx] -= 1

    def is_valid():
        for c in coverage_count:
            if c == 0:
                return False
        return True

    # --- Phase 1: Greedy Construction ---
    # We will use a list of uncovered triple indices to pick anchors efficiently.
    # Initially all are uncovered.
    uncovered_indices = list(range(num_triples))
    
    while uncovered_indices:
        if time.time() - start_time > timeout:
            break
        
        # Pick a random uncovered triple
        # To avoid bias, pick randomly from the list
        pos = random.randint(0, len(uncovered_indices) - 1)
        anchor_idx = uncovered_indices[pos]
        anchor_triple = all_triples[anchor_idx]
        anchor_set = set(anchor_triple)
        available_points = [p for p in range(v) if p not in anchor_set]
        
        best_block = None
        best_gain = -1
        
        # Generate candidates
        # Optimization: If we have many candidates, we might want to sample?
        # C(21,4) = 5985. Checking gain for each is 35 ops. 5985*35 ~ 200k ops.
        # This is fast in Python (<< 1ms).
        
        for comb in itertools.combinations(available_points, k - t):
            candidate_block = tuple(sorted(anchor_triple + comb))
            
            # Calculate gain: how many currently uncovered triples does this cover?
            gain = 0
            for t_idx in get_block_triple_indices(candidate_block):
                if coverage_count[t_idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
                
        if best_block is None:
            break
            
        # Add best block
        add_block(list(best_block))
        
        # Update uncovered_indices
        # We need to remove any triple that is now covered (count > 0)
        # Since we only added one block, only triples in this block changed from 0 to 1 (or higher).
        # We can filter the list.
        new_uncovered = []
        for idx in uncovered_indices:
            if coverage_count[idx] == 0:
                new_uncovered.append(idx)
        uncovered_indices = new_uncovered

    # If we timed out, we might have an invalid covering.
    if uncovered_indices:
        # Try to finish quickly with a simple greedy (no random, just first available)
        # Or just return invalid? The task says invalid scores 0.
        # Let's try to force completion if time permits.
        # For now, if we are here, we are likely close to timeout.
        # Let's do a fast fill.
        while uncovered_indices:
            if time.time() - start_time > timeout:
                break
            anchor_idx = uncovered_indices[0]
            anchor_triple = all_triples[anchor_idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            for comb in itertools.combinations(available_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + comb))
                gain = 0
                for t_idx in get_block_triple_indices(candidate_block):
                    if coverage_count[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block:
                add_block(list(best_block))
                # Update uncovered
                new_uncovered = []
                for idx in uncovered_indices:
                    if coverage_count[idx] == 0:
                        new_uncovered.append(idx)
                uncovered_indices = new_uncovered
            else:
                break

    if uncovered_indices:
        return blocks # Invalid

    # --- Phase 2: Local Search ---
    best_size = len(blocks)
    
    # Helper to check if we can remove a block
    def try_remove_block(block_idx):
        """
        Attempts to remove blocks[block_idx].
        Returns True if removed, False otherwise.
        Modifies blocks and coverage_count.
        """
        block = blocks[block_idx]
        # Temporarily remove
        for t_idx in get_block_triple_indices(block):
            coverage_count[t_idx] -= 1
        
        # Check if any triple became uncovered
        lost_triples = []
        for t_idx in get_block_triple_indices(block):
            if coverage_count[t_idx] == 0:
                lost_triples.append(t_idx)
        
        if not lost_triples:
            # Safe to remove
            blocks.pop(block_idx)
            return True
        
        # If we lost triples, we need to restore them.
        # Option 1: Revert removal.
        # Option 2: Find a new block that covers all lost_triples.
        # If we find such a block B, we can add B.
        # Then we have the same number of blocks.
        # But maybe B allows removing *another* block?
        # For now, let's just revert if we can't find a direct replacement that leads to a reduction.
        # Actually, if we add B, we are at size N. We can then try to remove *another* block.
        # This is a 2-step move.
        
        # Let's try to find a block B that covers all lost_triples.
        # To find B, we can use a greedy heuristic on the lost_triples.
        # Pick the first lost triple, find a block covering it, check if it covers the rest.
        
        # This is complex. Let's stick to simple removal first.
        # Revert
        for t_idx in get_block_triple_indices(block):
            coverage_count[t_idx] += 1
        return False

    # Simple Redundancy Removal Loop
    changed = True
    while changed:
        changed = False
        # Shuffle indices to avoid bias
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        for i in indices:
            if time.time() - start_time > timeout:
                return blocks
            if try_remove_block(i):
                changed = True
                best_size = len(blocks)
                break # Restart loop as indices changed

    # VNS: Perturb and Re-greedy (Simplified)
    # We will try to remove random sets of blocks and re-greedy.
    # But we will use the coverage_count state.
    
    best_blocks = list(blocks)
    
    # Limit iterations
    max_iterations = 100
    
    for _ in range(max_iterations):
        if time.time() - start_time > timeout:
            break
            
        # Save state
        current_blocks = list(blocks)
        current_counts = list(coverage_count)
        
        # Perturb: Remove random blocks
        num_remove = random.randint(1, 3)
        if len(blocks) <= num_remove:
            continue
            
        indices_to_remove = random.sample(range(len(blocks)), num_remove)
        
        # Perform removals
        for idx in sorted(indices_to_remove, reverse=True):
            block = blocks[idx]
            for t_idx in get_block_triple_indices(block):
                coverage_count[t_idx] -= 1
            blocks.pop(idx)
            
        # Check if still valid
        if is_valid():
            # We reduced size!
            if len(blocks) < len(best_blocks):
                best_blocks = list(blocks)
            # Continue search from this new state?
            # Yes, but we should probably do a redundancy removal pass first.
            # For now, just accept and continue loop.
            # We don't revert, we keep the new state.
            continue
        
        # If invalid, we need to re-greedy fill
        # Find uncovered triples
        uncovered = [i for i, c in enumerate(coverage_count) if c == 0]
        
        # Re-greedy
        while uncovered:
            if time.time() - start_time > timeout:
                break
            # Pick random uncovered
            pos = random.randint(0, len(uncovered) - 1)
            anchor_idx = uncovered[pos]
            anchor_triple = all_triples[anchor_idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            for comb in itertools.combinations(available_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + comb))
                gain = 0
                for t_idx in get_block_triple_indices(candidate_block):
                    if coverage_count[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block:
                add_block(list(best_block))
                # Update uncovered
                new_uncovered = []
                for idx in uncovered:
                    if coverage_count[idx] == 0:
                        new_uncovered.append(idx)
                uncovered = new_uncovered
            else:
                break
        
        # Check if valid now
        if is_valid():
            if len(blocks) < len(best_blocks):
                best_blocks = list(blocks)
            # Do a quick redundancy removal
            changed = True
            while changed:
                changed = False
                indices = list(range(len(blocks)))
                random.shuffle(indices)
                for i in indices:
                    if try_remove_block(i):
                        changed = True
                        break
        
        # If we didn't improve, revert?
        # The current logic modifies `blocks` and `coverage_count` in place.
        # If we didn't improve, we should probably revert to the best known state?
        # Or just keep searching?
        # VNS usually reverts if no improvement.
        # But here, we might have found a valid covering of the same size.
        # Let's revert to `best_blocks` if we didn't improve size.
        if len(blocks) >= len(best_blocks):
            # Revert
            blocks.clear()
            coverage_count.clear()
            coverage_count.extend(current_counts)
            blocks.extend(current_blocks)
        else:
            # We improved. Update best.
            best_blocks = list(blocks)

    return best_blocks

# Test locally
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    # Verify
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")
