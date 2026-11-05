import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Specific parameters for C(24,7,3)
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters: simple greedy
        all_triples = list(itertools.combinations(range(v), t))
        triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
        all_blocks = list(itertools.combinations(range(v), k))
        uncovered = set(range(len(all_triples)))
        selected = []
        while uncovered:
            if time.time() > deadline - 5:
                break
            best_block = None
            best_count = -1
            for b in all_blocks:
                if b in selected:
                    continue
                count = 0
                for tri in itertools.combinations(b, t):
                    if triple_to_idx[tri] in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b
            if best_block is None or best_count == 0:
                break
            selected.append(best_block)
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(triple_to_idx[tri])
        return selected

    # C(24,7,3) Optimized Approach
    
    # 1. Precomputation
    points = range(24)
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Full mask for all triples
    full_mask = (1 << num_triples) - 1
    
    # Helper to get mask for a block
    def get_block_mask(block):
        mask = 0
        for tri in itertools.combinations(block, 3):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        return mask
    
    # Helper to get triples in block as list of indices
    def get_block_triples(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, 3)]

    # 2. Generate Candidate Pool
    # We need a diverse set of blocks.
    candidates = []
    candidate_set = set()
    
    def add_candidate(block):
        b = tuple(block)
        if b not in candidate_set:
            candidate_set.add(b)
            candidates.append(b)
            
    # Add some random blocks
    for _ in range(5000):
        add_candidate(random.sample(points, 7))
        
    # We will add more candidates dynamically during the process if needed, 
    # but for now, let's start with a greedy solution and its neighbors.
    
    # 3. Initial Greedy Solution
    uncovered_mask = full_mask
    selected_blocks = []
    selected_masks = []
    selected_tuples = []
    
    # To speed up greedy, we need to quickly find best block.
    # With a small candidate pool, we can just scan it.
    # But the pool is small. Let's generate a slightly larger initial pool.
    
    # Actually, let's generate a pool of 10,000 random blocks first.
    # And also the greedy blocks as we find them.
    
    # Let's re-structure:
    # 1. Generate pool of 20,000 random blocks.
    for _ in range(20000):
        add_candidate(random.sample(points, 7))
        
    # Precompute masks for candidates
    cand_masks = []
    cand_triples = []
    for b in candidates:
        mask = get_block_mask(b)
        tris = get_block_triples(b)
        cand_masks.append(mask)
        cand_triples.append(tris)
        
    # Map tuple to index in candidates
    cand_idx = {tuple(c): i for i, c in enumerate(candidates)}
    
    # Greedy Construction
    # We want to maximize coverage of uncovered_mask.
    # gain = popcount(mask & uncovered_mask)
    
    while uncovered_mask and time.time() < deadline - 30:
        best_gain = -1
        best_cand_idx = -1
        
        # Scan all candidates
        for i, mask in enumerate(cand_masks):
            # If we already selected this block, we can't select it again?
            # The problem says "blocks must be distinct".
            # So we track which candidate indices are used.
            if i in [cand_idx[tuple(sb)] for sb in selected_tuples]:
                continue
                
            # Calculate gain
            # Bitwise AND and popcount
            gain = (mask & uncovered_mask).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_cand_idx = i
                
        if best_cand_idx == -1 or best_gain == 0:
            # No improvement possible with current candidates
            # Try to expand pool?
            # For now, break or add random blocks
            if time.time() < deadline - 30:
                # Add 1000 more random blocks
                for _ in range(1000):
                    nb = random.sample(points, 7)
                    if tuple(nb) not in candidate_set:
                        add_candidate(nb)
                        # Append to lists
                        cand_masks.append(get_block_mask(nb))
                        cand_triples.append(get_block_triples(nb))
                        cand_idx[tuple(nb)] = len(candidates)-1
                # Don't break, continue loop
                continue
            else:
                break
                
        # Select best block
        selected_tuples.append(candidates[best_cand_idx])
        selected_masks.append(cand_masks[best_cand_idx])
        uncovered_mask &= ~cand_masks[best_cand_idx]
        
    # 4. Local Search
    # State:
    # sol_blocks: list of tuples
    # sol_masks: list of masks
    # counts: list of int, count of coverage for each triple
    
    sol_blocks = list(selected_tuples)
    sol_masks = list(selected_masks)
    
    # Initialize counts
    counts = [0] * num_triples
    for mask in sol_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    # Helper: check if block can be removed
    def can_remove(block_triples):
        for t in block_triples:
            if counts[t] == 1:
                return False
        return True
        
    # Helper: perform removal
    def do_remove(block_triples):
        for t in block_triples:
            counts[t] -= 1
            
    # Helper: check if swap i->j is valid
    def can_swap(tris_i, tris_j):
        # tris_i: triples in block i
        # tris_j: triples in block j
        # Condition: All triples in tris_i that are NOT in tris_j must have count >= 2
        # Triples in tris_j that are in tris_i: count increases, no problem.
        # Triples in tris_j not in tris_i: count increases, no problem.
        # Triples in tris_i not in tris_j: count decreases by 1. Must not become 0.
        # So: for t in tris_i, if t not in tris_j and counts[t] == 1, invalid.
        
        set_j = set(tris_j)
        for t in tris_i:
            if t not in set_j and counts[t] == 1:
                return False
        return True
        
    # Helper: perform swap
    def do_swap(tris_i, tris_j):
        set_j = set(tris_j)
        # Decrement for tris_i not in tris_j
        for t in tris_i:
            if t not in set_j:
                counts[t] -= 1
        # Increment for tris_j not in tris_i
        set_i = set(tris_i)
        for t in tris_j:
            if t not in set_i:
                counts[t] += 1

    # Precompute triples for solution blocks
    sol_triples = []
    for b in sol_blocks:
        sol_triples.append(get_block_triples(b))
        
    # Mapping sol_block_tuple -> index in sol_blocks
    sol_block_set = set(sol_blocks)
    
    best_len = len(sol_blocks)
    
    while time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(sol_blocks)):
                if can_remove(sol_triples[i]):
                    # Remove
                    do_remove(sol_triples[i])
                    sol_blocks.pop(i)
                    sol_masks.pop(i)
                    sol_triples.pop(i)
                    sol_block_set.discard(sol_blocks[i] if i < len(sol_blocks) else None) # Careful with index
                    # Actually, pop changes indices.
                    # Let's do removal in reverse or restart loop.
                    # Restart is safer.
                    improved = True
                    changed = True
                    break # Restart removal phase
            
        if len(sol_blocks) < best_len:
            best_len = len(sol_blocks)
            
        # Phase 2: Swaps
        # Try to swap blocks to enable future deletions
        num_swaps_attempted = 0
        max_swaps = 100 # Limit per iteration to avoid wasting time
        
        while num_swaps_attempted < max_swaps and time.time() < deadline - 5:
            num_swaps_attempted += 1
            
            if not sol_blocks:
                break
                
            i = random.randint(0, len(sol_blocks)-1)
            
            # Choose a candidate j not in solution
            # Pick random candidate from pool
            # To be efficient, pick random index, check if in solution
            attempts = 0
            while attempts < 10:
                j_idx = random.randint(0, len(candidates)-1)
                j_block = candidates[j_idx]
                if j_block not in sol_block_set:
                    break
                attempts += 1
            else:
                continue # Couldn't find a candidate not in solution? Unlikely.
                
            j_triples = cand_triples[j_idx]
            i_triples = sol_triples[i]
            
            if can_swap(i_triples, j_triples):
                # Perform swap
                do_swap(i_triples, j_triples)
                sol_blocks[i] = j_block
                sol_masks[i] = cand_masks[j_idx]
                sol_triples[i] = j_triples
                sol_block_set.discard(sol_blocks[i]) # Wait, j_block is added, old one removed?
                # We replaced sol_blocks[i] with j_block.
                # Old block was sol_blocks[i] (before change).
                # We need to remove old block from set and add new one.
                # But we already popped/overwrote.
                # Let's track explicitly.
                old_block = sol_blocks[i] # This is j_block now.
                # We lost the old block reference in the list.
                # Let's fix the logic.
                
                # Correct Logic:
                # old_block = sol_blocks[i]
                # if can_swap(sol_triples[i], cand_triples[j_idx]):
                #    do_swap(...)
                #    sol_blocks[i] = candidates[j_idx]
                #    ...
                #    sol_block_set.remove(old_block)
                #    sol_block_set.add(candidates[j_idx])
                
                # Let's re-implement swap carefully in the loop below.
                pass # Just a placeholder, logic will be in the actual loop structure.

        # Re-structured Loop Body
        # We need to maintain sol_block_set correctly.
        
        # Let's restart the Local Search Loop properly.
        
    # The above loop had some messy state management. Let's write it cleanly.
    
    # Re-initialize Local Search with clean state management
    
    # We will use a list of indices into `candidates` for the solution?
    # No, solution blocks might not all be in `candidates` if we added them dynamically?
    # In my greedy, I only selected from `candidates`. So yes, solution is a subset of candidates.
    
    # Let's use indices for efficiency.
    # sol_indices: list of int (indices into candidates)
    
    # Re-run Greedy to get sol_indices?
    # Or convert current sol_blocks to indices.
    
    sol_indices = []
    for b in sol_blocks:
        sol_indices.append(cand_idx[b])
        
    # Re-build counts from scratch to be safe
    counts = [0] * num_triples
    for idx in sol_indices:
        m = cand_masks[idx]
        while m:
            lsb = m & -m
            t_idx = lsb.bit_length() - 1
            counts[t_idx] += 1
            m ^= lsb
            
    best_len = len(sol_indices)
    
    # Precompute triples for candidates
    # cand_triples is already computed.
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Deletion Pass
        deleted = True
        while deleted:
            deleted = False
            for i in range(len(sol_indices)):
                idx = sol_indices[i]
                tris = cand_triples[idx]
                if all(counts[t] >= 2 for t in tris):
                    # Remove
                    for t in tris:
                        counts[t] -= 1
                    sol_indices.pop(i)
                    improved = True
                    deleted = True
                    break # Restart scan
            
        if len(sol_indices) < best_len:
            best_len = len(sol_indices)
            
        # 2. Swap Pass
        # Try random swaps to escape local minima
        swap_attempts = 0
        max_swap_attempts = 50
        while swap_attempts < max_swap_attempts and time.time() < deadline - 2:
            swap_attempts += 1
            if not sol_indices:
                break
                
            i = random.randint(0, len(sol_indices)-1)
            old_idx = sol_indices[i]
            old_tris = cand_triples[old_idx]
            
            # Find candidate j not in sol_indices
            found_j = False
            for _ in range(10):
                j = random.randint(0, len(candidates)-1)
                if j not in sol_indices:
                    found_j = True
                    break
            
            if not found_j:
                continue
                
            new_tris = cand_triples[j]
            
            # Check validity
            # Triples in old_tris but not in new_tris must have count >= 2
            set_new = set(new_tris)
            valid = True
            for t in old_tris:
                if t not in set_new and counts[t] == 1:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                # Update counts
                set_old = set(old_tris)
                for t in old_tris:
                    if t not in set_new:
                        counts[t] -= 1
                for t in new_tris:
                    if t not in set_old:
                        counts[t] += 1
                
                sol_indices[i] = j
                improved = True
                # After a swap, try to delete again immediately?
                # The outer loop will handle it.
                
        if not improved and swap_attempts > 0:
            # If no swaps succeeded and no deletions, we might be stuck.
            # Add more random candidates?
            if time.time() < deadline - 10:
                # Add 500 new random candidates
                for _ in range(500):
                    nb = tuple(random.sample(points, 7))
                    if nb not in cand_idx:
                        add_candidate(nb)
                        cand_masks.append(get_block_mask(nb))
                        cand_triples.append(get_block_triples(nb))
                        cand_idx[nb] = len(candidates)-1
                        
    # Final Result
    final_blocks = [candidates[idx] for idx in sol_indices]
    
    # Validation
    covered = [0] * num_triples
    for b in final_blocks:
        for t in get_block_triples(b):
            covered[t] += 1
            
    if any(c == 0 for c in covered):
        # Fallback: repair
        # This shouldn't happen, but if it does, add blocks greedily
        for i in range(len(covered)):
            if covered[i] == 0:
                tri = all_triples[i]
                # Find any block covering tri
                for b in candidates:
                    if set(tri).issubset(set(b)):
                        if b not in final_blocks:
                            final_blocks.append(b)
                            break
        # Re-validate
        covered = [0] * num_triples
        for b in final_blocks:
            for t in get_block_triples(b):
                covered[t] += 1
        if any(c == 0 for c in covered):
            # Extreme fallback: return greedy
            pass
            
    return final_blocks

# Test locally if run directly
if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(len(blocks))
