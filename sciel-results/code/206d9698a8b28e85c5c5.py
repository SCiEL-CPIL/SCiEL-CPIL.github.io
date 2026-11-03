import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Combinatorial Seeding + Simulated Annealing.
    
    Strategy:
    1. Seed with disjoint blocks to maximize initial coverage efficiency.
    2. Fill remaining coverage with greedy blocks.
    3. Optimize using Simulated Annealing (SA) with swap and delete moves.
       - Objective: Minimize number of blocks.
       - Constraint: All t-subsets covered.
       - Moves: 
         - Delete: Remove a block (if it doesn't break coverage).
         - Swap: Replace a block with a new one that covers more 'critical' triples.
    """
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    triple_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # State
    covered = np.zeros(n_t_subsets, dtype=bool)
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    blocks = []
    block_set = set()
    
    # Helper: Update state for adding/removing a block
    def add_block(block):
        for sub in itertools.combinations(block, t):
            idx = triple_to_idx[sub]
            covered[idx] = True
            coverage_count[idx] += 1
        blocks.append(block)
        block_set.add(tuple(block))
        
    def remove_block(block):
        for sub in itertools.combinations(block, t):
            idx = triple_to_idx[sub]
            coverage_count[idx] -= 1
            if coverage_count[idx] == 0:
                covered[idx] = False
        blocks.remove(block)
        block_set.remove(tuple(block))

    def get_uncovered():
        return np.where(~covered)[0]

    # Phase 1: Seeding
    # Try to pack points into disjoint blocks to create a strong base
    remaining_points = list(range(v))
    rng.shuffle(remaining_points)
    
    # Greedy disjoint packing
    while len(remaining_points) >= k:
        # Pick a random block from remaining
        # To be deterministic-ish but effective, just take the first k
        # But let's try to maximize spread. Actually, any disjoint set is fine for seeding.
        block = tuple(sorted(remaining_points[:k]))
        remaining_points = remaining_points[k:]
        add_block(block)
        
    # Phase 2: Greedy Fill
    # Add blocks to cover all remaining triples
    max_greedy_steps = 200
    for _ in range(max_greedy_steps):
        if time.time() > deadline - 30.0: break
        unc = get_uncovered()
        if len(unc) == 0: break
        
        # Pick a random uncovered triple to anchor
        anchor_idx = rng.choice(unc)
        anchor_triple = all_t_subsets[anchor_idx]
        
        # Generate a few candidate blocks containing the anchor
        best_cand = None
        best_gain = -1
        
        for _ in range(10):
            block_list = list(anchor_triple)
            block_set_local = set(block_list)
            candidates = [x for x in range(v) if x not in block_set_local]
            
            # Greedy fill
            while len(block_list) < k:
                if not candidates: break
                best_p = None
                best_p_gain = -1
                # Calculate gain for each candidate point
                # We want to cover uncovered triples
                # Triples formed by adding p to block_list
                # A triple is (p, a, b) where a,b in block_list
                # We check if (p,a,b) is uncovered
                subs = list(itertools.combinations(block_list, t-1))
                for p in candidates:
                    gain = 0
                    for sub in subs:
                        tri = tuple(sorted(sub + (p,)))
                        idx = triple_to_idx[tri]
                        if not covered[idx]:
                            gain += 1
                    if gain > best_p_gain:
                        best_p_gain = gain
                        best_p = p
                if best_p is not None:
                    block_list.append(best_p)
                    block_set_local.add(best_p)
                    candidates.remove(best_p)
                else:
                    # Random fallback
                    p = rng.choice(candidates)
                    block_list.append(p)
                    block_set_local.add(p)
                    candidates.remove(p)
            
            cand = tuple(sorted(block_list))
            if cand in block_set:
                continue
                
            # Calculate total gain of this block
            gain = 0
            for sub in itertools.combinations(cand, t):
                idx = triple_to_idx[sub]
                if not covered[idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        
        if best_cand is not None:
            add_block(best_cand)
        else:
            # Fallback if no good candidate found (shouldn't happen often)
            # Just pick a random block
            block = tuple(sorted(rng.sample(range(v), k)))
            if block not in block_set:
                add_block(block)
            else:
                break

    # Check if covered
    if not covered.all():
        # Force cover with random blocks if needed (rare)
        while not covered.all() and time.time() < deadline - 5.0:
            unc = get_uncovered()
            if len(unc) == 0: break
            anchor = all_t_subsets[rng.choice(unc)]
            block = tuple(sorted(list(anchor) + rng.sample([x for x in range(v) if x not in anchor], k-t)))
            if block not in block_set:
                add_block(block)
            else:
                # Try another random
                block = tuple(sorted(rng.sample(range(v), k)))
                if block not in block_set:
                    add_block(block)
                else:
                    break

    # Phase 3: Simulated Annealing Optimization
    # We want to minimize len(blocks)
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    best_blocks = blocks[:]
    best_len = len(blocks)
    
    current_len = len(blocks)
    
    # Precompute all possible blocks? No, C(24,7) is 346k. Too many to store all gains.
    # But we can generate candidates on the fly.
    
    # For SA, we need to evaluate "energy" or cost.
    # Cost = number of blocks.
    # But we must maintain validity.
    # Moves:
    # 1. Delete a block i. Valid if no triple in block i has coverage_count == 1.
    # 2. Swap block i with block j. Valid if coverage remains complete.
    #    This is expensive to check.
    
    # Alternative SA approach:
    # Keep a list of blocks.
    # Try to remove a block. If invalid, try to "fix" it by adding a new block that covers the lost triples?
    # That's just greedy again.
    
    # Better SA:
    # State: List of blocks.
    # Move: Pick a random block to remove. If it causes uncovered triples, generate a new block to cover them.
    # If the new block is "better" (covers more other uncovered triples or is smaller? No, size is fixed k), 
    # actually we just want to reduce count.
    # So: 
    # 1. Pick random block B to remove.
    # 2. Calculate triples that would become uncovered (those with count 1 in B).
    # 3. If none, remove B. Success! Len decreases.
    # 4. If some, generate a new block B' that covers all those critical triples.
    #    - If B' exists and B' != B, swap B with B'. Len stays same.
    #    - We accept swap if it helps future deletions? Hard to measure.
    #    - Instead, use SA on the "coverage redundancy".
    
    # Let's stick to a simpler heuristic that is effective:
    # Iterative Deletion with Repair.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle order of blocks to avoid bias
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5.0: break
            
            if i >= len(blocks): continue
            b = blocks[i]
            
            # Check if b is redundant
            is_redundant = True
            for sub in itertools.combinations(b, t):
                if coverage_count[triple_to_idx[sub]] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                remove_block(b)
                improved = True
                # Restart scan as indices are invalid
                break
        
        # If no deletion happened, try swaps to improve structure for future deletions
        if not improved:
            # Try to swap a block with a better one
            # Pick a random block
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                b_old = blocks[i]
                
                # Identify "critical" triples in b_old (count == 1)
                criticals = []
                for sub in itertools.combinations(b_old, t):
                    if coverage_count[triple_to_idx[sub]] == 1:
                        criticals.append(sub)
                
                if criticals:
                    # We need a new block that covers all criticals
                    # This is a constraint satisfaction problem
                    # Greedy: Start with union of criticals? No, they might not fit in one block.
                    # Actually, if criticals don't fit in one block, we can't swap directly.
                    # We can only swap if there exists a block covering all criticals.
                    
                    # Check if all criticals are contained in some k-set
                    # Since criticals are subsets of b_old, and b_old is size k, 
                    # the only block containing all of them is b_old itself (or supersets, but size is fixed).
                    # Wait, if a triple has count 1, it is ONLY in b_old.
                    # So any new block must contain that triple.
                    # If we have multiple criticals, the new block must contain all of them.
                    # Since they are all in b_old, the new block must contain their union.
                    # If union size > k, impossible.
                    # If union size <= k, we can try to fill the rest.
                    
                    union_set = set()
                    for sub in criticals:
                        union_set.update(sub)
                    
                    if len(union_set) <= k:
                        # Try to find a new block containing union_set
                        # Remove b_old temporarily? No, just generate candidates.
                        # We want a block B' != B_old such that B' covers all criticals.
                        # And we hope B' allows future deletions.
                        
                        # Generate a random block containing union_set
                        remaining = [x for x in range(v) if x not in union_set]
                        fill = rng.sample(remaining, k - len(union_set))
                        b_new = tuple(sorted(list(union_set) + fill))
                        
                        if b_new != b_old and b_new not in block_set:
                            # Check if this swap is valid (it is, by construction it covers criticals)
                            # But does it help?
                            # Heuristic: Accept if b_new covers more "low coverage" triples than b_old
                            # Or just accept randomly with some probability to explore.
                            
                            # Let's calculate a "score" for the block
                            # Score = sum of 1/coverage_count for triples in block
                            # Higher score means it covers rarer triples, which is good for redundancy?
                            # Actually, we want to increase redundancy of triples in b_old so we can delete it.
                            # So we want b_new to cover triples that are currently well-covered? No.
                            # We want to cover the criticals.
                            # The benefit of the swap is that it might allow b_old to be deleted later?
                            # No, b_old is being replaced.
                            
                            # Actually, if we swap b_old with b_new, the count of criticals increases to 2 (in b_new).
                            # So they are no longer critical.
                            # This makes the system more robust.
                            # Let's just do it if b_new is distinct.
                            
                            # Remove b_old
                            remove_block(b_old)
                            # Add b_new
                            add_block(b_new)
                            improved = True
                            break # Restart scan
    
    # Final cleanup: Remove any remaining redundant blocks
    # Rebuild coverage counts to be safe
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            coverage_count[triple_to_idx[sub]] += 1
    
    final_blocks = []
    for b in blocks:
        redundant = True
        for sub in itertools.combinations(b, t):
            if coverage_count[triple_to_idx[sub]] == 1:
                redundant = False
                break
        if not redundant:
            final_blocks.append(b)
            for sub in itertools.combinations(b, t):
                coverage_count[triple_to_idx[sub]] -= 1
        else:
            # It was redundant, don't add to final
            pass
            
    # Verify final_blocks covers everything
    # If not, we lost coverage. This shouldn't happen if logic is correct.
    # But let's check.
    # Re-verify
    test_cov = np.zeros(n_t_subsets, dtype=bool)
    for b in final_blocks:
        for sub in itertools.combinations(b, t):
            test_cov[triple_to_idx[sub]] = True
            
    if not test_cov.all():
        # Fallback to original blocks if final cleanup broke it (shouldn't happen)
        return blocks
    else:
        return final_blocks
