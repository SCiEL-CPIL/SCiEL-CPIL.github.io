import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    # n_triplets is ~2024, which fits easily in a Python integer
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            # Find block with max gain
            # We scan all blocks. 346k iterations * bit_count is fast.
            for b_idx in range(n_blocks):
                # If block is already used, gain is 0 (since uncovered bits are 0 there)
                # But we can just calculate gain directly.
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_gain == 0:
                # Should not happen if uncovered != 0, unless we ran out of time or logic error
                break
                
            # Randomly pick from candidates to diversify
            # If many candidates, picking one randomly helps avoid local minima
            chosen_idx = rng.choice(candidates)
            
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_blocks

    def refine_solution(blocks):
        # Convert to list of lists for mutability
        blocks = [list(b) for b in blocks]
        
        # Helper to calculate coverage counts
        def get_counts(blk_list):
            counts = np.zeros(n_triplets, dtype=np.int32)
            for b in blk_list:
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    counts[tid] += 1
            return counts

        # Helper to get exclusive triplets for a block given counts
        def get_exclusive(block, counts):
            exc = []
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    exc.append(tid)
            return exc

        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            
            # 1. Remove redundant blocks
            counts = get_counts(blocks)
            new_blocks = []
            for b in blocks:
                # Check if block is redundant
                redundant = True
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    changed = True
            blocks = new_blocks
            
            if not blocks:
                break

            # 2. Try merge 2-for-1
            counts = get_counts(blocks)
            # Precompute exclusive triplets for each block
            exclusives = []
            for b in blocks:
                exclusives.append(get_exclusive(b, counts))
                
            merged = False
            n_sol = len(blocks)
            
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                if not exclusives[i]:
                    continue
                    
                for j in range(i + 1, n_sol):
                    if time.time() > deadline - 5:
                        break
                    if not exclusives[j]:
                        continue
                    
                    # Union of exclusive triplets
                    # We need to check if there is a block that covers all these triplets.
                    # The points involved in these triplets must be contained in the new block.
                    points_needed = set()
                    # We can optimize by just collecting points from the triplets
                    # But we need to ensure the new block covers the triplets.
                    # If the new block contains all points of the triplets, it covers them.
                    
                    # Let's collect points from exclusive triplets of i and j
                    # Note: exclusives are triplet IDs. We need to map back to points.
                    # This is slightly expensive. Let's precompute triplet points?
                    # all_triplets[tid] is the tuple of points.
                    
                    for tid in exclusives[i]:
                        points_needed.update(all_triplets[tid])
                    for tid in exclusives[j]:
                        points_needed.update(all_triplets[tid])
                        
                    if len(points_needed) > k:
                        continue
                        
                    # If points_needed <= k, we can form a block.
                    # We need to find a block that contains points_needed.
                    # Actually, ANY block containing points_needed will cover the triplets in points_needed?
                    # Yes, because the triplets are subsets of points_needed.
                    # So we just need to check if such a block exists in our universe.
                    # Since we are allowed to add ANY block, we can just construct one.
                    # But we must ensure it's a valid k-subset.
                    
                    # Construct candidate block
                    cand_set = set(points_needed)
                    # Fill remaining spots with arbitrary points
                    # To be deterministic and simple, pick smallest available
                    for p in range(v):
                        if len(cand_set) == k:
                            break
                        if p not in cand_set:
                            cand_set.add(p)
                            
                    if len(cand_set) != k:
                        continue
                        
                    new_block = tuple(sorted(cand_set))
                    
                    # Check if this new block is already in the solution?
                    # If it is, we just remove i and j.
                    # If it's not, we add it and remove i and j.
                    
                    # We need to verify that new_block covers all triplets in exclusives[i] and exclusives[j].
                    # By construction, it contains the points, so it covers the triplets.
                    
                    # Perform the merge
                    # Remove i and j, add new_block
                    # Be careful with indices if we modify list in place.
                    # Let's build a new list.
                    
                    new_blocks_list = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks_list.append(b)
                    new_blocks_list.append(list(new_block))
                    
                    # Verify validity? The evaluator does this. 
                    # But we should ensure we didn't break coverage.
                    # The only triplets that were uniquely covered by i or j are in exclusives[i] and exclusives[j].
                    # new_block covers them.
                    # Triplets covered by both i and j (but not exclusively) are still covered by the other block (or others).
                    # So coverage is maintained.
                    
                    blocks = new_blocks_list
                    merged = True
                    changed = True
                    break
                if merged:
                    break
            if merged:
                continue

            # 3. Try swap: replace one block with another that covers more
            # This is a 1-for-1 move that might reduce total count if we can then remove another
            # Actually, let's try a different local search: try to find a block that can replace two blocks
            # We already did 2-for-1. Let's try 1-for-1 where the new block covers exclusive parts of the old
            # and allows removing another block.
            
            # For now, let's just do a few random restarts of the refinement if stuck
            if time.time() < deadline - 10:
                # Try a random perturbation: remove a random block and try to re-cover its exclusive parts
                # This is risky and complex. Let's stick to the 2-for-1 merge which is safe.
                pass

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    # Bitmask greedy is fast. We can afford more seeds.
    num_seeds = 30
    for seed in range(num_seeds):
        if time.time() > deadline - 40:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish
    best_blocks = refine_solution(best_blocks)
    
    # Try to refine other good solutions too if we have time
    if time.time() < deadline - 20:
        # Re-run greedy with different seeds and refine
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            blocks = greedy_set_cover(seed + 100)
            if len(blocks) < best_count + 2:  # Only refine if it's close to best
                refined = refine_solution(blocks)
                if len(refined) < best_count:
                    best_count = len(refined)
                    best_blocks = refined
    
    return best_blocks

if __name__ == "__main__":
    pass
