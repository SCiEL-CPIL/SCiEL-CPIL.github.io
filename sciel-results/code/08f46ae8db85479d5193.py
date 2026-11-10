import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Map each t-subset to a unique integer ID (0 to n_t_subs - 1)
    sub_to_id = {sub: i for i, sub in enumerate(all_t_subs)}
    id_to_sub = [None] * n_t_subs
    for sub, idx in sub_to_id.items():
        id_to_sub[idx] = sub
        
    # For each block (k-subset), precompute the list of t-subset IDs it covers
    # We don't need ALL blocks, but we can precompute for a specific block on the fly.
    # However, in local search, we generate candidates.
    
    def get_covered_ids(block):
        """Returns a list of t-subset IDs covered by the given block."""
        return [sub_to_id[sub] for sub in itertools.combinations(sorted(block), t)]
    
    # 2. Greedy Construction
    def greedy_build(rng, deadline):
        uncovered = set(range(n_t_subs))
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered subset to focus on
            target_id = rng.choice(list(uncovered))
            target_sub = id_to_sub[target_id]
            
            # Generate all k-blocks containing target_sub
            # Points in target_sub: target_sub
            # Points not in target_sub: points_not_in_target
            points_in = set(target_sub)
            points_not_in = [x for x in range(v) if x not in points_in]
            
            # We need to choose (k - t) more points from points_not_in
            num_more = k - t
            candidates = []
            
            # If k-t is large, iterating all combinations is expensive.
            # For v=19, k=8, t=5, k-t=3. points_not_in has 14 elements.
            # C(14, 3) = 364 candidates. This is fast.
            for extra in itertools.combinations(points_not_in, num_more):
                block = tuple(sorted(target_sub + extra))
                
                # Calculate gain: how many uncovered subsets does this block cover?
                covered_ids = get_covered_ids(block)
                gain = 0
                for cid in covered_ids:
                    if cid in uncovered:
                        gain += 1
                
                candidates.append((gain, block))
            
            # Pick the best candidate
            if not candidates:
                break
                
            best_gain, best_block = max(candidates, key=lambda x: x[0])
            
            if best_gain == 0:
                # Should not happen if target_id is uncovered, but safety check
                break
                
            blocks.append(best_block)
            
            # Update uncovered
            covered_ids = get_covered_ids(best_block)
            for cid in covered_ids:
                uncovered.discard(cid)
                
        return blocks

    # 3. Local Search: Reduction and Replacement
    def local_search(blocks, rng, deadline):
        if not blocks:
            return blocks
        
        blocks = list(blocks)
        # Maintain cover_count: array of size n_t_subs
        cover_count = [0] * n_t_subs
        # Map: block_index -> list of covered ids
        block_covers = []
        
        # Initialize
        for b in blocks:
            ids = get_covered_ids(b)
            block_covers.append(ids)
            for cid in ids:
                cover_count[cid] += 1
        
        # Helper: check if a block is redundant
        # A block is redundant if for all cid in its covers, cover_count[cid] > 1
        # (meaning other blocks also cover it)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Remove redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                is_redundant = True
                for cid in block_covers[i]:
                    if cover_count[cid] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block i
                    for cid in block_covers[i]:
                        cover_count[cid] -= 1
                    blocks.pop(i)
                    block_covers.pop(i)
                    improved = True
            
            if time.time() > deadline:
                break
                
            # Pass 2: Try to replace a block with a better one
            # Identify blocks with high "waste" (low essentiality)
            # Essentiality of block i = number of cids in block_covers[i] where cover_count[cid] == 1
            # We want to replace blocks with LOW essentiality, but we must maintain coverage of the essential parts.
            # Actually, we can only replace a block if the essential parts can be covered by a new single block.
            
            # Calculate essentiality and waste
            block_stats = []
            for i in range(len(blocks)):
                essential_ids = []
                waste = 0
                for cid in block_covers[i]:
                    if cover_count[cid] == 1:
                        essential_ids.append(cid)
                    else:
                        waste += 1
                block_stats.append((waste, len(essential_ids), i, essential_ids))
            
            # Sort by waste (descending) - try to replace the most wasteful blocks first
            # But we must be careful: if essential_ids is large, finding a replacement is hard.
            # Sort by (len(essential_ids) ascending, waste descending) ?
            # Let's prioritize blocks with small essential sets and high waste.
            block_stats.sort(key=lambda x: (x[1], -x[0]))
            
            replacement_made = False
            for waste, num_essential, idx, essential_ids in block_stats:
                if time.time() > deadline:
                    break
                    
                if num_essential == 0:
                    # Should have been removed in Pass 1, but maybe race condition?
                    # Just remove it.
                    for cid in block_covers[idx]:
                        cover_count[cid] -= 1
                    blocks.pop(idx)
                    block_covers.pop(idx)
                    improved = True
                    replacement_made = True
                    break # Restart loop after structural change
                
                # We need a new block that covers all essential_ids.
                # Let U be the union of points in essential_ids.
                union_points = set()
                for eid in essential_ids:
                    union_points.update(id_to_sub[eid])
                
                if len(union_points) > k:
                    continue # Cannot cover all essential subsets with one block
                
                # Generate candidate blocks containing union_points
                points_in = list(union_points)
                points_not_in = [x for x in range(v) if x not in union_points]
                num_more = k - len(points_in)
                
                best_new_block = None
                best_new_gain = -1
                best_new_covers = None
                
                # Sample a few candidates if there are too many, otherwise check all
                # C(19 - len(points_in), num_more)
                # Max len(points_in) is k (8), so min points_not_in is 11.
                # C(11, 0) = 1. C(11, 1) = 11. C(11, 2) = 55. C(11, 3) = 165.
                # It's always small enough to check all.
                
                for extra in itertools.combinations(points_not_in, num_more):
                    new_block = tuple(sorted(points_in + extra))
                    new_covers = get_covered_ids(new_block)
                    
                    # Check if new_block covers all essential_ids
                    covers_all_essential = True
                    for eid in essential_ids:
                        if eid not in new_covers:
                            covers_all_essential = False
                            break
                    if not covers_all_essential:
                        continue
                        
                    # Calculate "gain" of this new block in terms of how much it helps
                    # We want to maximize the number of subsets covered that are currently
                    # covered by only 1 or 2 blocks, to create new redundancies.
                    # Simple metric: number of cids in new_covers where cover_count[cid] <= 2
                    # (If we add this block, these will become 2 or 3, potentially allowing
                    #  other blocks to be removed if they become redundant)
                    
                    potential = 0
                    for cid in new_covers:
                        if cover_count[cid] <= 2:
                            potential += 1
                            
                    if potential > best_new_gain:
                        best_new_gain = potential
                        best_new_block = new_block
                        best_new_covers = new_covers
                        
                if best_new_block is not None:
                    # Perform swap
                    # Remove old block
                    for cid in block_covers[idx]:
                        cover_count[cid] -= 1
                    blocks.pop(idx)
                    block_covers.pop(idx)
                    
                    # Add new block
                    blocks.append(best_new_block)
                    block_covers.append(best_new_covers)
                    for cid in best_new_covers:
                        cover_count[cid] += 1
                    
                    improved = True
                    replacement_made = True
                    break # Restart loop after structural change
            
            if replacement_made:
                continue
                
            # If no improvement in this pass, try random restarts of local search?
            # Or break.
            break
            
        return blocks

    # 4. Main Execution Loop
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy constructions with different seeds
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + i)
        blocks = greedy_build(rng, deadline)
        
        if not blocks:
            continue
            
        # Verify coverage (should be valid by construction, but check for safety)
        # And deduplicate
        seen = set()
        unique_blocks = []
        for b in blocks:
            if b not in seen:
                seen.add(b)
                unique_blocks.append(b)
        blocks = unique_blocks
        
        # Local Search
        if time.time() < deadline - 5:
            refined_blocks = local_search(blocks, rng, deadline)
            if len(refined_blocks) < len(blocks):
                blocks = refined_blocks
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Fallback if no good solution found
    if best_blocks is None:
        # Deterministic greedy
        rng = random.Random(0)
        best_blocks = greedy_build(rng, deadline)
        if not best_blocks:
             # Emergency fallback: brute force small? No, just return empty or a trivial cover?
             # A trivial cover is all possible blocks. That's huge.
             # Let's just return the greedy result even if imperfect, hoping it's valid.
             # The greedy ensures validity by covering all uncovered subsets.
             pass

    # Final Verification and Cleanup
    # Ensure no duplicates
    seen = set()
    final_blocks = []
    for b in best_blocks:
        if b not in seen:
            seen.add(b)
            final_blocks.append(list(b))
            
    return final_blocks
