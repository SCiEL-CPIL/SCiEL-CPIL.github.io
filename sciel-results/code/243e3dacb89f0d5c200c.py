import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    # Parameters for C(19, 7, 4)
    # v=19, k=7, t=4
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for overhead/return
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    N_t = len(all_t_subs)
    
    # Pre-compute for each k-subset (block), the list of t-subset indices it covers
    # There are C(19, 7) = 50388 possible blocks
    # We don't need to pre-compute all, but it helps for candidate generation
    # However, generating candidates on the fly is fine.
    
    # Greedy Initialization
    uncovered = [1] * N_t
    blocks = []
    block_to_tsubs = {} # block tuple -> list of t-sub indices
    
    # Helper to add a block to the state
    def add_block_to_state(block_tuple, uncovered_state, blocks_list, block_to_tsubs_dict):
        t_subs = [t_sub_to_idx[ts] for ts in itertools.combinations(block_tuple, t)]
        for idx in t_subs:
            if uncovered_state[idx] > 0:
                uncovered_state[idx] -= 1
        blocks_list.append(block_tuple)
        block_to_tsubs_dict[block_tuple] = t_subs
        return t_subs

    # Helper to remove a block from the state
    def remove_block_from_state(block_tuple, uncovered_state, blocks_list, block_to_tsubs_dict):
        t_subs = block_to_tsubs_dict[block_tuple]
        for idx in t_subs:
            if uncovered_state[idx] < 0: # Should not happen if valid
                pass
            uncovered_state[idx] += 1
        blocks_list.remove(block_tuple)
        del block_to_tsubs_dict[block_tuple]

    # Greedy Loop
    while any(uncovered):
        if time.time() > deadline:
            break
        best_block = None
        best_gain = -1
        # Randomize order of points to break ties and allow different runs?
        # For determinism in this single run, we just iterate.
        # To get a good start, we pick the block covering the most uncovered subsets.
        # Optimization: Pick a random uncovered subset, then find best block containing it?
        # This is faster than checking all 50k blocks.
        
        # Pick a random uncovered t-subset
        unc_idx = [i for i, c in enumerate(uncovered) if c > 0]
        if not unc_idx:
            break
        target_idx = random.choice(unc_idx)
        target_sub = all_t_subs[target_idx]
        
        # Find all blocks containing target_sub
        # A block contains target_sub if it is a superset of target_sub.
        # Points in target_sub: target_sub
        # Points not in target_sub: rest
        # We need to choose k - t points from the rest.
        rest_points = [p for p in range(v) if p not in target_sub]
        num_to_pick = k - t
        
        # Iterate through combinations of rest_points
        # To make it faster, we can limit the number of candidates or use a heuristic.
        # For now, iterate all, but this is C(15, 3) = 455 candidates.
        # 455 * 35 (t-subs in block) = ~16k ops.
        # This is fast.
        
        for cand_extra in itertools.combinations(rest_points, num_to_pick):
            cand_block = tuple(sorted(target_sub + cand_extra))
            
            # Calculate gain
            gain = 0
            for ts in itertools.combinations(cand_block, t):
                idx = t_sub_to_idx[ts]
                if uncovered[idx] > 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = cand_block
                if gain == 0:
                    break # No point in checking more if gain is 0? 
                    # Actually, we want max gain. If max possible is small, we stop.
                    # But gain can be up to 35.
        
        if best_block is None:
            break
            
        add_block_to_state(best_block, uncovered, blocks, block_to_tsubs)
        
        # Early termination if all covered
        if not any(uncovered):
            break

    # Convert uncovered array to a set of indices for local search?
    # For local search, we need count_4 and unique_count.
    # Let's rebuild the state for local search.
    
    count_4 = [0] * N_t
    unique_count = {} # block tuple -> count of unique t-subs
    
    for b in blocks:
        unique_count[b] = 0
        for ts in itertools.combinations(b, t):
            idx = t_sub_to_idx[ts]
            count_4[idx] += 1
            
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = t_sub_to_idx[ts]
            if count_4[idx] == 1:
                unique_count[b] += 1
                
    # Local Search
    # We want to minimize len(blocks)
    
    # Helper: get t-sub indices for a block
    def get_t_subs(block):
        return [t_sub_to_idx[ts] for ts in itertools.combinations(block, t)]
        
    # Helper: remove block and update counts
    def do_remove(block):
        t_subs = get_t_subs(block)
        for idx in t_subs:
            old_count = count_4[idx]
            count_4[idx] -= 1
            new_count = count_4[idx]
            
            if old_count == 1 and new_count == 0:
                # This should not happen if block was redundant
                pass
            elif old_count == 2 and new_count == 1:
                # The other block covering this subset is now unique
                # Find the other block
                # This is expensive. 
                # Alternative: maintain a list of blocks covering each t-sub.
                # Let's maintain `t_sub_to_blocks`
                pass
            
        del unique_count[block]
        blocks.remove(block)
        
    # The above remove logic is flawed because we don't know which other block becomes unique.
    # We need `t_sub_to_blocks`.
    
    t_sub_to_blocks = defaultdict(list)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = t_sub_to_idx[ts]
            t_sub_to_blocks[idx].append(b)
            
    # Re-verify unique_count with t_sub_to_blocks
    for b in list(unique_count.keys()):
        unique_count[b] = 0
        for ts in itertools.combinations(b, t):
            idx = t_sub_to_idx[ts]
            if len(t_sub_to_blocks[idx]) == 1:
                unique_count[b] += 1

    def remove_block_safe(block):
        # Assumes block is redundant
        t_subs = get_t_subs(block)
        for idx in t_subs:
            old_count = len(t_sub_to_blocks[idx])
            t_sub_to_blocks[idx].remove(block)
            new_count = len(t_sub_to_blocks[idx])
            
            if old_count == 2 and new_count == 1:
                # The remaining block is now unique for this subset
                remaining_block = t_sub_to_blocks[idx][0]
                unique_count[remaining_block] += 1
            # If old_count > 2, no change in uniqueness for others.
            # If old_count == 1, error (block wasn't redundant).
        
        del unique_count[block]
        blocks.remove(block)

    def add_block_safe(block):
        t_subs = get_t_subs(block)
        for idx in t_subs:
            old_count = len(t_sub_to_blocks[idx])
            t_sub_to_blocks[idx].append(block)
            new_count = len(t_sub_to_blocks[idx])
            
            if old_count == 1 and new_count == 2:
                # The previous unique block loses uniqueness
                prev_block = t_sub_to_blocks[idx][0]
                # Wait, we just appended. So index 0 is the old one?
                # Yes, if we append to end.
                unique_count[prev_block] -= 1
                
        unique_count[block] = 0
        # Calculate initial unique count for new block
        for idx in t_subs:
            if len(t_sub_to_blocks[idx]) == 1:
                unique_count[block] += 1

    # Initial cleanup
    redundant = [b for b in blocks if unique_count[b] == 0]
    for b in redundant:
        remove_block_safe(b)

    # Local Search Loop
    last_improvement = time.time()
    
    while time.time() < deadline:
        # Check for redundant blocks
        redundant = [b for b in blocks if unique_count[b] == 0]
        if redundant:
            # Remove all redundant
            for b in redundant:
                remove_block_safe(b)
            last_improvement = time.time()
            continue
            
        # No redundant blocks. Try a swap.
        # Pick a random block
        if not blocks:
            break
        b_i = random.choice(blocks)
        
        # Find unique t-subs of b_i
        u_i = []
        for ts in itertools.combinations(b_i, t):
            idx = t_sub_to_idx[ts]
            if len(t_sub_to_blocks[idx]) == 1:
                u_i.append(idx)
                
        if not u_i:
            # Should have been caught by redundant check
            continue
            
        # Union of points in u_i
        p_union = set()
        for idx in u_i:
            p_union.update(all_t_subs[idx])
            
        if len(p_union) > k:
            continue
            
        # Find candidate b_j containing p_union
        # p_union is a set of points.
        # We need to add k - len(p_union) points from the rest.
        rest_points = [p for p in range(v) if p not in p_union]
        num_to_pick = k - len(p_union)
        
        # To speed up, limit candidates?
        # C(19-|P|, 7-|P|)
        # If |P|=4, C(15,3)=455.
        # If |P|=7, C(12,0)=1.
        
        best_b_j = None
        best_score = -1
        
        # Heuristic: Score b_j by how many "critical" subsets it covers
        # Critical subset: count==1, unique to some block B_k != b_i
        # If b_j covers such a subset, B_k loses a unique subset, becoming closer to redundant.
        
        candidates = list(itertools.combinations(rest_points, num_to_pick))
        random.shuffle(candidates)
        
        # Limit candidates to avoid long stalls
        max_candidates = 50
        for cand_extra in candidates[:max_candidates]:
            cand_block = tuple(sorted(tuple(p_union) + cand_extra))
            
            # Check if cand_block is already in blocks
            if cand_block in blocks:
                continue
                
            # Calculate score
            score = 0
            for ts in itertools.combinations(cand_block, t):
                idx = t_sub_to_idx[ts]
                if len(t_sub_to_blocks[idx]) == 1:
                    # Check if the unique block is not b_i
                    if t_sub_to_blocks[idx][0] != b_i:
                        score += 1
                        
            if score > best_score:
                best_score = score
                best_b_j = cand_block
                
        if best_b_j is None:
            continue
            
        # Perform swap
        # 1. Remove b_i
        remove_block_safe(b_i)
        # 2. Add b_j
        add_block_safe(best_b_j)
        
        # Check for new redundant blocks
        redundant = [b for b in blocks if unique_count[b] == 0]
        if redundant:
            for b in redundant:
                remove_block_safe(b)
            last_improvement = time.time()
        else:
            # No improvement. 
            # To escape local minima, we might accept this swap with some probability.
            # Or revert.
            # Let's revert to avoid cycling, but try a different block next time.
            # Actually, if we revert, we are stuck.
            # Let's accept the swap with probability 0.1 (annealing-like).
            if random.random() > 0.1:
                # Revert
                remove_block_safe(best_b_j)
                add_block_safe(b_i)
            else:
                # Keep the swap (exploration)
                pass
                
    return [list(b) for b in blocks]
