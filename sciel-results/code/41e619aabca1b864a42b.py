import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer for overhead
    
    # 1. Precompute subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_to_idx = {tsub: i for i, tsub in enumerate(all_t_subs)}
    
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Precompute coverage: 
    # block_t_subs[i] = set of t-subset indices covered by k-subset i
    # t_to_ksubs[j] = list of k-subset indices covering t-subset j
    block_t_subs = [set() for _ in range(n_k_subs)]
    t_to_ksubs = [[] for _ in range(n_t_subs)]
    
    for i, ksub in enumerate(all_k_subs):
        for tsub in itertools.combinations(ksub, t):
            idx = t_sub_to_idx[tsub]
            block_t_subs[i].add(idx)
            t_to_ksubs[idx].append(i)
            
    # Convert to frozensets for faster operations if needed, but sets are fine
    # Actually, for fast "covers all deficit" check, we need intersection or subset check.
    # deficit is a set of indices. candidate is a set of indices.
    # deficit.issubset(candidate)
    
    # 2. Fast Greedy Initialization
    # cover_count[i] = number of blocks in solution covering t-subset i
    cover_count = [0] * n_t_subs
    current_blocks = []
    selected_set = set()
    
    # To speed up greedy, we don't scan all ksubs.
    # We maintain a set of uncovered t-subsets.
    # However, finding the best ksub for a specific uncovered tsub requires checking candidates.
    
    uncovered_set = set(range(n_t_subs))
    
    while uncovered_set:
        # Pick a random uncovered t-subset to anchor the selection
        # This introduces randomness and avoids getting stuck in deterministic local minima
        target_t_idx = random.choice(list(uncovered_set))
        
        # Get candidates: k-subsets covering target_t_idx
        candidates = t_to_ksubs[target_t_idx]
        
        best_block = -1
        best_gain = -1
        
        # Evaluate candidates
        # Gain = number of currently uncovered t-subsets covered by this block
        for b_idx in candidates:
            # Calculate gain
            # We can optimize this by precomputing the set of t-subsets for each block
            # gain = len(block_t_subs[b_idx] & uncovered_set)
            # Since uncovered_set is large, this intersection is costly.
            # Alternative: 
            # We just need to pick a "good" block. 
            # Let's just pick the first one? No, greedy needs some quality.
            # Let's sample a few candidates if the list is long.
            pass
            
        # For simplicity and speed in the initial phase, let's just pick a random candidate 
        # that covers the most uncovered items among a small sample.
        # Or just pick the first one? That might be too slow to converge.
        
        # Let's do a mini-greedy:
        # If candidates list is small, check all. If large, sample.
        sample_size = min(len(candidates), 20)
        sampled = random.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
        
        for b_idx in sampled:
            # Count how many of its covered t-subsets are in uncovered_set
            # Using set intersection
            gain = len(block_t_subs[b_idx] & uncovered_set)
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            # Should not happen if target_t_idx is uncovered, at least one candidate covers it
            # But maybe we didn't sample it?
            # Fallback: pick any candidate
            best_block = random.choice(candidates)
            
        # Add block
        current_blocks.append(best_block)
        selected_set.add(best_block)
        
        # Update cover_count and uncovered_set
        for t_idx in block_t_subs[best_block]:
            cover_count[t_idx] += 1
            if cover_count[t_idx] == 1:
                uncovered_set.discard(t_idx)
                
        if time.time() - start_time > time_limit * 0.2:
            # Stop early if taking too long, though greedy should be fast
            break

    # Ensure we have a valid cover. If not, fall back to full greedy?
    # The greedy above should produce a valid cover.
    
    # 3. Tabu Search / Local Search
    best_size = len(current_blocks)
    best_blocks = current_blocks[:]
    
    rng = random.Random(123)
    
    # Tabu data structure
    # tabu_until[block_idx] = iteration number until which it is tabu
    # We use a dictionary to store only active tabus to save memory/time
    tabu_dict = {}
    tabu_len = 50
    
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try Removal
        if current_blocks:
            # Pick a random block to try removing
            # Prioritize blocks that cover "redundant" parts?
            # Just random for now.
            idx_to_remove = rng.randrange(len(current_blocks))
            b_out = current_blocks[idx_to_remove]
            
            # Check if removal is valid
            # A block is redundant if for all t in block_t_subs[b_out], cover_count[t] > 1
            redundant = True
            for t_idx in block_t_subs[b_out]:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for t_idx in block_t_subs[b_out]:
                    cover_count[t_idx] -= 1
                current_blocks.pop(idx_to_remove)
                selected_set.discard(b_out)
                
                # Update best
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                    
                # Tabu the removed block? 
                # If we remove it, we might want to add it back later.
                # Tabu it for a while to avoid oscillation.
                tabu_dict[b_out] = iteration + tabu_len
                continue # Restart loop with new state
                
        # 2. Try Swap
        # Pick a block to swap out
        if not current_blocks:
            break
            
        idx_out = rng.randrange(len(current_blocks))
        b_out = current_blocks[idx_out]
        
        # Calculate deficit: t-subsets that would become uncovered if we remove b_out
        deficit = set()
        for t_idx in block_t_subs[b_out]:
            if cover_count[t_idx] == 1:
                deficit.add(t_idx)
                
        if not deficit:
            # This is a removal case, should have been caught above.
            # But maybe cover_count changed? No, we check immediately.
            # If deficit is empty, it means b_out is redundant.
            # Why wasn't it caught in removal phase? 
            # Maybe we picked a different idx? 
            # Let's just handle it here as a removal.
            for t_idx in block_t_subs[b_out]:
                cover_count[t_idx] -= 1
            current_blocks.pop(idx_out)
            selected_set.discard(b_out)
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = current_blocks[:]
            tabu_dict[b_out] = iteration + tabu_len
            continue
            
        # Find a block b_in that covers all of deficit
        # And is not in current_blocks (or is, but we are swapping so it shouldn't be the same)
        # And is not tabu
        
        # Strategy: Pick a random t from deficit, get candidates, filter.
        # If deficit is large, this is hard.
        # If deficit is small (1 or 2), it's easier.
        
        # To speed up, pick a random t from deficit
        target_t = rng.choice(list(deficit))
        candidates = t_to_ksubs[target_t]
        
        # Filter candidates:
        # 1. Not in selected_set (unless it's b_out? No, b_out is in selected_set. We are swapping OUT b_out. So b_in must not be in selected_set - {b_out})
        # 2. Not tabu
        # 3. Covers all of deficit
        
        found_b_in = -1
        # Sample candidates to avoid checking all 50k
        # The number of candidates for a t-subset is C(v-t, k-t) = C(15, 3) = 455.
        # 455 is small enough to check all!
        
        for b_in in candidates:
            if b_in in selected_set:
                continue
            if b_in in tabu_dict and tabu_dict[b_in] > iteration:
                continue
                
            # Check if b_in covers all deficit
            # deficit is a set, block_t_subs[b_in] is a set
            if deficit.issubset(block_t_subs[b_in]):
                found_b_in = b_in
                break
                
        if found_b_in != -1:
            # Perform swap
            # Remove b_out
            for t_idx in block_t_subs[b_out]:
                cover_count[t_idx] -= 1
            current_blocks[idx_out] = found_b_in
            selected_set.discard(b_out)
            selected_set.add(found_b_in)
            
            # Add found_b_in
            for t_idx in block_t_subs[found_b_in]:
                cover_count[t_idx] += 1
                
            # Tabu both
            tabu_dict[b_out] = iteration + tabu_len
            tabu_dict[found_b_in] = iteration + tabu_len
            
            # Check size? Swap keeps size same. 
            # But maybe it enables future removals.
            # We don't update best_size here unless we do a removal afterwards.
            # However, if we find a better arrangement, it might lead to smaller size later.
            
        # Clean up tabu dict occasionally to prevent memory bloat? 
        # It's small (keys are block indices).
        
    # Restore best solution
    # We need to reconstruct cover_count for best_blocks? No, we just return the blocks.
    # The evaluator checks validity.
    
    return [all_k_subs[i] for i in best_blocks]
