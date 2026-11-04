import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, store the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
    
    # Convert to a numpy array for faster access, but keep lists for iteration speed if needed
    # Actually, storing as a list of arrays is often faster for indexing than a 2D array with ragged rows
    # But since all blocks have the same number of t-subsets (C(k,t)), we can use a 2D array
    t_per_block = math.comb(k, t)
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)

    # Inverse: for each t-subset, which blocks cover it?
    # This is a sparse matrix essentially. 
    # tsub_blocks[tsub_idx] = list of block indices
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i in range(num_blocks):
        for j in block_tsubs[i]:
            tsub_blocks[j].append(i)

    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin

    def check_time(budget=10.0):
        return time.time() - start_time > deadline - budget

    # --- 1. Greedy Initialization ---
    # Strategy: Iteratively pick the block that covers the most *currently uncovered* t-subsets.
    # To speed this up, we can maintain a "weight" for each uncovered t-subset.
    # Weight of a t-subset = 1 / (number of blocks covering it).
    # Then score of a block = sum of weights of uncovered t-subsets it covers.
    
    # Precompute weights
    tsub_weights = np.zeros(num_tsubs, dtype=np.float64)
    for i in range(num_tsubs):
        tsub_weights[i] = 1.0 / len(tsub_blocks[i]) if tsub_blocks[i] else 0.0

    def greedy_init():
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        selected = []
        
        # Precompute block scores based on initial weights
        # score[b] = sum(uncovered[t] * weight[t] for t in block_tsubs[b])
        # Initially all uncovered, so score[b] = sum(weight[t] for t in block_tsubs[b])
        
        # To update scores efficiently, we can't just recompute everything.
        # But for the first few steps, or if we just recompute, it might be slow.
        # Let's use a simpler heuristic: 
        # 1. Pick random uncovered t-subset s.
        # 2. Among blocks covering s, pick the one covering the most *other* uncovered t-subsets.
        
        while uncovered_count > 0:
            if check_time(5.0):
                break
            
            # Pick a random uncovered t-subset
            # To avoid scanning the whole array, we can keep a list of uncovered indices
            # But updating that list is O(N). 
            # Random probing is O(1) expected if density is high, but gets slow when few are left.
            # Given num_tsubs ~ 11k, and we pick ~300 blocks, density stays high initially.
            
            s_idx = -1
            for _ in range(100):
                cand = random.randint(0, num_tsubs - 1)
                if uncovered[cand]:
                    s_idx = cand
                    break
            
            if s_idx == -1:
                # Fallback: linear scan (should be rare)
                for i in range(num_tsubs):
                    if uncovered[i]:
                        s_idx = i
                        break
                if s_idx == -1:
                    break # All covered?
            
            # Get candidates covering s_idx
            candidates = tsub_blocks[s_idx]
            if not candidates:
                break
                
            # Evaluate candidates
            best_block = -1
            best_gain = -1
            
            # Optimization: Instead of checking all candidates, maybe just sample a few?
            # No, for greedy, we want the best.
            # Number of candidates can be large.
            # Let's compute gain for each candidate.
            # Gain = number of uncovered t-subsets in the block.
            
            # We can vectorize this if we convert candidates to an array
            cand_arr = np.array(candidates, dtype=np.int32)
            # block_tsubs_np[cand_arr] is a 2D array (num_candidates x t_per_block)
            # We want sum(uncovered[t] for t in block)
            
            # This creates a large temporary array if num_candidates is large.
            # Max candidates? 
            # A t-subset is contained in C(v-t, k-t) blocks.
            # C(14, 3) = 364. So max candidates is ~364.
            # 364 * 56 = ~20k elements. This is fine for NumPy.
            
            gains = np.sum(uncovered[block_tsubs_np[cand_arr]], axis=1)
            best_local_idx = np.argmax(gains)
            best_block = int(cand_arr[best_local_idx])
            best_gain = int(gains[best_local_idx])
            
            if best_gain == 0:
                break # Should not happen if s_idx is uncovered and in best_block
            
            selected.append(best_block)
            
            # Update uncovered
            for j in block_tsubs[best_block]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected

    # --- 2. Reduction Phase ---
    # Remove redundant blocks.
    def reduce_blocks(block_indices):
        block_indices = list(block_indices)
        if not block_indices:
            return block_indices

        # cover_count[t] = number of selected blocks covering t
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs_np[b_idx]] += 1

        changed = True
        max_passes = 50
        for _ in range(max_passes):
            if check_time(5.0):
                break
            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                b_tsubs = block_tsubs_np[b_idx]
                # If any t-subset in this block has cover_count == 1, it's essential
                if np.any(cover_count[b_tsubs] == 1):
                    new_block_indices.append(b_idx)
                else:
                    # Redundant
                    changed = True
                    cover_count[b_tsubs] -= 1
            block_indices = new_block_indices
            if not changed:
                break
        return block_indices

    # --- 3. Local Search (1-Opt and 2-Opt) ---
    # State: list of block indices.
    # Energy: len(blocks) + penalty * num_uncovered.
    # We want to minimize this.
    
    def local_search(initial_blocks, seed=42):
        rng = random.Random(seed)
        blocks = list(initial_blocks)
        
        # Build cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in blocks:
            cover_count[block_tsubs_np[b_idx]] += 1
        
        num_uncovered = int(np.sum(cover_count == 0))
        penalty = 1000
        current_energy = len(blocks) + penalty * num_uncovered
        best_energy = current_energy
        best_blocks = list(blocks)
        
        # Precompute a set of block indices for fast lookup?
        # Or just use a boolean array.
        in_blocks = np.zeros(num_blocks, dtype=np.uint8)
        for b in blocks:
            in_blocks[b] = 1
            
        # Helper to compute energy delta for adding/removing
        # This is complex to do incrementally without bugs.
        # Let's stick to a simpler, robust move evaluation.
        
        T = 5.0
        alpha = 0.99
        max_iters = 50000
        
        while T > 0.1 and not check_time(10.0):
            T *= alpha
            
            # 1-Opt: Try removing a random block
            if len(blocks) > 1:
                rem_idx = rng.randint(0, len(blocks) - 1)
                rem_block = blocks[rem_idx]
                
                # Check if any t-subset becomes uncovered
                b_tsubs = block_tsubs_np[rem_block]
                # If min(cover_count[b_tsubs]) > 1, safe to remove
                if np.min(cover_count[b_tsubs]) > 1:
                    # Remove
                    cover_count[b_tsubs] -= 1
                    blocks.pop(rem_idx)
                    in_blocks[rem_block] = 0
                    
                    current_energy -= 1 # Len decreased by 1, uncovered unchanged
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = list(blocks)
                else:
                    # Not safe, or would increase uncovered. 
                    # We could still try it if we accept worse energy (SA), 
                    # but for 1-Opt reduction, we only do safe moves.
                    pass
            
            # 2-Opt: Swap a block for another
            # Pick a random block to remove
            if len(blocks) > 1 and not check_time(1.0):
                rem_idx = rng.randint(0, len(blocks) - 1)
                rem_block = blocks[rem_idx]
                
                # Identify t-subsets that are "critical" for rem_block (cover_count == 1)
                b_tsubs = block_tsubs_np[rem_block]
                critical_tsubs = b_tsubs[cover_count[b_tsubs] == 1]
                
                if len(critical_tsubs) == 0:
                    # Safe to remove, handled by 1-Opt logic ideally, 
                    # but maybe we want to swap to a "better" block?
                    # For now, if safe, just remove.
                    # Actually, if safe, 1-Opt would have removed it.
                    # So here we assume there are critical t-subsets.
                    
                    # We need a replacement block that covers these critical t-subsets.
                    # Pick one critical t-subset randomly
                    target_tsub = rng.choice(critical_tsubs)
                    target_tsub_idx = int(target_tsub)
                    
                    # Candidates: blocks covering target_tsub_idx, not in current set
                    candidates = [b for b in tsub_blocks[target_tsub_idx] if not in_blocks[b]]
                    
                    if not candidates:
                        continue
                    
                    # Pick a random candidate? Or the best one?
                    # Let's pick the one that covers the most *other* critical t-subsets.
                    best_add = -1
                    best_score = -1
                    
                    # Limit candidates to avoid slowness
                    if len(candidates) > 50:
                        candidates = rng.sample(candidates, 50)
                    
                    for cand in candidates:
                        cand_tsubs = block_tsubs_np[cand]
                        # Score: number of critical t-subsets covered by cand
                        # We can use numpy intersect or just sum
                        # critical_tsubs is a small array.
                        # Let's just check if cand covers all critical tsubs? No, just maximize overlap.
                        # Overlap = sum(1 for t in critical_tsubs if t in cand_tsubs)
                        # Efficient way: 
                        # Since critical_tsubs is small, we can just loop.
                        score = 0
                        for t in critical_tsubs:
                            # Check if t is in cand_tsubs
                            # cand_tsubs is sorted? No.
                            # We can use a set for cand_tsubs? Too slow to build.
                            # Use np.isin? 
                            # For small arrays, python loop is okay.
                            if t in cand_tsubs:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_add = cand
                    
                    if best_add == -1:
                        continue
                    
                    # Perform Swap
                    # Remove rem_block
                    cover_count[rem_block_tsubs] -= 1 # Wait, rem_block_tsubs is b_tsubs
                    cover_count[b_tsubs] -= 1
                    # Add best_add
                    add_tsubs = block_tsubs_np[best_add]
                    cover_count[add_tsubs] += 1
                    
                    # Calculate Energy Change
                    # Old Uncovered: num_uncovered
                    # New Uncovered: 
                    # T-subsets in rem_block that were critical (count 1) become 0 unless covered by add_block.
                    # T-subsets in add_block that were 0 become 1.
                    
                    # Let's just recompute uncovered count for affected t-subsets.
                    affected = np.union1d(b_tsubs, add_tsubs)
                    
                    # Save old counts
                    old_counts = cover_count[affected].copy()
                    # Note: cover_count has already been updated!
                    # So old_counts are the NEW counts.
                    # We need to compare NEW uncovered vs OLD uncovered.
                    
                    # Wait, I updated cover_count before checking.
                    # Let's revert and do it properly.
                    
                    # Revert
                    cover_count[b_tsubs] += 1
                    cover_count[add_tsubs] -= 1
                    
                    # Proper Swap Evaluation
                    # 1. Mark rem_block as removed, add_block as added in cover_count
                    cover_count[b_tsubs] -= 1
                    cover_count[add_tsubs] += 1
                    
                    # 2. Count new uncovered in affected
                    new_uncovered_affected = np.sum(cover_count[affected] == 0)
                    
                    # 3. Count old uncovered in affected (from state before swap)
                    # We need the state before swap.
                    # Let's store it.
                    
                    # Revert again
                    cover_count[b_tsubs] += 1
                    cover_count[add_tsubs] -= 1
                    
                    old_uncovered_affected = np.sum(cover_count[affected] == 0)
                    
                    # Apply swap
                    cover_count[b_tsubs] -= 1
                    cover_count[add_tsubs] += 1
                    
                    delta_uncovered = int(new_uncovered_affected) - int(old_uncovered_affected)
                    delta_energy = delta_uncovered * penalty # Block count unchanged
                    
                    if delta_energy < 0 or rng.random() < math.exp(-delta_energy / T):
                        # Accept
                        blocks[rem_idx] = best_add
                        in_blocks[rem_block] = 0
                        in_blocks[best_add] = 1
                        current_energy += delta_energy
                        num_uncovered += delta_uncovered # Track global? Not strictly needed if we use energy
                        
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = list(blocks)
                    
                    else:
                        # Reject
                        cover_count[b_tsubs] += 1
                        cover_count[add_tsubs] -= 1

        # Final reduction
        best_blocks = reduce_blocks(best_blocks)
        return best_blocks

    # Main Execution
    initial = greedy_init()
    if not initial:
        return []
    
    initial = reduce_blocks(initial)
    
    best_blocks = list(initial)
    best_len = len(best_blocks)
    
    # Run Local Search with multiple seeds
    for i in range(10):
        if check_time(5.0):
            break
        result = local_search(initial, seed=i*10+1)
        if result and len(result) < best_len:
            best_len = len(result)
            best_blocks = result
            # Optionally re-initialize from this better solution?
            # initial = reduce_blocks(result)
    
    best_blocks = reduce_blocks(best_blocks)
    
    return [list(all_blocks[b]) for b in best_blocks]
