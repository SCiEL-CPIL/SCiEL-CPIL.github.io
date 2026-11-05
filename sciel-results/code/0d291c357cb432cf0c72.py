import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit (or 120s if strict)
    
    # Precompute all t-subsets and map them to IDs
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset IDs it covers
    # block_ids[block_idx] = [id1, id2, ...]
    block_ids = []
    for b in all_blocks:
        ids = []
        for sub in itertools.combinations(b, t):
            ids.append(tsub_to_id[sub])
        block_ids.append(ids)
        
    # Precompute for each t-subset ID, the list of block indices that contain it
    # sub_blocks[tsub_id] = [block_idx1, block_idx2, ...]
    sub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, ids in enumerate(block_ids):
        for id in ids:
            sub_blocks[id].append(b_idx)
            
    best_blocks = None
    best_len = float('inf')
    
    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(r * 1000 + 42)
        
        # 1. Greedy Construction
        blocks = _greedy_cover_v2(v, k, t, n_tsubs, block_ids, sub_blocks, rng, deadline)
        
        if blocks is None:
            continue
            
        # 2. Local Search / Refinement
        blocks = _local_search_v2(blocks, v, k, t, n_tsubs, block_ids, sub_blocks, rng, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Early exit if we hit the known best or better
        if best_len <= 76:
            break

    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]

def _greedy_cover_v2(v, k, t, n_tsubs, block_ids, sub_blocks, rng, deadline):
    covered = [False] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            return None
            
        # Pick a random uncovered t-subset
        # Instead of maintaining a list of uncovered, we can just pick a random ID 
        # and check if it's uncovered. If so, proceed. If not, retry.
        # For efficiency, we can maintain a list of uncovered IDs and shrink it.
        # But shrinking a list is O(N). Let's just pick randomly and check.
        # With 2024 subsets, this is fast.
        
        seed_id = rng.randint(0, n_tsubs - 1)
        if covered[seed_id]:
            # Find an uncovered one
            # To avoid infinite loops, we can scan
            found = False
            for i in range(n_tsubs):
                if not covered[i]:
                    seed_id = i
                    found = True
                    break
            if not found:
                break # Should not happen if uncovered_count is maintained correctly
                
        candidates = sub_blocks[seed_id]
        if not candidates:
            return None # Error: no block covers this subset
            
        best_block = -1
        best_gain = -1
        best_tiebreak = []
        
        # Calculate gain for each candidate
        for b_idx in candidates:
            gain = 0
            for id in block_ids[b_idx]:
                if not covered[id]:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                best_tiebreak = [b_idx]
            elif gain == best_gain and gain > 0:
                best_tiebreak.append(b_idx)
                
        if best_block == -1:
            # No gain? Should not happen if seed_id is uncovered and covered by candidates
            return None
            
        # Pick randomly from tiebreaks to add diversity
        chosen_block = rng.choice(best_tiebreak)
        blocks.append(all_blocks[chosen_block]) # Need to return actual block values
        
        # Update covered
        for id in block_ids[chosen_block]:
            if not covered[id]:
                covered[id] = True
                uncovered_count -= 1
                
    return blocks

def _local_search_v2(blocks, v, k, t, n_tsubs, block_ids, sub_blocks, rng, deadline):
    # Map block value to index in block_ids for quick lookup
    # Since blocks are tuples, we can use a dictionary
    # But we need to be careful with duplicates. The problem says blocks must be distinct.
    # We assume input blocks are distinct.
    
    # Convert blocks to indices in the global all_blocks list?
    # It's easier to keep the actual block tuples and compute their coverage on the fly
    # or map them to global indices.
    
    # Let's map the current blocks to global indices to use precomputed data.
    block_val_to_idx = {b: i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for b in blocks:
        current_block_indices.append(block_val_to_idx[b])
        
    # Compute cover count for each t-subset
    cover_count = [0] * n_tsubs
    for b_idx in current_block_indices:
        for id in block_ids[b_idx]:
            cover_count[id] += 1
            
    improved_indices = list(current_block_indices)
    
    # Iterative improvement
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # 1. Remove redundant blocks
        for i in range(len(improved_indices) - 1, -1, -1):
            b_idx = improved_indices[i]
            # Check if any t-subset is covered ONLY by this block
            is_redundant = True
            for id in block_ids[b_idx]:
                if cover_count[id] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Remove block
                for id in block_ids[b_idx]:
                    cover_count[id] -= 1
                improved_indices.pop(i)
                changed = True
                break # Restart loop after removal
                
        if changed:
            continue
            
        # 2. Try to swap/replace blocks to enable removal of others
        # For each block, find its unique subsets
        n = len(improved_indices)
        if n == 0:
            break
            
        # Pick a random block to try to improve
        b_idx_pos = rng.randint(0, n - 1)
        b_idx = improved_indices[b_idx_pos]
        
        # Find unique subsets of b_idx
        unique_subs = [id for id in block_ids[b_idx] if cover_count[id] == 1]
        
        if not unique_subs:
            # This block is redundant? Should have been caught above.
            # But cover_count might have changed.
            # Actually, if unique_subs is empty, it means all subsets are covered by others.
            # So it is redundant.
            for id in block_ids[b_idx]:
                cover_count[id] -= 1
            improved_indices.pop(b_idx_pos)
            changed = True
            continue
            
        # Try to find a new block that covers all unique_subs
        # Intersect the list of blocks covering each unique sub
        # Start with the list of blocks covering the first unique sub
        candidate_indices = set(sub_blocks[unique_subs[0]])
        
        for id in unique_subs[1:]:
            candidate_indices.intersection_update(sub_blocks[id])
            if not candidate_indices:
                break
                
        if candidate_indices:
            # Filter out blocks already in the solution
            current_set = set(improved_indices)
            valid_candidates = [c for c in candidate_indices if c not in current_set]
            
            if valid_candidates:
                # Pick a random candidate
                new_b_idx = rng.choice(valid_candidates)
                
                # Apply swap
                # Remove old block
                for id in block_ids[b_idx]:
                    cover_count[id] -= 1
                # Add new block
                for id in block_ids[new_b_idx]:
                    cover_count[id] += 1
                    
                improved_indices[b_idx_pos] = new_b_idx
                changed = True
                # We don't break here, we continue to check for redundancies in the next loop iteration
                
    # Convert back to block values
    return [all_blocks[i] for i in improved_indices]

# Helper to access all_blocks from within functions if needed, 
# but passing it explicitly is cleaner. 
# However, _local_search_v2 uses all_blocks which is defined in construct.
# We need to make all_blocks global or pass it.
# Let's refactor to pass all_blocks.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {sub: i for i, sub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_ids = []
    for b in all_blocks:
        ids = []
        for sub in itertools.combinations(b, t):
            ids.append(tsub_to_id[sub])
        block_ids.append(ids)
        
    sub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, ids in enumerate(block_ids):
        for id in ids:
            sub_blocks[id].append(b_idx)
            
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(r * 1000 + 42)
        
        blocks = _greedy_cover_v2(v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline)
        
        if blocks is None:
            continue
            
        blocks = _local_search_v2(blocks, v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 76:
            break

    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]

def _greedy_cover_v2(v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline):
    covered = [False] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            return None
            
        seed_id = rng.randint(0, n_tsubs - 1)
        if covered[seed_id]:
            found = False
            for i in range(n_tsubs):
                if not covered[i]:
                    seed_id = i
                    found = True
                    break
            if not found:
                break
                
        candidates = sub_blocks[seed_id]
        if not candidates:
            return None
            
        best_block = -1
        best_gain = -1
        best_tiebreak = []
        
        for b_idx in candidates:
            gain = 0
            for id in block_ids[b_idx]:
                if not covered[id]:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                best_tiebreak = [b_idx]
            elif gain == best_gain and gain > 0:
                best_tiebreak.append(b_idx)
                
        if best_block == -1:
            return None
            
        chosen_block = rng.choice(best_tiebreak)
        blocks.append(all_blocks[chosen_block])
        
        for id in block_ids[chosen_block]:
            if not covered[id]:
                covered[id] = True
                uncovered_count -= 1
                
    return blocks

def _local_search_v2(blocks, v, k, t, n_tsubs, block_ids, sub_blocks, all_blocks, rng, deadline):
    block_val_to_idx = {b: i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for b in blocks:
        current_block_indices.append(block_val_to_idx[b])
        
    cover_count = [0] * n_tsubs
    for b_idx in current_block_indices:
        for id in block_ids[b_idx]:
            cover_count[id] += 1
            
    improved_indices = list(current_block_indices)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        for i in range(len(improved_indices) - 1, -1, -1):
            b_idx = improved_indices[i]
            is_redundant = True
            for id in block_ids[b_idx]:
                if cover_count[id] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                for id in block_ids[b_idx]:
                    cover_count[id] -= 1
                improved_indices.pop(i)
                changed = True
                break
                
        if changed:
            continue
            
        n = len(improved_indices)
        if n == 0:
            break
            
        b_idx_pos = rng.randint(0, n - 1)
        b_idx = improved_indices[b_idx_pos]
        
        unique_subs = [id for id in block_ids[b_idx] if cover_count[id] == 1]
        
        if not unique_subs:
            for id in block_ids[b_idx]:
                cover_count[id] -= 1
            improved_indices.pop(b_idx_pos)
            changed = True
            continue
            
        candidate_indices = set(sub_blocks[unique_subs[0]])
        for id in unique_subs[1:]:
            candidate_indices.intersection_update(sub_blocks[id])
            if not candidate_indices:
                break
                
        if candidate_indices:
            current_set = set(improved_indices)
            valid_candidates = [c for c in candidate_indices if c not in current_set]
            
            if valid_candidates:
                new_b_idx = rng.choice(valid_candidates)
                
                for id in block_ids[b_idx]:
                    cover_count[id] -= 1
                for id in block_ids[new_b_idx]:
                    cover_count[id] += 1
                    
                improved_indices[b_idx_pos] = new_b_idx
                changed = True
                
    return [all_blocks[i] for i in improved_indices]
