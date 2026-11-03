import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _simulated_annealing(blocks, block_covers, sub_to_blocks, n_t, deadline, rng):
    """
    Simulated Annealing for Set Cover.
    State: set of selected block indices.
    Energy: len(selected) + penalty * sum(1 for uncovered subsets).
    Moves: Add a random block, Remove a random block.
    """
    if not blocks:
        return set()
    
    n_blocks = len(blocks)
    
    # Initial state: Random subset of blocks
    # Start with a small random set and let SA grow/shrink, or start with a greedy-like set?
    # Starting with a random small set is risky if it doesn't cover anything.
    # Let's start with a random set of size ~ n_t / (k choose t) * 2
    # C(8,5) = 56. n_t = C(19,5) = 11628.
    # Approx blocks needed: 11628/56 ~ 207.
    # Let's start with a random set of 200 blocks.
    initial_size = min(n_blocks, 200)
    selected = set(rng.sample(range(n_blocks), initial_size))
    
    # Calculate initial uncovered count
    # We need to efficiently track coverage.
    # Since n_t is ~11k and blocks cover ~56 subsets each, updating counts is fast.
    # However, doing it from scratch every step is O(n_t * |selected|) which might be slow.
    # Better: Maintain a count array `cover_counts` of size n_t.
    
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_covers[b]:
            cover_counts[s] += 1
            
    uncovered_count = np.sum(cover_counts == 0)
    current_energy = len(selected) + 10 * uncovered_count # Penalty for uncovered is high
    
    best_selected = set(selected)
    best_energy = current_energy
    
    # SA Parameters
    # Time budget per restart is roughly 10-15s.
    # Let's run for a fixed number of steps or until deadline.
    start_time = time.time()
    # Estimate steps: ~10k steps per second? 
    # Update is O(|block_covers[b]|) ~ 56 operations.
    # So ~100k steps per second.
    # 10 seconds -> 1 million steps.
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99999 # Very slow cooling to allow many steps
    
    while _deadline_ok(deadline):
        # Check time periodically to avoid long loops
        if time.time() - start_time > 12: # Hard cap per restart inside SA
            break
            
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to add a block
            # Pick a random block not in selected
            # To be efficient, we can pick a random block and check if it's in selected.
            # Or maintain a list of unselected? That's complex to update.
            # Given n_blocks ~ 75582, picking random is fine.
            b = rng.randint(0, n_blocks - 1)
            if b in selected:
                continue
                
            # Calculate delta
            # Gain in coverage: how many currently uncovered subsets does b cover?
            # gain = sum(1 for s in block_covers[b] if cover_counts[s] == 0)
            gain = 0
            for s in block_covers[b]:
                if cover_counts[s] == 0:
                    gain += 1
            
            # Cost: +1 block
            # Energy change: +1 - 10*gain
            delta_E = 1 - 10 * gain
            
            if delta_E < 0 or (T > T_min and rng.random() < np.exp(-delta_E / T)):
                # Accept move
                selected.add(b)
                for s in block_covers[b]:
                    cover_counts[s] += 1
                    if cover_counts[s] == 1:
                        uncovered_count -= 1
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = set(selected)
                    
        else:
            # Try to remove a block
            if not selected:
                continue
                
            # Pick a random block from selected
            # Converting to list every time is O(|selected|). 
            # |selected| ~ 200-300. This is acceptable.
            b = rng.choice(list(selected))
            
            # Calculate delta
            # Loss in coverage: how many subsets are ONLY covered by b?
            # loss = sum(1 for s in block_covers[b] if cover_counts[s] == 1)
            loss = 0
            for s in block_covers[b]:
                if cover_counts[s] == 1:
                    loss += 1
            
            # Cost: -1 block
            # Energy change: -1 + 10*loss
            delta_E = -1 + 10 * loss
            
            if delta_E < 0 or (T > T_min and rng.random() < np.exp(-delta_E / T)):
                # Accept move
                selected.discard(b)
                for s in block_covers[b]:
                    cover_counts[s] -= 1
                    if cover_counts[s] == 0:
                        uncovered_count += 1
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_selected = set(selected)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final Pruning: Remove any redundant blocks from best_selected
    # This ensures we return a minimal set if possible
    final_counts = np.zeros(n_t, dtype=np.int32)
    for b in best_selected:
        for s in block_covers[b]:
            final_counts[s] += 1
            
    pruned_selected = set(best_selected)
    for b in list(pruned_selected):
        redundant = True
        for s in block_covers[b]:
            if final_counts[s] <= 1:
                redundant = False
                break
        if redundant:
            pruned_selected.discard(b)
            for s in block_covers[b]:
                final_counts[s] -= 1
                
    return pruned_selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks (might not be strictly needed for SA if we use random moves, 
    # but useful for initial greedy or if we switch back)
    # For now, we skip building sub_to_blocks to save time if not used.
    # SA uses random block selection, so sub_to_blocks is not required.
    
    # 3. Multiple SA restarts
    best_result = None
    best_size = float('inf')
    
    # Number of restarts
    # Each restart takes ~10-12s.
    # We have ~115s. So ~9 restarts.
    n_restarts = 9
    
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Run SA
        selected = _simulated_annealing(blocks, block_covers, None, n_t, deadline, rng)
        
        if not selected:
            continue
            
        # Verify validity and size
        # SA should maintain validity if initialized with a valid cover, but we start with random.
        # Wait, my SA starts with random blocks. It might not cover everything.
        # The energy function penalizes uncovered subsets.
        # If the final solution has uncovered subsets, it's invalid.
        # We must check if uncovered_count is 0.
        
        # Let's re-verify the final set
        check_counts = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            for s in block_covers[b]:
                check_counts[s] += 1
                
        if np.all(check_counts > 0):
            if len(selected) < best_size:
                best_size = len(selected)
                best_result = set(selected)
                
        # Early stopping
        if best_size <= 270:
            break
            
    # Fallback if SA failed to find a valid cover (unlikely with high penalty, but possible if stuck)
    if best_result is None:
        # Simple Greedy Fallback
        uncovered = np.ones(n_t, dtype=bool)
        uncovered_count = n_t
        selected = set()
        
        # Precompute sub_to_blocks for greedy
        sub_to_blocks = [[] for _ in range(n_t)]
        for i, covers in enumerate(block_covers):
            for j in covers:
                sub_to_blocks[j].append(i)
                
        while uncovered_count > 0 and _deadline_ok(deadline):
            # Find subset with fewest candidates
            min_count = n_blocks + 1
            target_sub = -1
            for s in range(n_t):
                if uncovered[s]:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
                            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            best_block = -1
            max_gain = -1
            
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if uncovered[s]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    
            if best_block == -1:
                break
                
            selected.add(best_block)
            for s in block_covers[best_block]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    
        if selected:
            best_result = selected
            best_size = len(selected)
            
    if best_result is None:
        return []
        
    result = [list(blocks[i]) for i in best_result]
    
    return result
