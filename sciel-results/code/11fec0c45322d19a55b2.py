import itertools
import random
import time
import sys
import numpy as np

def _greedy_covering(v, k, t, rng, time_limit):
    """Fast greedy construction to initialize VNS."""
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x not in S:
                        tset = tuple(sorted(S + (x,)))
                        if tset in uncovered:
                            gain += 1
                # Add small random noise to break ties and explore
                score = gain + rng.random() * 0.01
                if score > best_gain:
                    best_gain = score
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]

def _to_bitsets(blocks, v, t):
    """Convert blocks to bitset representation for fast coverage checks."""
    n_blocks = len(blocks)
    # Map each t-set to an index 0..C(v,t)-1
    tset_to_idx = {}
    idx = 0
    for ts in itertools.combinations(range(v), t):
        tset_to_idx[ts] = idx
        idx += 1
    
    n_tsets = idx
    # coverage[i] is a bitset where bit j is 1 if block i covers t-set j
    coverage = np.zeros((n_blocks, n_tsets), dtype=np.uint8)
    
    for i, b in enumerate(blocks):
        bset = set(b)
        for ts in itertools.combinations(b, t):
            coverage[i, tset_to_idx[ts]] = 1
    
    return coverage, tset_to_idx, n_tsets

def _vns_search(blocks, v, k, t, rng, time_limit):
    """
    Variable Neighborhood Search (VNS) for covering design.
    State: list of blocks.
    Neighborhood N_r: swap r random blocks with r new random blocks.
    Use bitsets for O(1) validity check of the resulting covering.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    if not blocks:
        return blocks
    
    # Initial bitset setup
    coverage, tset_to_idx, n_tsets = _to_bitsets(blocks, v, t)
    current_size = len(blocks)
    
    # Precompute all t-sets for fast index lookup if needed, but we use tset_to_idx
    # Actually, for checking if a NEW block covers a t-set, we just iterate combinations of the new block.
    
    def is_valid_covering(coverage_matrix):
        """Check if union of rows covers all columns."""
        if coverage_matrix.shape[0] == 0:
            return n_tsets == 0
        # A column is covered if any row has a 1 in it
        # np.any(coverage_matrix, axis=0) returns a boolean array of length n_tsets
        # We need all to be True
        return np.all(np.any(coverage_matrix, axis=0))
    
    # Ensure initial state is valid (greedy should guarantee this, but just in case)
    if not is_valid_covering(coverage):
        # Fallback: try to fix or just return
        pass

    best_blocks = [list(b) for b in blocks]
    best_size = current_size
    
    # VNS Loop
    r = 1
    max_r = 3  # Limit r to keep complexity manageable within time
    
    while time.time() < deadline:
        if r > max_r:
            r = 1  # Reset if no improvement found at high r, or restart VNS cycle
            
        # Generate a neighbor in N_r
        # Pick r distinct indices to remove
        if current_size < r:
            r = max(1, current_size)
            if r > max_r: break
            
        indices_to_remove = rng.sample(range(current_size), r)
        indices_to_remove.sort(reverse=True) # Remove from end first to keep indices stable? 
        # Actually, let's just create a new list
        
        new_blocks = [list(b) for b in blocks]
        
        # Remove the selected blocks
        for idx in indices_to_remove:
            new_blocks.pop(idx)
        
        # Generate r new random blocks
        # Heuristic: Random blocks might be too weak. 
        # Better: Generate blocks that try to cover currently poorly covered t-sets?
        # For VNS, pure random is standard for exploration, but might be too slow to find valid coverings.
        # Let's use a "smart random": 
        # 1. Identify t-sets with low coverage count.
        # 2. Construct new blocks biased towards those points.
        
        # Compute coverage counts for current state (before removal) to guide new block generation?
        # No, we removed them. Let's look at the state AFTER removal.
        temp_coverage = np.vstack([coverage[i] for i in range(current_size) if i not in indices_to_remove]) if new_blocks else np.zeros((0, n_tsets), dtype=np.uint8)
        
        # Find t-sets with 0 coverage (invalid state) or low coverage
        # If invalid, we MUST cover the uncovered ones.
        # Let's identify uncovered t-sets in the temporary state
        if temp_coverage.shape[0] > 0:
            uncovered_mask = ~np.any(temp_coverage, axis=0)
        else:
            uncovered_mask = np.ones(n_tsets, dtype=bool)
        
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # If there are uncovered t-sets, we must generate blocks that cover them.
        # Map index back to t-set
        idx_to_tset = {v: k for k, v in tset_to_idx.items()}
        
        for _ in range(r):
            # Try to construct a block
            # If there are uncovered t-sets, pick one at random and expand it
            if len(uncovered_indices) > 0:
                # Pick a random uncovered t-set
                u_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
                u_tset = idx_to_tset[u_idx]
                block = list(u_tset)
            else:
                # No uncovered t-sets, pick a random t-set or random block
                # Pick random block points to maintain density?
                # Or just random
                block = []
            
            # Fill the rest of the block randomly
            while len(block) < k:
                x = rng.randint(0, v-1)
                if x not in block:
                    block.append(x)
            
            block.sort()
            new_blocks.append(block)
            
            # Update temporary coverage and uncovered mask for next new block?
            # This is expensive if done naively. 
            # For simplicity and speed in this limited r, let's just add to new_blocks 
            # and do a full validity check at the end of the r-loop.
            # However, if we don't update, subsequent new blocks might not target remaining uncovered sets.
            # Given r is small (1-3), we can afford a quick check.
            
            # Quick update: Add this new block's coverage to temp_coverage
            new_row = np.zeros(n_tsets, dtype=np.uint8)
            bset = set(block)
            for ts in itertools.combinations(block, t):
                new_row[tset_to_idx[ts]] = 1
            
            if temp_coverage.shape[0] > 0:
                temp_coverage = np.vstack([temp_coverage, new_row])
            else:
                temp_coverage = new_row.reshape(1, -1)
                
            # Update uncovered mask
            uncovered_mask = ~np.any(temp_coverage, axis=0)
            uncovered_indices = np.where(uncovered_mask)[0]
            
        # Check validity of the new state
        if is_valid_covering(temp_coverage):
            new_size = len(new_blocks)
            if new_size < best_size:
                best_size = new_size
                best_blocks = new_blocks
                # Reset r to 1 as we found an improvement
                r = 1
                # Update the main state
                blocks = new_blocks
                coverage, tset_to_idx, n_tsets = _to_bitsets(blocks, v, t)
                current_size = len(blocks)
            else:
                # No improvement, increase r
                r += 1
        else:
            # Invalid neighbor, increase r (or retry? VNS usually moves to next neighborhood)
            r += 1

    return best_blocks

def _verify_covering(blocks, v, t):
    """Verify that every t-subset is covered by at least one block."""
    tset_to_blocks = {}
    for i, b in enumerate(blocks):
        bset = set(b)
        for ts in itertools.combinations(b, t):
            if ts not in tset_to_blocks:
                tset_to_blocks[ts] = []
            tset_to_blocks[ts].append(i)
    for ts in itertools.combinations(range(v), t):
        if ts not in tset_to_blocks or not tset_to_blocks[ts]:
            return False
    return True

def construct(v, k, t):
    start_time = time.time()
    total_budget = 145.0 # Safety margin
    
    # Phase 1: Initial Construction
    # Run a few greedy instances to get a decent starting point
    # VNS is sensitive to initial state, so we want a good one.
    greedy_budget = 20.0
    ls_budget = total_budget - greedy_budget - 5.0 # 5s margin
    
    best_blocks = None
    best_len = float('inf')
    
    # Run 3 greedy instances
    for i in range(3):
        elapsed = time.time() - start_time
        if elapsed > total_budget - ls_budget - 10.0:
            break
        rng = random.Random(100 + i)
        per_time = greedy_budget / 3.0
        blocks = _greedy_covering(v, k, t, rng, per_time)
        
        # Quick verify
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        # Fallback if greedy failed (shouldn't happen)
        best_blocks = _greedy_covering(v, k, t, random.Random(1), 5.0)
        best_len = len(best_blocks)

    # Phase 2: VNS Refinement
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        rng = random.Random(42)
        refined = _vns_search(best_blocks, v, k, t, rng, remaining)
        
        # Verify refined
        if _verify_covering(refined, v, t):
            if len(refined) < best_len:
                best_blocks = refined
                best_len = len(refined)
            else:
                # If VNS didn't improve, maybe try a different seed for VNS on the same start?
                # Or just keep the best.
                pass
    
    return best_blocks
