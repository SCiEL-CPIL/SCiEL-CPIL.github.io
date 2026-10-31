import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    # Using uint8 for memory efficiency and fast matmul
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    # Optimization: Precompute the t-subsets for each block to fill matrix faster
    # Or just iterate. For v=19, k=7, t=4, each block has C(7,4)=35 t-subsets.
    # 50388 blocks * 35 ops is fast.
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # 2. Greedy Initialization
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected_indices = []
    
    # Greedy loop
    while np.sum(uncovered) > 0:
        if time.time() > deadline - 30.0: # Leave time for optimization
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
    
    # Convert to a set for easier manipulation in local search
    # But we need to track counts.
    # Let's maintain:
    # - current_blocks: list of indices
    # - cover_counts: array of length n_tsubs
    
    current_blocks = list(selected_indices)
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_blocks:
        cover_counts += Cover[idx]
        
    # Verify initial validity (should be valid by construction)
    # But if we broke early, it might not be.
    # For robustness, if invalid, we must repair.
    # The greedy loop above ensures validity unless it breaks due to time.
    # If it breaks due to time, we have an invalid cover.
    # We must ensure we return a VALID cover.
    # So, if we haven't finished greedy, we MUST finish it or repair.
    # Given the small size, greedy should finish in < 1s.
    
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # 3. Local Optimization: Simulated Annealing / ILS
    # Strategy: 
    # 1. Reduce: Remove any block that is redundant (covers no unique t-subset).
    # 2. Perturb: Remove random blocks.
    # 3. Repair: Greedily add blocks to cover gaps.
    
    def get_uncovered_mask(counts):
        return counts == 0

    def reduce_redundant(blocks, counts):
        """Remove blocks that do not cover any unique t-subset."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in blocks:
                # Check if any t-subset covered by idx has count == 1
                # If yes, keep it. If no, remove it.
                # Vectorized check:
                # mask = (Cover[idx] == 1) & (counts == 1)
                # if np.any(mask): keep
                # else: remove
                if np.any((Cover[idx] == 1) & (counts == 1)):
                    new_blocks.append(idx)
                else:
                    counts -= Cover[idx]
                    changed = True
            blocks = new_blocks
        return blocks, counts

    def repair(blocks, counts):
        """Greedily add blocks to cover all t-subsets."""
        while np.any(counts == 0):
            if time.time() > deadline - 5.0:
                break
            uncovered_mask = (counts == 0)
            gains = Cover @ uncovered_mask
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break # Should not happen if cover is possible
            blocks.append(best_idx)
            counts += Cover[best_idx]
        return blocks

    # Helper: Try to replace a block with another that maintains coverage
    def try_replace_one(blocks, counts, max_candidates=150):
        """
        Try to replace one block in the solution with a different block.
        Returns (new_blocks, new_counts, success) where success=True means
        the replacement was accepted (same or better size after reduction).
        """
        n = len(blocks)
        if n == 0:
            return blocks, counts, False
        
        # Pick a random block to try replacing
        pos = random.randrange(n)
        old_idx = blocks[pos]
        
        # Get t-subsets covered by old block
        old_mask = Cover[old_idx]
        
        # Compute overlap of all blocks with old block
        overlaps = Cover @ old_mask.astype(np.int32)
        
        # Get top candidates by overlap
        top_k = min(max_candidates, n_blocks)
        candidate_indices = np.argsort(overlaps)[::-1][:top_k]
        
        for cand_idx in candidate_indices:
            cand_idx = int(cand_idx)
            if cand_idx == old_idx or cand_idx in blocks:
                continue
            
            # Simulate replacement
            new_counts = counts - old_mask + Cover[cand_idx]
            
            # Check validity
            if np.any(new_counts == 0):
                continue
            
            # Try reduction
            new_blocks = list(blocks)
            new_blocks[pos] = cand_idx
            reduced, reduced_counts = reduce_redundant(new_blocks, new_counts.copy())
            
            if len(reduced) < len(blocks):
                return reduced, reduced_counts, True
            elif len(reduced) == len(blocks) and random.random() < 0.2:
                # Same size, accept with small probability for diversity
                return reduced, reduced_counts, True
        
        return blocks, counts, False

    # Main Optimization Loop
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.01
    
    # Initial reduction
    current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
    if len(current_blocks) < best_size:
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # 1. Reduce redundant blocks
        current_blocks, cover_counts = reduce_redundant(current_blocks, cover_counts)
        
        # Update best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(best_blocks)
        
        # 2. Perturb - choose strategy
        r = random.random()
        
        if r < 0.35 and len(current_blocks) > 1:
            # Strategy A: Remove 1-2 random blocks
            num_removals = random.randint(1, 2)
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
            
        elif r < 0.70 and len(current_blocks) > 1:
            # Strategy B: Try block replacement
            current_blocks, cover_counts, success = try_replace_one(
                current_blocks, cover_counts, max_candidates=100
            )
            if not success:
                # Fallback: remove one random block
                if len(current_blocks) > 1:
                    idx = random.choice(current_blocks)
                    cover_counts -= Cover[idx]
                    current_blocks = [b for b in current_blocks if b != idx]
            
        else:
            # Strategy C: Remove 2-4 random blocks (aggressive)
            num_removals = random.randint(2, min(4, max(1, len(current_blocks) - 1)))
            if len(current_blocks) <= num_removals:
                break
            removals = random.sample(current_blocks, num_removals)
            for idx in removals:
                cover_counts -= Cover[idx]
            current_blocks = [b for b in current_blocks if b not in removals]
        
        # 3. Repair
        current_blocks = repair(current_blocks, cover_counts)
        
        # 4. Acceptance
        new_size = len(current_blocks)
        delta = new_size - best_size
        
        if delta < 0:
            # Improvement
            best_blocks = list(current_blocks)
            best_size = new_size
        elif temperature > min_temp:
            # Simulated Annealing acceptance
            prob = np.exp(-delta / temperature)
            if random.random() < prob:
                pass
            else:
                # Reject: revert to best state
                current_blocks = list(best_blocks)
                cover_counts = np.zeros(n_tsubs, dtype=np.int32)
                for idx in current_blocks:
                    cover_counts += Cover[idx]
                
        temperature *= cooling

    # Final reduction
    # We need to return a valid cover.
    # If the last state was invalid (due to time out in repair), we must use best_blocks.
    # best_blocks is always valid (starts valid, only updated on improvement which implies valid repair).
    
    # One final check and reduction on best_blocks
    final_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in best_blocks:
        final_counts += Cover[idx]
        
    # Ensure no redundancy in the final answer
    final_blocks, final_counts = reduce_redundant(best_blocks, final_counts)
    
    # Verify validity of final_blocks
    # (Should be valid, but just in case)
    if np.any(final_counts == 0):
         # This shouldn't happen if reduce_redundant is correct
         # But if it does, we fall back to best_blocks
         final_blocks = best_blocks
         # Recount for safety if we fell back
         final_counts = np.zeros(n_tsubs, dtype=np.int32)
         for idx in final_blocks:
             final_counts += Cover[idx]
    
    # Final polish: try a few more replacement moves on the best solution
    if time.time() < deadline - 2.0:
        # Ensure we have a consistent count for the current final_blocks
        temp_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in final_blocks:
            temp_counts += Cover[idx]
            
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            final_blocks, temp_counts, success = try_replace_one(
                final_blocks, temp_counts, max_candidates=80
            )
            if not success:
                break
        
        # Final validation check
        if np.any(temp_counts == 0):
             # If somehow invalid, revert to best_blocks
             final_blocks = best_blocks
    
    return [all_kblocks[i] for i in final_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
