import time
import itertools
import numpy as np
import random

def _precompute(v, k, t):
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Incidence matrix M[i, j] = 1 if block i covers t-subset j
    # M is n_blocks x n_tsubs
    M = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        B_set = set(B)
        for j, ts in enumerate(all_tsubs):
            if all(x in B_set for x in ts):
                M[i, j] = 1
                
    return all_kblocks, all_tsubs, M

def _greedy_construct(M, n_tsubs, deadline, seed=42):
    rng = random.Random(seed)
    n_blocks = M.shape[0]
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Calculate gain for each block
        # gains[i] = number of uncovered t-subsets covered by block i
        gains = M[:, uncovered].sum(axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        
        selected.append(best_idx)
        uncovered &= ~M[best_idx, :].astype(bool)
        
    return selected

def _local_optimize(selected, M, n_tsubs, deadline, seed=42):
    rng = random.Random(seed)
    selected = list(selected)
    
    # Build cover counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected:
        cover_counts += M[idx, :]
        
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Remove redundant blocks
        new_selected = []
        for idx in selected:
            # Check if this block covers any t-subset uniquely
            # Unique if cover_counts[j] == 1 and M[idx, j] == 1
            unique_covered = (cover_counts == 1) & (M[idx, :] == 1)
            if unique_covered.any():
                new_selected.append(idx)
            else:
                # Remove block
                cover_counts -= M[idx, :]
                improved = True
        selected = new_selected
        
        # 2. Try 1-for-1 swaps
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            idx = selected[i]
            unique_covered = (cover_counts == 1) & (M[idx, :] == 1)
            if not unique_covered.any():
                # Should have been removed, but maybe state changed?
                cover_counts -= M[idx, :]
                selected.pop(i)
                improved = True
                i -= 1
                continue
            
            # Find candidates that cover all unique_covered
            # M[j, unique_covered].all(axis=1)
            # This is a boolean array of size n_blocks
            candidates = np.where(M[:, unique_covered].all(axis=1))[0]
            
            # Exclude current block
            candidates = candidates[candidates != idx]
            
            if len(candidates) == 0:
                continue
                
            # Pick a candidate
            # Prefer one that covers many "redundant" t-subsets (count > 1)
            # to potentially make other blocks redundant?
            # Or just random.
            # Let's pick the one that maximizes coverage of t-subsets with count >= 2
            # This might help in future redundancy removal?
            # Actually, simpler: just pick random.
            best_alt = rng.choice(candidates)
            
            # Swap
            cover_counts -= M[idx, :]
            cover_counts += M[best_alt, :]
            selected[i] = best_alt
            improved = True
            
        # 3. Try 2-for-1 merges
        # This is more expensive. Only try if time permits.
        if time.time() < deadline - 5:
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(selected)):
                    if time.time() > deadline:
                        break
                    idx1 = selected[i]
                    idx2 = selected[j]
                    
                    # Union of unique covered t-subsets
                    # A t-subset is uniquely covered by {idx1, idx2} if:
                    # 1. It is covered by idx1 or idx2
                    # 2. It is NOT covered by any other block in selected
                    # 3. It is covered by idx1 or idx2 with count 1?
                    # Actually, if we remove idx1 and idx2, a t-subset becomes uncovered
                    # if and only if it was covered ONLY by idx1 or ONLY by idx2.
                    
                    # Let's compute the set of t-subsets that would become uncovered
                    # if we removed idx1 and idx2.
                    # These are t-subsets where cover_counts == 1 AND M[idx1, :] == 1 OR M[idx2, :] == 1
                    # Wait, if cover_counts == 1, it's covered by exactly one block.
                    # If that block is idx1 or idx2, it becomes uncovered.
                    
                    mask1 = M[idx1, :]
                    mask2 = M[idx2, :]
                    
                    # T-subsets uniquely covered by idx1
                    uniq1 = (cover_counts == 1) & mask1
                    # T-subsets uniquely covered by idx2
                    uniq2 = (cover_counts == 1) & mask2
                    
                    # T-subsets that would be lost
                    lost = uniq1 | uniq2
                    
                    if not lost.any():
                        # We can remove one of them?
                        # If no unique t-subsets are lost, we can remove one block.
                        # But we should have removed redundant blocks in step 1.
                        # So this case should be rare or impossible if step 1 is robust.
                        # However, if we remove idx1, idx2 might become redundant?
                        # Let's just try to find a block that covers `lost`.
                        pass
                    
                    # Find a block k that covers all `lost` t-subsets
                    candidates = np.where(M[:, lost].all(axis=1))[0]
                    candidates = candidates[(candidates != idx1) & (candidates != idx2)]
                    
                    if len(candidates) > 0:
                        best_k = rng.choice(candidates)
                        
                        # Check if this swap is beneficial
                        # We remove 2 blocks, add 1. Net -1.
                        # But we must ensure coverage is maintained.
                        # The `lost` t-subsets are covered by best_k.
                        # Are there any t-subsets covered by idx1 or idx2 that are NOT in `lost`?
                        # Yes, those covered by other blocks too.
                        # Those are fine.
                        
                        # So, if best_k covers `lost`, we can replace {idx1, idx2} with {best_k}.
                        
                        # Update counts
                        cover_counts -= M[idx1, :]
                        cover_counts -= M[idx2, :]
                        cover_counts += M[best_k, :]
                        
                        # Update selected
                        # Remove idx1 and idx2, add best_k
                        new_selected = []
                        for idx in selected:
                            if idx != idx1 and idx != idx2:
                                new_selected.append(idx)
                        new_selected.append(best_k)
                        selected = new_selected
                        improved = True
                        break # Restart inner loop as indices changed
                if improved:
                    break
                    
    return selected

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_kblocks, all_tsubs, M = _precompute(v, k, t)
    n_tsubs = len(all_tsubs)
    
    best_selected = None
    best_count = float('inf')
    
    # Multi-restart
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        seed = 42 + restart * 7
        selected = _greedy_construct(M, n_tsubs, deadline - 10, seed)
        
        if not selected:
            continue
            
        selected = _local_optimize(selected, M, n_tsubs, deadline - 5, seed)
        
        # Verify
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected:
            cover_counts += M[idx, :]
            
        if np.all(cover_counts > 0):
            if len(selected) < best_count:
                best_count = len(selected)
                best_selected = list(selected)
                
    if best_selected is None:
        # Fallback
        selected = _greedy_construct(M, n_tsubs, deadline - 5, 42)
        best_selected = selected
        
    blocks = [all_kblocks[i] for i in best_selected]
    return blocks

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
