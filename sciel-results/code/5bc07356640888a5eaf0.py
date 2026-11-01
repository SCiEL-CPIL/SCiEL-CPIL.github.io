"""Covering design C(v,k,t) using core-periphery decomposition:
1. Build a core via greedy packing (minimize 4-way collisions)
2. Repair residual gaps with targeted block selection
3. Local optimization with block swaps"""
import time
import itertools
import numpy as np
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Precompute all k-blocks and t-subsets
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # Core-Periphery decomposition
    # Phase 1: Build core via greedy packing (minimize collisions)
    core_indices = _build_core_packing(Cover, n_tsubs, deadline - 30.0)
    
    # Compute which t-subsets are covered by the core
    core_covered = np.zeros(n_tsubs, dtype=np.uint8)
    for idx in core_indices:
        core_covered |= Cover[idx]
    
    # Phase 2: Repair residual gaps
    uncovered_mask = (core_covered == 0)
    n_uncovered = int(np.sum(uncovered_mask))
    
    if n_uncovered > 0 and time.time() < deadline - 15.0:
        repair_indices = _repair_gaps(uncovered_mask, Cover, n_tsubs, deadline - 10.0)
        core_indices = core_indices + repair_indices
    
    # Remove duplicates
    core_indices = list(set(core_indices))
    
    # Phase 3: Local optimization
    if time.time() < deadline - 5.0:
        core_indices = _local_optimize(core_indices, Cover, all_kblocks, n_tsubs, deadline - 3.0)
    
    blocks = [all_kblocks[i] for i in core_indices]
    return blocks


def _build_core_packing(Cover, n_tsubs, deadline):
    """Build a core of blocks that minimizes 4-way collisions (maximizes packing efficiency).
    
    Greedily select blocks that cover the most NEW t-subsets while minimizing
    overlap with already-selected blocks. This creates a 'packing' where each
    block contributes uniquely to the coverage."""
    selected = []
    covered = np.zeros(n_tsubs, dtype=np.uint8)
    n_blocks = Cover.shape[0]
    
    while time.time() < deadline - 5.0:
        # Compute gain: new t-subsets covered by each block
        gains = np.sum(Cover & (~covered).astype(np.uint8), axis=1)
        
        # Also compute collision penalty: how many t-subsets are already covered
        collisions = np.sum(Cover & covered, axis=1)
        
        # Score: prefer high gain, low collision
        # Use a smooth surrogate: score = gain - alpha * collisions
        alpha = 0.5
        scores = gains.astype(np.float64) - alpha * collisions.astype(np.float64)
        
        best_idx = int(np.argmax(scores))
        
        if gains[best_idx] == 0:
            break
        
        selected.append(best_idx)
        covered |= Cover[best_idx]
        
        # Early stop if all covered
        if np.all(covered == 1):
            break
    
    return selected


def _repair_gaps(uncovered_mask, Cover, n_tsubs, deadline):
    """Repair residual uncovered t-subsets with targeted block selection.
    
    This is a sparse repair problem: only need to cover the specific gaps.
    Use greedy set cover on just the uncovered t-subsets."""
    n_blocks = Cover.shape[0]
    remaining_gaps = uncovered_mask.astype(np.uint8)
    selected = []
    
    while np.sum(remaining_gaps) > 0 and time.time() < deadline - 3.0:
        # Compute gains only for uncovered t-subsets
        gains = np.sum(Cover & remaining_gaps, axis=1)
        
        best_idx = int(np.argmax(gains))
        
        if gains[best_idx] == 0:
            break
        
        selected.append(best_idx)
        remaining_gaps = remaining_gaps & (Cover[best_idx] == 0)
    
    return selected


def _local_optimize(block_indices, Cover, all_kblocks, n_tsubs, deadline):
    """Remove redundant blocks and attempt block replacements."""
    if not block_indices:
        return block_indices
    
    # Compute coverage counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in block_indices:
        cover_counts += Cover[idx]
    
    # Iteratively remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        new_block_indices = []
        for idx in block_indices:
            # Check if this block covers any t-subset with count 1
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_block_indices.append(idx)
            else:
                cover_counts -= Cover[idx]
                changed = True
        block_indices = new_block_indices
    
    # Try block replacement
    if time.time() < deadline - 2.0:
        block_indices = _try_block_replacement(block_indices, Cover, n_tsubs, cover_counts, deadline)
    
    return block_indices


def _try_block_replacement(block_indices, Cover, n_tsubs, cover_counts, deadline):
    """Try to replace blocks with alternatives that maintain coverage."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_block_indices = []
        for idx in block_indices:
            # Find critical t-subsets covered by this block (count == 1)
            critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
            critical_tsubs = np.where(critical_mask)[0]
            
            if len(critical_tsubs) == 0:
                # No critical t-subsets; this block is redundant
                cover_counts -= Cover[idx]
                changed = True
                continue
            
            # Find alternative blocks that cover ALL critical t-subsets
            candidate_scores = np.sum(Cover[:, critical_tsubs], axis=1)
            valid_candidates = candidate_scores == len(critical_tsubs)
            valid_candidates[idx] = False
            
            if not np.any(valid_candidates):
                new_block_indices.append(idx)
                continue
            
            # Among valid candidates, pick one that covers the most low-count t-subsets
            low_count_mask = (cover_counts <= 2).astype(np.float64)
            candidate_low_score = np.sum(Cover * low_count_mask, axis=1)
            
            valid_indices = np.where(valid_candidates)[0]
            best_alt = valid_indices[np.argmax(candidate_low_score[valid_indices])]
            
            # Replace
            cover_counts -= Cover[idx]
            cover_counts += Cover[best_alt]
            new_block_indices.append(best_alt)
            changed = True
        
        block_indices = new_block_indices
    
    return block_indices


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
