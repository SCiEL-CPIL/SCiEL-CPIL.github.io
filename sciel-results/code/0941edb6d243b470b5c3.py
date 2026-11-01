import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # Precompute: for each t-subset, which blocks cover it (for reverse lookup)
    # This is a sparse structure: tsub_idx -> list of block indices
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for i in range(n_blocks):
        for j in np.where(Cover[i] == 1)[0]:
            tsub_to_blocks[j].append(i)
    
    # 1. Greedy Initialization
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected = []
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    
    while np.any(uncovered) and time.time() < deadline - 25.0:
        gains = np.sum(Cover & uncovered, axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected.append(best_idx)
        cover_counts += Cover[best_idx]
        uncovered = (cover_counts == 0).astype(np.uint8)
        
    if np.any(uncovered):
        pass

    # 2. Local Optimization
    selected_set = set(selected)
    
    def is_redundant(idx, counts):
        mask = Cover[idx] == 1
        if not np.any(mask):
            return True
        return np.all(counts[mask] > 1)

    def remove_block(idx):
        nonlocal cover_counts, selected_set
        cover_counts -= Cover[idx]
        selected_set.remove(idx)

    def add_block(idx):
        nonlocal cover_counts, selected_set
        cover_counts += Cover[idx]
        selected_set.add(idx)

    # Phase 2a: Remove all redundant blocks
    changed = True
    while changed and time.time() < deadline - 15.0:
        changed = False
        to_remove = []
        for idx in list(selected_set):
            if is_redundant(idx, cover_counts):
                to_remove.append(idx)
        for idx in to_remove:
            remove_block(idx)
            changed = True

    # Phase 2b: Swap optimization with pre-filtering
    all_indices = list(range(n_blocks))
    
    # Build reverse index: for each block, list of t-subsets it covers
    block_to_tsubs = [np.where(Cover[i] == 1)[0] for i in range(n_blocks)]
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_blocks = list(selected_set)
        random.shuffle(current_blocks)
        
        for idx in current_blocks:
            if time.time() > deadline - 5.0:
                break
            
            if is_redundant(idx, cover_counts):
                remove_block(idx)
                improved = True
                continue
            
            critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
            critical_tsubs = np.where(critical_mask)[0]
            
            if len(critical_tsubs) == 0:
                remove_block(idx)
                improved = True
                continue
            
            # Pre-filter candidates: must cover ALL critical t-subsets
            # Use intersection of block lists for each critical t-subset
            candidate_sets = [set(tsub_to_blocks[c]) for c in critical_tsubs]
            # Intersect all candidate sets
            candidates = set.intersection(*candidate_sets) if candidate_sets else set()
            candidates -= selected_set  # exclude already selected
            
            if not candidates:
                continue
            
            # Score candidates
            low_count_mask = (cover_counts <= 2).astype(np.uint8)
            
            best_alt = -1
            best_score = -1
            
            for j in candidates:
                # Score: weighted sum of low-count coverage + bonus for covering other criticals
                score = np.sum(Cover[j] * low_count_mask)
                
                # Bonus: how many other blocks' critical subsets does j cover?
                # This is expensive, so use a proxy: count of t-subsets with count==1 that j covers
                other_critical = np.sum((Cover[j] == 1) & (cover_counts == 1))
                score += 0.1 * other_critical
                
                if score > best_score:
                    best_score = score
                    best_alt = j
            
            if best_alt != -1 and best_score > 0:
                cover_counts -= Cover[idx]
                selected_set.remove(idx)
                cover_counts += Cover[best_alt]
                selected_set.add(best_alt)
                improved = True
                
                # Quick redundancy pass after swap
                for k in list(selected_set):
                    if is_redundant(k, cover_counts):
                        remove_block(k)
                        improved = True

    # Final cleanup
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        to_remove = []
        for idx in list(selected_set):
            if is_redundant(idx, cover_counts):
                to_remove.append(idx)
        for idx in to_remove:
            remove_block(idx)
            changed = True

    blocks = [all_kblocks[i] for i in selected_set]
    
    return blocks

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
