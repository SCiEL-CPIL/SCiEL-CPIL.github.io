import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # Greedy
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    greedy_deadline = deadline - 40.0
    
    while np.sum(uncovered) > 0:
        if time.time() > greedy_deadline:
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)

    # Initial cover counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_indices:
        cover_counts += Cover[idx]

    def _local_optimize(selected_indices, cover_counts, deadline):
        # 1. Remove Redundant (1-0)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for idx in selected_indices:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    changed = True
            selected_indices = new_selected

        # 2. 1-1 Swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = []
            for idx in selected_indices:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected.append(idx)
                    continue
                
                cover_counts -= Cover[idx]
                uncovered_tsubs = np.where(cover_counts == 0)[0]
                
                if len(uncovered_tsubs) == 0:
                    improved = True
                    continue
                
                best_rep = -1
                best_gain = 0
                # Only try if uncovered set is small enough to be covered by 1 block
                # Max t-subsets in a k-block is C(k, t)
                if len(uncovered_tsubs) <= 35: 
                    for cand in range(n_blocks):
                        if cand in selected_indices:
                            continue
                        if np.all(Cover[cand, uncovered_tsubs] == 1):
                            gain = np.sum(Cover[cand] == 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_rep = cand
                    if best_rep >= 0:
                        cover_counts += Cover[best_rep]
                        new_selected.append(best_rep)
                        improved = True
                        continue
                
                cover_counts += Cover[idx]
                new_selected.append(idx)
            
            selected_indices = new_selected

        # 3. 2-1 Swaps (Remove 2, Add 1)
        # This is the key improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_sel = len(selected_indices)
            if n_sel < 2:
                break
            
            # We iterate through pairs. To avoid O(N^2) full scan every time, 
            # we can just do one pass and repeat if improved.
            for i in range(n_sel):
                if time.time() > deadline:
                    return selected_indices, cover_counts
                for j in range(i + 1, n_sel):
                    # Check if essential
                    # If either i or j is essential alone, we can't remove it without replacement
                    # But 2-1 swap removes both.
                    
                    # Tentatively remove i and j
                    cover_counts -= Cover[selected_indices[i]]
                    cover_counts -= Cover[selected_indices[j]]
                    
                    uncovered_tsubs = np.where(cover_counts == 0)[0]
                    
                    if len(uncovered_tsubs) == 0:
                        # Both redundant
                        # Remove from list
                        # Need to handle index shifting carefully
                        # It's safer to rebuild the list
                        new_selected = []
                        for idx in selected_indices:
                            if idx != selected_indices[i] and idx != selected_indices[j]:
                                new_selected.append(idx)
                        selected_indices = new_selected
                        improved = True
                        # Restart loops as indices changed
                        break
                    else:
                        # Try to find 1 block to cover them
                        if len(uncovered_tsubs) <= 35:
                            best_rep = -1
                            best_gain = 0
                            for cand in range(n_blocks):
                                if cand in selected_indices:
                                    continue
                                if np.all(Cover[cand, uncovered_tsubs] == 1):
                                    gain = np.sum(Cover[cand] == 1)
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_rep = cand
                            if best_rep >= 0:
                                # Replace i, j with best_rep
                                new_selected = []
                                for idx in selected_indices:
                                    if idx == selected_indices[i] or idx == selected_indices[j]:
                                        if idx == selected_indices[i]: # Only add once
                                            new_selected.append(best_rep)
                                        # skip the other
                                    else:
                                        new_selected.append(idx)
                                cover_counts += Cover[best_rep]
                                selected_indices = new_selected
                                improved = True
                                break
                        # If no replacement, restore
                        cover_counts += Cover[selected_indices[i]]
                        cover_counts += Cover[selected_indices[j]]
                
                if improved:
                    break # Break inner loop, restart outer or while
            
            if not improved:
                break

        # Final redundancy check
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for idx in selected_indices:
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_selected.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    changed = True
            selected_indices = new_selected

        return selected_indices, cover_counts

    # Initial Local Opt
    selected_indices, cover_counts = _local_optimize(selected_indices, cover_counts, deadline - 10.0)
    best_solution = list(selected_indices)
    best_count = len(best_solution)
    
    # ILS
    ils_deadline = deadline - 5.0
    random.seed(42)
    
    while time.time() < ils_deadline:
        perturbed = list(selected_indices)
        
        # Perturb: Remove 1 or 2 blocks randomly
        n_removals = random.choice([1, 2])
        if len(perturbed) < n_removals:
            break
            
        remove_indices = random.sample(range(len(perturbed)), n_removals)
        removed_blocks = [perturbed[i] for i in remove_indices]
        
        # Remove from perturbed
        for i in sorted(remove_indices, reverse=True):
            perturbed.pop(i)
            
        # Recompute counts
        pert_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in perturbed:
            pert_counts += Cover[idx]
            
        # Repair: Try to add blocks to cover holes
        # Greedy repair
        uncovered = np.where(pert_counts == 0)[0]
        while len(uncovered) > 0:
            if time.time() > ils_deadline:
                break
            # Find best block to cover most uncovered
            # Use only blocks not in perturbed
            gains = np.zeros(n_blocks, dtype=np.int32)
            # This is slow if done naively. 
            # Since we are in ILS, maybe just pick random candidates?
            # Or use the precomputed Cover matrix.
            # gains = Cover[:, uncovered].sum(axis=1)
            # But we must exclude blocks already in perturbed.
            
            # Optimization: If uncovered is small, check candidates directly
            if len(uncovered) <= 35:
                best_cand = -1
                best_gain = 0
                for cand in range(n_blocks):
                    if cand in perturbed:
                        continue
                    if np.all(Cover[cand, uncovered] == 1):
                        gain = np.sum(Cover[cand] == 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand
                if best_cand >= 0:
                    perturbed.append(best_cand)
                    pert_counts += Cover[best_cand]
                    uncovered = np.where(pert_counts == 0)[0]
                else:
                    # Cannot repair with 1 block? 
                    # If we removed 2, maybe we need 2?
                    # For simplicity, if we can't repair, abort this perturbation
                    perturbed = list(selected_indices)
                    pert_counts = cover_counts.copy()
                    break
            else:
                # Too many uncovered, use vectorized greedy
                gains = Cover[:, uncovered].sum(axis=1)
                # Mask out existing blocks
                for idx in perturbed:
                    gains[idx] = -1
                best_cand = int(np.argmax(gains))
                if gains[best_cand] <= 0:
                    perturbed = list(selected_indices)
                    pert_counts = cover_counts.copy()
                    break
                perturbed.append(best_cand)
                pert_counts += Cover[best_cand]
                uncovered = np.where(pert_counts == 0)[0]

        # Local Optimize
        if time.time() > ils_deadline - 2.0:
            break
            
        if np.any(pert_counts == 0):
            continue
            
        perturbed, pert_counts = _local_optimize(perturbed, pert_counts, ils_deadline - 1.0)
        
        if len(perturbed) < best_count:
            best_count = len(perturbed)
            best_solution = list(perturbed)
            selected_indices = list(perturbed)
            cover_counts = pert_counts.copy()

    return [all_kblocks[i] for i in best_solution]
