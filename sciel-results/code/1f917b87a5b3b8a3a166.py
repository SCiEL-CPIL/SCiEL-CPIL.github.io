import itertools
import time
import random
import numpy as np
import heapq
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build block_tsubs: for each block, list of t-subset indices it covers
    # Also build tsub_blocks: for each t-subset, list of block indices that cover it
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            tsubs.append(idx)
            tsub_blocks[idx].append(bi)
        block_tsubs[bi] = tsubs
    
    # Precompute sets for faster intersection in local search
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    
    # Build the incidence matrix A (sparse) for MILP
    row_indices = []
    col_indices = []
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            row_indices.append(ts_idx)
            col_indices.append(bi)
    data = np.ones(len(row_indices), dtype=np.float64)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_tsubsets, n_blocks))
    
    best_blocks = None
    best_indices = None
    best_len = float('inf')
    
    def greedy_cover(seed=0):
        """
        Greedy set cover using a max-heap with lazy deletion.
        """
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
        
        # Max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Pop until we find a valid (non-stale) entry with positive gain
            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
            else:
                break
            
            if max_gain <= 0:
                break
            
            selected.append(best_bi)
            
            # Update gains for all blocks that share t-subsets with best_bi
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        heapq.heappush(heap, (-gain[bj], bj))
            
            # Mark best_bi as used
            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))
        
        return selected
    
    def remove_redundant(block_set, cover_count):
        """Remove redundant blocks, updating cover_count. Returns list of removed blocks."""
        to_remove = []
        for bi in list(block_set):
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)
        
        for bi in to_remove:
            block_set.discard(bi)
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] -= 1
        
        return to_remove
    
    def local_search(blocks, seed, max_rounds=50):
        """
        Aggressive local search: swap-then-remove cycles.
        """
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
        
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds and time.time() < deadline - 10:
            improved = False
            rounds += 1
            
            # Phase 1: Remove redundant blocks
            removed = remove_redundant(block_set, cover_count)
            if removed:
                improved = True
                continue
            
            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 10:
                    break
                if bi not in block_set:
                    continue
                
                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                
                if not unique_tsubs:
                    # This block is redundant (should have been caught, but just in case)
                    block_set.discard(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                
                # Find candidate blocks that cover ALL unique_tsubs
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                
                candidates.discard(bi)
                
                if not candidates:
                    continue
                
                # Try candidates - limit to a reasonable number
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:30]:
                    if bj in block_set:
                        continue
                    
                    # Perform swap
                    block_set.discard(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                    
                    improved = True
                    swapped = True
                    break
                
                if swapped:
                    break
            
            # Phase 3: Try 2-for-1 swaps (limited)
            if not improved and len(block_set) > 2:
                cover_count = [0] * n_tsubsets
                for bi in block_set:
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] += 1
                
                current_blocks = list(block_set)
                rng.shuffle(current_blocks)
                
                n_pairs_to_check = 50
                pairs_checked = 0
                indices = list(range(len(current_blocks)))
                
                for idx_i in range(len(indices)):
                    if time.time() > deadline - 10 or pairs_checked >= n_pairs_to_check:
                        break
                    i = indices[idx_i]
                    for idx_j in range(idx_i + 1, len(indices)):
                        if time.time() > deadline - 10 or pairs_checked >= n_pairs_to_check:
                            break
                        j = indices[idx_j]
                        pairs_checked += 1
                        
                        bi1 = current_blocks[i]
                        bi2 = current_blocks[j]
                        if bi1 not in block_set or bi2 not in block_set:
                            continue
                        
                        tsubs_bi1 = set(block_tsubs[bi1])
                        tsubs_bi2 = set(block_tsubs[bi2])
                        union = tsubs_bi1 | tsubs_bi2
                        unique_to_pair = []
                        for ts in union:
                            delta = (1 if ts in tsubs_bi1 else 0) + (1 if ts in tsubs_bi2 else 0)
                            if cover_count[ts] - delta == 0:
                                unique_to_pair.append(ts)
                        
                        if not unique_to_pair:
                            continue
                        
                        unique_to_pair_sorted = sorted(unique_to_pair, key=lambda x: len(tsub_blocks[x]))
                        candidates = set(tsub_block_sets[unique_to_pair_sorted[0]])
                        for ts_idx in unique_to_pair_sorted[1:]:
                            candidates &= tsub_block_sets[ts_idx]
                            if not candidates:
                                break
                        
                        candidates -= block_set
                        if not candidates:
                            continue
                        
                        cand_list = list(candidates)
                        rng.shuffle(cand_list)
                        
                        for new_bi in cand_list[:10]:
                            new_set = set(block_set)
                            new_set.discard(bi1)
                            new_set.discard(bi2)
                            new_set.add(new_bi)
                            block_set = new_set
                            improved = True
                            break
                        
                        if improved:
                            break
                    
                    if improved:
                            break
            
            if not improved:
                break
        
        # Final cleanup
        remove_redundant(block_set, cover_count)
        
        return list(block_set)
    
    def intensive_polish(block_indices, seed, max_rounds=100):
        """
        Intensive polish with more swap rounds and 2-for-1 emphasis.
        """
        if not block_indices:
            return block_indices
        
        rng = random.Random(seed)
        block_set = set(block_indices)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
        
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            
            # Remove redundant
            removed = remove_redundant(block_set, cover_count)
            if removed:
                improved = True
                continue
            
            # Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 5:
                    break
                if bi not in block_set:
                    continue
                
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi] if cover_count[ts_idx] == 1]
                if not unique_tsubs:
                    block_set.discard(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break
                
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                for ts_idx in unique_tsubs[1:]:
                    candidates &= tsub_block_sets[ts_idx]
                    if not candidates:
                        break
                
                candidates.discard(bi)
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for bj in cand_list[:50]:
                    if bj in block_set:
                        continue
                    
                    block_set.discard(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                    
                    improved = True
                    break
                
                if improved:
                    break
            
            # Try 2-for-1 swaps
            if not improved and len(block_set) > 2:
                cover_count = [0] * n_tsubsets
                for bi in block_set:
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] += 1
                
                current_blocks = list(block_set)
                rng.shuffle(current_blocks)
                
                n_pairs = 100
                pairs_checked = 0
                indices = list(range(len(current_blocks)))
                
                for idx_i in range(len(indices)):
                    if time.time() > deadline - 5 or pairs_checked >= n_pairs:
                        break
                    i = indices[idx_i]
                    for idx_j in range(idx_i + 1, len(indices)):
                        if time.time() > deadline - 5 or pairs_checked >= n_pairs:
                            break
                        j = indices[idx_j]
                        pairs_checked += 1
                        
                        bi1 = current_blocks[i]
                        bi2 = current_blocks[j]
                        if bi1 not in block_set or bi2 not in block_set:
                            continue
                        
                        tsubs_bi1 = set(block_tsubs[bi1])
                        tsubs_bi2 = set(block_tsubs[bi2])
                        union = tsubs_bi1 | tsubs_bi2
                        unique_to_pair = []
                        for ts in union:
                            delta = (1 if ts in tsubs_bi1 else 0) + (1 if ts in tsub_blocks[ts_idx] and False else 0)
                            # Simplified: check if removing both would leave ts uncovered
                            delta = (1 if ts in tsubs_bi1 else 0) + (1 if ts in tsubs_bi2 else 0)
                            if cover_count[ts] - delta == 0:
                                unique_to_pair.append(ts)
                        
                        if not unique_to_pair:
                            continue
                        
                        unique_to_pair_sorted = sorted(unique_to_pair, key=lambda x: len(tsub_blocks[x]))
                        candidates = set(tsub_block_sets[unique_to_pair_sorted[0]])
                        for ts_idx in unique_to_pair_sorted[1:]:
                            candidates &= tsub_block_sets[ts_idx]
                            if not candidates:
                                break
                        
                        candidates -= block_set
                        if not candidates:
                            continue
                        
                        cand_list = list(candidates)
                        rng.shuffle(cand_list)
                        
                        for new_bi in cand_list[:15]:
                            new_set = set(block_set)
                            new_set.discard(bi1)
                            new_set.discard(bi2)
                            new_set.add(new_bi)
                            block_set = new_set
                            improved = True
                            break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        remove_redundant(block_set, cover_count)
        return list(block_set)
    
    # Strategy: Multiple Greedy + Local Search
    num_trials = 40
    for seed in range(num_trials):
        if time.time() > deadline - 30:
            break
        
        blocks_idx = greedy_cover(seed)
        if not blocks_idx:
            continue
        
        # Apply local search
        blocks_idx = local_search(blocks_idx, seed, max_rounds=30)
        
        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_indices = blocks_idx
            best_blocks = [all_blocks[i] for i in blocks_idx]
            
            # If we've found a good solution, do intensive polish
            if best_len <= 148 and time.time() < deadline - 15:
                polished = intensive_polish(blocks_idx, seed + 1000, max_rounds=80)
                if len(polished) < best_len:
                    best_len = len(polished)
                    best_indices = polished
                    best_blocks = [all_blocks[i] for i in polished]
        
        if best_len <= 143:
            break
    
    # Try MILP with warm start if we have time
    if best_indices and time.time() < deadline - 15:
        c = np.ones(n_blocks, dtype=np.float64)
        lb = np.ones(n_tsubsets, dtype=np.float64)
        ub = np.full(n_tsubsets, np.inf, dtype=np.float64)
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n_blocks, dtype=np.int8)
        bounds = Bounds(0, 1)
        
        x0 = np.zeros(n_blocks)
        for i in best_indices:
            x0[i] = 1.0
        
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                options={'time_limit': max(1, int(deadline - time.time() - 5)), 'mip_gap': 0.005}
            )
            
            if result.status in (0, 1):
                x = result.x
                selected_idx = [i for i in range(n_blocks) if x[i] > 0.5]
                if selected_idx and len(selected_idx) < best_len:
                    best_len = len(selected_idx)
                    best_indices = selected_idx
                    best_blocks = [all_blocks[i] for i in selected_idx]
        except Exception:
            pass
    
    if best_blocks is None:
        # Fallback
        blocks_idx = greedy_cover(0)
        blocks_idx = local_search(blocks_idx, 0, max_rounds=20)
        best_blocks = [all_blocks[i] for i in blocks_idx]
    
    return best_blocks
