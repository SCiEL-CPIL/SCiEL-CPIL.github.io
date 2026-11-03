import itertools
import time
import random
import numpy as np
from collections import defaultdict
import heapq

def construct(v, k, t):
    deadline = time.time() + 115.0

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Bitset representation for fast intersection checks
    # tsub_mask[b] is a Python int where bit i is 1 if tsub i is covered by block b
    tsub_mask = [0] * n_blocks
    # tsub_blocks_mask[i] is a Python int where bit b is 1 if block b covers tsub i
    # n_blocks is ~50k, so this int is ~50k bits. Python handles this.
    tsub_blocks_mask = [0] * n_tsubsets

    for bi, block in enumerate(all_blocks):
        mask = 0
        tsubs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            tsubs.append(idx)
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << bi)
            tsub_blocks[idx].append(bi)
        block_tsubs[bi] = tsubs
        tsub_mask[bi] = mask

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[i] = number of uncovered t-subsets in block i
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        # Use a max-heap for efficient max-gain selection
        # Heap stores (-gain, tiebreaker, block_index)
        heap = [(-gain[bi], rng.randint(0, 10000), bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            
            # Pop from heap until we find a valid candidate
            best_bi = -1
            best_gain = -1
            
            while heap:
                neg_g, tie, bi = heapq.heappop(heap)
                current_gain = -neg_g
                if current_gain <= 0:
                    break
                if gain[bi] == current_gain and gain[bi] > 0:
                    best_bi = bi
                    best_gain = current_gain
                    break
            
            if best_bi == -1:
                break
            
            selected.append(best_bi)
            
            # Update gains and heap
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        # Push updated gain to heap
                        if gain[bj] > 0:
                            heapq.heappush(heap, (-gain[bj], rng.randint(0, 10000), bj))
                        
            gain[best_bi] = -1 # Prevent re-selection
            
        return selected

    def local_search(blocks, seed):
        rng = random.Random(seed)
        # blocks is a list of indices
        # We work with a set for O(1) lookup
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            # A block is redundant if all its t-subsets are covered by other blocks
            # i.e., cover_count[ts] > 1 for all ts in block
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed significantly
            
            # Phase 2: Try swaps
            # For each block, try to replace it with another that covers its unique subsets
            # and potentially allows further reduction
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Should have been caught in Phase 1, but just in case
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks that cover all unique_tsubs
                # Candidate must not be in block_set
                # Intersection of tsub_blocks[ts] for ts in unique_tsubs
                # Use bitsets for speed
                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                
                if cand_mask == 0:
                    continue
                    
                # Remove current block from candidates
                cand_mask &= ~(1 << bi)
                
                if cand_mask == 0:
                    continue
                
                # Extract candidates from mask
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 50:
                        break
                        
                if not candidates:
                    continue
                
                rng.shuffle(candidates)
                
                swapped = False
                for bj in candidates:
                    if bj in block_set:
                        continue
                        
                    # Perform swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    def two_block_swap_search(blocks, seed):
        """
        Try to replace two blocks (bi, bj) with a single block bk that covers
        all unique t-subsets of bi and bj. This is a 2-for-1 improvement.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Try pairs of blocks
            # To keep it fast, we limit the number of pairs we check per pass
            # or just iterate through shuffled blocks and pick a random partner
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Pick a random partner bj from the remaining blocks
                if len(block_set) < 2:
                    break
                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)
                
                n_checked += 1
                if n_checked > 8000: # Increase limit slightly for more search
                    break
                
                # Find unique t-subsets for bi and bj
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                
                set_bi = set(block_tsubs[bi])
                set_bj = set(block_tsubs[bj])
                
                required = set()
                for ts in req_bi:
                    required.add(ts)
                for ts in req_bj:
                    required.add(ts)
                
                # Check intersection
                inter = set_bi & set_bj
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)
                
                if not required:
                    # Both blocks are redundant? Phase 1 should have handled this.
                    continue
                
                # Pick ts with fewest covering blocks
                min_ts = min(required, key=lambda x: len(tsub_blocks[x]))
                
                # Use bitset intersection for candidates
                cand_mask = tsub_blocks_mask[min_ts]
                for ts in required:
                    if ts == min_ts:
                        continue
                    cand_mask &= tsub_blocks_mask[ts]
                    if cand_mask == 0:
                        break
                        
                if cand_mask == 0:
                    continue
                
                # Remove bi and bj from candidates
                cand_mask &= ~(1 << bi)
                cand_mask &= ~(1 << bj)
                
                if cand_mask == 0:
                    continue
                
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 15:
                        break
                        
                if not candidates:
                    continue
                
                rng.shuffle(candidates)
                
                swapped = False
                for bk in candidates:
                    if bk in block_set:
                        continue
                        
                    # Perform swap: remove bi, bj; add bk
                    block_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                        
                    block_set.remove(bj)
                    for ts in block_tsubs[bj]:
                        cover_count[ts] -= 1
                        
                    block_set.add(bk)
                    for ts in block_tsubs[bk]:
                        cover_count[ts] += 1
                        
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    def simulated_annealing(blocks, seed):
        """
        Simulated Annealing for covering design.
        State: set of block indices.
        Energy: number of blocks (cost) + penalty * uncovered_tsubsets.
        Moves:
        1. Add a random block.
        2. Remove a random block (if it doesn't uncover tsubsets, or with penalty).
        3. Swap a random block with another random block.
        4. 2-for-1 swap (remove 2, add 1) if valid.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        # Calculate initial energy
        def get_energy():
            uncovered = 0
            for c in cover_count:
                if c == 0:
                    uncovered += 1
            return len(block_set) + 100 * uncovered # High penalty for uncovered
            
        current_energy = get_energy()
        best_energy = current_energy
        best_blocks = list(block_set)
        
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        while T > T_min and time.time() < deadline - 2:
            # Pick a move
            move_type = rng.randint(0, 2)
            new_block_set = block_set.copy()
            delta_energy = 0
            
            if move_type == 0: # Add
                if len(block_set) > 0:
                    bi = rng.choice(list(block_set))
                    # Find a random block NOT in set
                    # This is expensive to do naively. 
                    # Instead, pick a random block from all_blocks, if not in set.
                    # If in set, retry a few times.
                    for _ in range(10):
                        new_bi = rng.randint(0, n_blocks - 1)
                        if new_bi not in block_set:
                            break
                    else:
                        continue # Couldn't find a new block? Unlikely unless set is full.
                    
                    # Calculate delta
                    # Adding new_bi increases cover_count for its tsubs.
                    # It reduces uncovered count if it covers any uncovered tsub.
                    # It increases block count by 1.
                    for ts in block_tsubs[new_bi]:
                        if cover_count[ts] == 0:
                            delta_energy -= 100 # Uncovered -> Covered
                    delta_energy += 1 # Block count +1
                    
                    if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                        block_set.add(new_bi)
                        for ts in block_tsubs[new_bi]:
                            cover_count[ts] += 1
                        current_energy += delta_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = list(block_set)
                            
            elif move_type == 1: # Remove
                if len(block_set) > 0:
                    bi = rng.choice(list(block_set))
                    # Calculate delta
                    # Removing bi decreases cover_count for its tsubs.
                    # It increases uncovered count if it was the only cover for any tsub.
                    # It decreases block count by 1.
                    for ts in block_tsubs[bi]:
                        if cover_count[ts] == 1:
                            delta_energy += 100 # Covered -> Uncovered
                    delta_energy -= 1 # Block count -1
                    
                    if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                        block_set.remove(bi)
                        for ts in block_tsubs[bi]:
                            cover_count[ts] -= 1
                        current_energy += delta_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = list(block_set)
                            
            else: # Swap
                if len(block_set) > 0:
                    bi = rng.choice(list(block_set))
                    # Pick random new block
                    for _ in range(10):
                        new_bi = rng.randint(0, n_blocks - 1)
                        if new_bi not in block_set:
                            break
                    else:
                        continue
                    
                    # Calculate delta
                    # Remove bi, Add new_bi.
                    # Block count change: 0.
                    # Uncovered change:
                    # For ts in bi: if count==1 and not in new_bi -> +100
                    # For ts in new_bi: if count==0 -> -100 (but wait, if ts in bi and new_bi, count changes differently)
                    
                    # Let's just compute energy before and after? No, that's O(N).
                    # Delta calculation:
                    # delta_uncovered = 0
                    # set_bi = set(block_tsubs[bi])
                    # set_new = set(block_tsubs[new_bi])
                    
                    # ts in bi only: count decreases by 1. If count was 1, becomes 0. +100
                    # ts in new only: count increases by 1. If count was 0, becomes 1. -100
                    # ts in both: count unchanged.
                    
                    set_bi = set(block_tsubs[bi])
                    set_new = set(block_tsubs[new_bi])
                    
                    for ts in set_bi - set_new:
                        if cover_count[ts] == 1:
                            delta_energy += 100
                    for ts in set_new - set_bi:
                        if cover_count[ts] == 0:
                            delta_energy -= 100
                            
                    if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                        block_set.remove(bi)
                        for ts in block_tsubs[bi]:
                            cover_count[ts] -= 1
                        block_set.add(new_bi)
                        for ts in block_tsubs[new_bi]:
                            cover_count[ts] += 1
                        current_energy += delta_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_blocks = list(block_set)
                            
            T *= cooling
            
        return best_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 35
    for i in range(num_trials):
        if time.time() > deadline - 35:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search (single swaps)
        blocks = local_search(blocks, i)
        
        # Two-block swap search
        if time.time() < deadline - 20:
            blocks = two_block_swap_search(blocks, i + 1000)
            
        # Simulated Annealing Polish
        if time.time() < deadline - 12:
            blocks = simulated_annealing(blocks, i + 2000)
            # Quick local search after SA to clean up
            blocks = local_search(blocks, i + 3000)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # If we have time, try to improve the best solution further
    if best_blocks is not None and time.time() < deadline - 8:
        best_blocks = local_search(best_blocks, 99999)
        if time.time() < deadline - 4:
            best_blocks = two_block_swap_search(best_blocks, 88888)
        if time.time() < deadline - 2:
            best_blocks = simulated_annealing(best_blocks, 77777)
            best_blocks = local_search(best_blocks, 66666)

    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
