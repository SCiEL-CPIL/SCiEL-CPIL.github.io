import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    
    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
        
    # Build reverse index: tsub -> list of blocks
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for j in range(num_kblocks):
        for idx in block_tsub_idxs[j]:
            tsub_to_blocks[idx].append(j)

    def polish(selected):
        if not selected:
            return selected
            
        counts = np.zeros(num_tsubs, dtype=np.int8)
        for b_idx in selected:
            for idx in block_tsub_idxs[b_idx]:
                counts[idx] += 1
                
        changed = True
        while changed and time.monotonic() < deadline - 5.0:
            changed = False
            new_selected = []
            for b_idx in selected:
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected = new_selected
            if not changed:
                break
                
        return selected

    def greedy_construct_lazy():
        """
        Greedy Set Cover with lazy evaluation using a priority queue.
        This is more efficient than recalculating gains for all blocks from scratch.
        """
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        selected_set = set()
        
        # Priority queue: (-gain, block_index)
        # We use a max-heap via negative gains.
        # Initial gains: all blocks have gain = len(block_tsub_idxs[j]) which is C(k,t) = 35.
        # But we can start with a heap of all blocks.
        # However, maintaining the heap is tricky because gains change.
        # Standard "lazy" greedy:
        # 1. Push all blocks with initial gain.
        # 2. Pop top. If gain matches current calculated gain, select it.
        #    Else, update gain and push back.
        
        # To speed up, we don't push all 50k blocks initially if we can be smarter?
        # No, lazy eval requires them to be in the heap to be considered.
        # But 50k items in a heap is fine.
        
        import heapq
        
        # Initial heap: (-35, j) for all j?
        # Actually, we can just iterate all blocks to find the best one initially?
        # No, the point is to avoid scanning all blocks every step.
        # But we need to initialize the heap.
        
        heap = []
        for j in range(num_kblocks):
            heapq.heappush(heap, (-len(block_tsub_idxs[j]), j))
            
        while time.monotonic() < deadline - 10.0:
            if np.all(covered):
                break
                
            best_gain = -1
            best_block = -1
            
            # Find the best valid block
            while heap:
                neg_gain, j = heapq.heappop(heap)
                current_gain = -neg_gain
                
                # Calculate actual gain
                actual_gain = 0
                for idx in block_tsub_idxs[j]:
                    if not covered[idx]:
                        actual_gain += 1
                        if actual_gain > current_gain:
                            # This shouldn't happen if heap is consistent, but just in case
                            # Actually, gain can only decrease. So actual_gain <= current_gain.
                            pass
                
                if actual_gain == current_gain and actual_gain > 0:
                    best_gain = actual_gain
                    best_block = j
                    break
                else:
                    # Gain changed, push back with new gain
                    if actual_gain > 0:
                        heapq.heappush(heap, (-actual_gain, j))
                    # If actual_gain == 0, discard it (it won't help anymore)
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            selected_set.add(best_block)
            
            # Update covered
            for idx in block_tsub_idxs[best_block]:
                if not covered[idx]:
                    covered[idx] = True
        
        return selected

    def local_search(selected):
        best_selected = polish(selected[:])
        best_len = len(best_selected)
        
        current_selected = best_selected[:]
        
        no_improve = 0
        max_no_improve = 20 # Reduced to allow more restarts? Or keep?
        
        while time.monotonic() < deadline - 2.0 and no_improve < max_no_improve:
            candidate = current_selected[:]
            
            if candidate:
                # Perturb: Remove a random block
                rem_idx = random.randint(0, len(candidate) - 1)
                candidate.pop(rem_idx)
                
                # Repair with greedy
                covered = np.zeros(num_tsubs, dtype=bool)
                for b_idx in candidate:
                    for idx in block_tsub_idxs[b_idx]:
                        covered[idx] = True
                
                # Use a fast greedy repair
                # We can reuse the lazy greedy logic but it's heavy.
                # Let's do a simple greedy repair:
                # Find best block to add.
                # To speed up, we can just check a few random blocks or the best from a small sample?
                # Or just do 1-2 steps of standard greedy.
                
                # Let's do standard greedy repair for up to 3 steps.
                for _ in range(3):
                    if np.all(covered):
                        break
                    best_gain = -1
                    best_block = -1
                    
                    # Sample some blocks? Or all?
                    # All is 50k. 3 * 50k = 150k ops. Fast enough.
                    for j in range(num_kblocks):
                        if j in candidate: # Slow check. Use set?
                            continue
                        gain = 0
                        for idx in block_tsub_idxs[j]:
                            if not covered[idx]:
                                gain += 1
                                if gain > best_gain:
                                    if gain == 35: break
                        if gain > best_gain:
                            best_gain = gain
                            best_block = j
                            
                    if best_block == -1 or best_gain == 0:
                        break
                    candidate.append(best_block)
                    for idx in block_tsub_idxs[best_block]:
                        covered[idx] = True

            candidate = polish(candidate)
            
            if len(candidate) < best_len:
                best_selected = candidate[:]
                best_len = len(candidate)
                current_selected = candidate[:]
                no_improve = 0
            else:
                no_improve += 1
                current_selected = best_selected[:]
                
        return best_selected

    # Main Loop
    # 1. Initial Greedy (Lazy)
    initial = greedy_construct_lazy()
    initial = polish(initial)
    
    # 2. Local Search
    best = local_search(initial)
    
    # 3. Restarts
    # The lazy greedy is deterministic. To get diversity, we need randomness.
    # We can add randomness by perturbing the initial heap or picking randomly from top-k.
    # For now, let's just run a few more instances of the lazy greedy?
    # They will be identical.
    
    # Let's implement a randomized version of the lazy greedy or just use the top-k greedy from before?
    # The top-k greedy was slow.
    
    # Let's try a different restart:
    # Start with a random subset of blocks? No.
    
    # Let's stick to one good run and local search.
    # If time permits, try a different initial construction.
    
    # Alternative: Randomized Greedy where we pick from top 5% of gains.
    # This requires calculating all gains, which is slow.
    
    # Let's just do a few more local search rounds from the same start? No, it's deterministic.
    
    # Let's try to implement a fast randomized greedy.
    # Idea: Pick a random uncovered t-subset. Pick a random block covering it.
    # This is very fast but might yield worse solutions.
    
    def greedy_random_tsub():
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        # List of uncovered tsubs
        uncovered_list = list(range(num_tsubs))
        
        while uncovered_list and time.monotonic() < deadline - 10.0:
            # Pick random uncovered tsub
            idx_pos = random.randint(0, len(uncovered_list) - 1)
            tsub_idx = uncovered_list[idx_pos]
            
            # Pick random block covering it
            blocks = tsub_to_blocks[tsub_idx]
            if not blocks:
                break
            block_idx = blocks[random.randint(0, len(blocks) - 1)]
            
            selected.append(block_idx)
            
            # Update uncovered
            new_uncovered = []
            for u_idx in uncovered_list:
                if u_idx == tsub_idx:
                    continue
                # Check if covered by new block
                # This is slow if we check every time.
                # Better: mark covered, and filter at the end?
                pass
            
            # More efficient update:
            # Mark covered
            for idx in block_tsub_idxs[block_idx]:
                covered[idx] = True
            
            # Filter uncovered_list
            # This is O(N) per step. 150 * 3876 = ~500k ops. Fine.
            new_uncovered = [u for u in uncovered_list if not covered[u]]
            uncovered_list = new_uncovered
            
        return selected

    # Run a few restarts with random tsub greedy
    for i in range(5):
        if time.monotonic() > deadline - 20.0:
            break
        candidate = greedy_random_tsub()
        candidate = polish(candidate)
        candidate = local_search(candidate)
        
        if len(candidate) < len(best):
            best = candidate
            
    # Final polish
    best = polish(best)
    
    # Convert indices to lists
    result_blocks = [list(all_kblocks[i]) for i in best]
    
    return result_blocks
