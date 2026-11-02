import itertools
import random
import time
import numpy as np
from collections import defaultdict
import heapq

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets (targets)
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        ts_list = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_ts.append(ts_list)

    # Precompute: for each t-subset index, which blocks contain it
    ts_to_blocks = defaultdict(list)
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # --- 1. Greedy Initialization with Heap ---
    def greedy_init(seed):
        local_rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        
        # gains[b] = number of currently uncovered t-subsets covered by block b
        gains = np.zeros(n_blocks, dtype=np.int32)
        for b in range(n_blocks):
            gains[b] = len(block_ts[b])
        
        # Heap of (-gain, block_index)
        heap = [(-gains[b], b) for b in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 5.0:
                break
            
            # Pop valid max gain
            while heap:
                neg_g, b = heap[0]
                g = -neg_g
                if g == gains[b] and gains[b] > 0:
                    heapq.heappop(heap)
                    break
                else:
                    heapq.heappop(heap)
            
            if not heap:
                break
                
            neg_g, best_b = heap[0]
            best_gain = -neg_g
            if best_gain <= 0:
                break
                
            heapq.heappop(heap)
            selected.append(best_b)
            
            # Update
            for s_idx in block_ts[best_b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    uncovered_count -= 1
                    for other_b in ts_to_blocks[s_idx]:
                        gains[other_b] -= 1
                        if gains[other_b] < 0:
                            gains[other_b] = 0
                        heapq.heappush(heap, (-gains[other_b], other_b))
        
        return selected

    # --- 2. Local Search Optimization ---
    def local_search_optimize(initial_blocks, seed, time_limit):
        local_rng = random.Random(seed)
        start_time = time.time()
        
        # State
        in_block = np.zeros(n_blocks, dtype=bool)
        coverage = np.zeros(n_ts, dtype=np.int32)
        
        selected_list = list(initial_blocks)
        for b in selected_list:
            in_block[b] = True
            for s in block_ts[b]:
                coverage[s] += 1
        
        n_uncovered = np.sum(coverage == 0)
        n_selected = len(selected_list)
        
        P = 10000
        def get_obj():
            return n_selected + P * n_uncovered
            
        current_obj = get_obj()
        best_obj = current_obj
        best_selected = selected_list.copy()
        
        # 1-Opt: Try removing a block
        def try_remove(b):
            nonlocal n_uncovered, n_selected, current_obj, best_obj, best_selected, in_block, coverage, selected_list
            if not in_block[b]:
                return False
            
            # Check if any set becomes uncovered
            new_uncovered = 0
            for s in block_ts[b]:
                if coverage[s] == 1:
                    new_uncovered += 1
            
            if new_uncovered == 0:
                # Accept removal
                in_block[b] = False
                selected_list.remove(b)
                n_selected -= 1
                for s in block_ts[b]:
                    coverage[s] -= 1
                current_obj -= 1
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_selected = selected_list.copy()
                return True
            return False

        # 1-Swap: Remove b1, Add b2
        def try_swap(b1, b2):
            nonlocal n_uncovered, n_selected, current_obj, best_obj, best_selected, in_block, coverage, selected_list
            if not in_block[b1] or in_block[b2]:
                return False
            
            # Calculate delta
            # Remove b1
            lost = 0
            for s in block_ts[b1]:
                if coverage[s] == 1:
                    lost += 1
                coverage[s] -= 1
            
            # Add b2
            gained = 0
            for s in block_ts[b2]:
                if coverage[s] == 0:
                    gained += 1
                coverage[s] += 1
            
            delta_uncovered = lost - gained
            # n_selected unchanged
            
            if delta_uncovered < 0:
                # Accept
                selected_list.remove(b1)
                selected_list.append(b2)
                in_block[b1] = False
                in_block[b2] = True
                n_uncovered += delta_uncovered
                current_obj += P * delta_uncovered
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_selected = selected_list.copy()
                return True
            else:
                # Revert
                for s in block_ts[b2]:
                    coverage[s] -= 1
                for s in block_ts[b1]:
                    coverage[s] += 1
                return False

        # Main loop
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Try removals
            for b in selected_list[:]:
                if try_remove(b):
                    improved = True
            
            # Try swaps
            # Randomized to avoid getting stuck
            local_rng.shuffle(selected_list)
            for b1 in selected_list:
                if time.time() - start_time > time_limit:
                    break
                # Try a few random b2
                for _ in range(10):
                    b2 = local_rng.randint(0, n_blocks - 1)
                    if try_swap(b1, b2):
                        improved = True
                        break
        
        # Ensure validity
        # Reconstruct coverage for best_selected
        best_cov = np.zeros(n_ts, dtype=np.int32)
        for b in best_selected:
            for s in block_ts[b]:
                best_cov[s] += 1
        
        if np.any(best_cov == 0):
            return initial_blocks
        else:
            return best_selected

    # --- 3. Repair ---
    def repair(blocks):
        coverage = np.zeros(n_ts, dtype=np.int32)
        in_block = set(blocks)
        
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
        
        uncovered = np.where(coverage == 0)[0]
        
        while len(uncovered) > 0 and time.time() - _start_time < _DEADLINE - 2.0:
            s = uncovered[0]
            candidates = ts_to_blocks[s]
            
            best_cand = -1
            best_gain = 0
            for b in candidates:
                if b in in_block:
                    continue
                gain = 0
                for ss in block_ts[b]:
                    if coverage[ss] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = b
            
            if best_cand == -1:
                break
                
            blocks.append(best_cand)
            in_block.add(best_cand)
            
            for ss in block_ts[best_cand]:
                coverage[ss] += 1
            
            uncovered = np.where(coverage == 0)[0]
            
        return blocks

    # --- 4. Polish ---
    def polish(blocks):
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
                
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 1.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # --- Main Execution ---
    
    # 1. Greedy Init
    initial_blocks = greedy_init(42)
    
    # 2. Local Search Optimization
    ls_time = _remaining() * 0.8
    if ls_time > 5:
        optimized_blocks = local_search_optimize(initial_blocks, 12345, ls_time)
    else:
        optimized_blocks = initial_blocks
    
    # 3. Repair
    optimized_blocks = repair(optimized_blocks)
    
    # 4. Polish
    optimized_blocks = polish(optimized_blocks)
    
    # 5. Convert to list of lists of ints
    result = [list(all_blocks[b]) for b in optimized_blocks]
    
    return result
