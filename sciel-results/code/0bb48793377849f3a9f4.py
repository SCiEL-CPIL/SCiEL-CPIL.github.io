import itertools
import random
import time
import numpy as np
import heapq
from collections import defaultdict

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

    # Helper: Get coverage array for a list of blocks
    def get_coverage(blocks_list):
        cov = np.zeros(n_ts, dtype=np.int32)
        for b in blocks_list:
            for s in block_ts[b]:
                cov[s] += 1
        return cov

    # Helper: Check if a block is redundant
    def is_redundant(b, cov):
        for s in block_ts[b]:
            if cov[s] == 1:
                return False
        return True

    # Helper: Greedy repair
    def greedy_repair(blocks, cov, time_limit):
        uncovered_idx = np.where(cov == 0)[0]
        if len(uncovered_idx) == 0:
            return blocks, cov, True

        candidates = set()
        for s in uncovered_idx:
            candidates.update(ts_to_blocks[s])
        
        # Remove blocks already in the solution from candidates
        in_block = set(blocks)
        candidates -= in_block
        
        if not candidates:
            return blocks, cov, False

        heap = []
        for b in candidates:
            gain = 0
            for s in block_ts[b]:
                if cov[s] == 0:
                    gain += 1
            if gain > 0:
                heapq.heappush(heap, (-gain, b))
                
        while heap and _remaining() > time_limit:
            neg_gain, b = heapq.heappop(heap)
            gain = -neg_gain
            
            curr_gain = 0
            for s in block_ts[b]:
                if cov[s] == 0:
                    curr_gain += 1
            
            if curr_gain != gain:
                if curr_gain > 0:
                    heapq.heappush(heap, (-curr_gain, b))
                continue
            
            if curr_gain == 0:
                break
            
            blocks.append(b)
            in_block.add(b)
            
            for s in block_ts[b]:
                cov[s] += 1
            
            if np.all(cov > 0):
                break

        if np.all(cov > 0):
            return blocks, cov, True
        else:
            return blocks, cov, False

    def greedy_init(seed):
        local_rng = random.Random(seed)
        cov = np.zeros(n_ts, dtype=np.int32)
        blocks = []
        in_block = set()
        
        while np.any(cov == 0) and _remaining() > 5.0:
            best_b = -1
            best_gain = 0
            for b in range(n_blocks):
                if b in in_block:
                    continue
                gain = 0
                for s in block_ts[b]:
                    if cov[s] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
            blocks.append(best_b)
            in_block.add(best_b)
            for s in block_ts[best_b]:
                cov[s] += 1
                
        return blocks, cov

    def remove_redundant(blocks, cov):
        changed = True
        while changed and _remaining() > 1.0:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                b = blocks[i]
                if is_redundant(b, cov):
                    blocks.pop(i)
                    for s in block_ts[b]:
                        cov[s] -= 1
                    changed = True
                    break
        return blocks, cov

    best_blocks = None
    best_size = float('inf')
    
    # 1. Multi-start Greedy
    num_starts = 5
    for i in range(num_starts):
        if _remaining() < 5.0:
            break
        blocks, cov = greedy_init(42 + i)
        blocks, cov = remove_redundant(blocks, cov)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    # 2. ILS
    if best_blocks is not None and _remaining() > 10.0:
        current_blocks = best_blocks[:]
        current_cov = get_coverage(current_blocks)
        
        if np.any(current_cov == 0):
            current_blocks, current_cov, _ = greedy_repair(current_blocks, current_cov, _remaining() * 0.1)
        
        while _remaining() > 5.0:
            if not current_blocks:
                break
            
            prev_blocks = current_blocks[:]
            prev_cov = current_cov.copy()
            
            idx = random.randint(0, len(current_blocks) - 1)
            removed_b = current_blocks.pop(idx)
            
            for s in block_ts[removed_b]:
                current_cov[s] -= 1
                
            new_blocks, new_cov, ok = greedy_repair(current_blocks, current_cov, _remaining() * 0.05)
            
            if ok:
                new_blocks, new_cov = remove_redundant(new_blocks, new_cov)
                
                if len(new_blocks) < best_size:
                    best_size = len(new_blocks)
                    best_blocks = new_blocks[:]
                    
                if len(new_blocks) < len(prev_blocks):
                    current_blocks = new_blocks
                    current_cov = new_cov
                elif len(new_blocks) == len(prev_blocks) and random.random() < 0.5:
                    current_blocks = new_blocks
                    current_cov = new_cov
                else:
                    current_blocks = prev_blocks
                    current_cov = prev_cov
            else:
                current_blocks = prev_blocks
                current_cov = prev_cov

    if best_blocks is not None:
        best_blocks, _ = remove_redundant(best_blocks, get_coverage(best_blocks))
    
    result = [list(all_blocks[b]) for b in best_blocks]
    return result
