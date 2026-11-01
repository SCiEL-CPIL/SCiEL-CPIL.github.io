import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Helper for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # If parameters are not the specific one, use a simple fallback
    if v != 19 or k != 8 or t != 5:
        # Fallback: Standard Greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            best_block = None
            best_gain = -1
            remaining = [x for x in range(v) if x not in target]
            need = k - t
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    # Specific optimization for v=19, k=8, t=5
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage structures
    # block_to_ts[j]: list of t-subset indices covered by block j
    block_to_ts = []
    # ts_to_blocks[i]: list of block indices covering t-subset i
    ts_to_blocks = [[] for _ in range(n_t)]
    
    for j, block in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(covered_indices)
        
    # Convert to numpy arrays for speed
    block_ts_arrays = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_blocks_arrays = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    def run_greedy(seed):
        random.seed(seed)
        # gain[j] = number of currently uncovered t-subsets covered by block j
        gain = np.full(n_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(n_t, True, dtype=bool)
        is_selected = np.full(n_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Randomly pick one to diversify
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1 # Mark as selected
            
            # Identify newly covered t-subsets
            newly_covered_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    newly_covered_ts.append(ts_idx)
            
            if not newly_covered_ts:
                continue
                
            # Vectorized gain update
            # We need to decrement gain for all blocks that contain any of the newly_covered_ts
            # and are not selected.
            # Collect all block indices that contain any of these t-subsets
            # Using a list to accumulate, then bincount is efficient
            blocks_to_decrement = []
            for ts_idx in newly_covered_ts:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])
                
            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=n_blocks)
                gain -= counts
                # Ensure selected blocks stay low
                gain[is_selected] = -1000

        return sol

    def optimize_redundancy(blocks):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
            
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(t_subset_to_idx[ts])
                block_ts_indices.append(ts_list)
                
        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 2.0:
                    return blocks
                    
                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break
                    
        return blocks

    def local_search_swap(blocks, max_iterations=5000):
        """
        Local search: try to replace a block with another to reduce total size? 
        Actually, simple removal is already done. 
        A more effective move is 'swap': remove block A, add block B if it covers 
        more uncovered subsets than A did, or simply if it allows removing another block later.
        
        Given the time, a simpler and effective heuristic is:
        1. Try removing any block. If valid, keep it (redundancy).
        2. Try swapping a block with a random other block. If the new block covers 
           at least as many subsets as the old one (considering the rest of the solution), 
           and ideally allows for further reduction, accept.
           
        However, the most powerful simple improvement for covering designs is often 
        just finding a better greedy start or using a more sophisticated local search 
        like '2-opt' or '3-opt' which is expensive.
        
        Let's stick to a robust redundancy removal + random restarts. 
        The vectorized greedy is the main driver here.
        """
        # For now, this function is a placeholder for future improvements, 
        # but redundancy removal is the key low-hanging fruit.
        return optimize_redundancy(blocks)

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy restarts
    # The vectorized greedy is fast, so we can afford many restarts.
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        sol = run_greedy(i)
        if sol is None:
            continue
            
        # Post-process: remove redundant blocks
        sol = optimize_redundancy(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = [list(b) for b in sol]
            
        # If we hit the theoretical lower bound, we can stop
        # Lower bound is ceil(n_t / C(k,t)) = ceil(11628 / 56) = 208
        if best_len <= 208:
            break

    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = run_greedy(0)
        if best_blocks is None:
            return []
        best_blocks = optimize_redundancy(best_blocks)

    return best_blocks
