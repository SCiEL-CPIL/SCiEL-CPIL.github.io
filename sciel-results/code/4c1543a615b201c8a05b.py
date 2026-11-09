"""Covering design C(v,k,t) — Optimized Greedy Set Cover with Local Search and Randomized Restarts."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 110.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    # Convert tsub_blocks to numpy arrays for faster vectorized updates
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    def run_greedy(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Initial gains: number of t-subsets covered by each block
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            if randomize:
                max_gain = np.max(block_gain)
                if max_gain <= 0:
                    break
                # Find indices with max gain
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                max_gain = np.max(block_gain)
                if max_gain <= 0:
                    break
                best_block_idx = np.argmax(block_gain)
            
            selected_blocks.append(all_blocks[best_block_idx])
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Vectorized update of gains for blocks covering this t-subset
                    block_gain[tsub_blocks_np[si]] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        if not blocks:
            return blocks
        
        # Precompute subs for each block in the solution
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs.append(subs)
            
        n = len(blocks)
        cov = np.zeros(N_t, dtype=np.int32)
        for subs in block_subs:
            for si in subs:
                cov[si] += 1
                
        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_subs[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_subs[i]:
                        cov[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    def local_search(blocks, max_iter=1000):
        """Try to improve by removing redundant ones."""
        if time.time() > DEADLINE:
            return blocks
        
        n = len(blocks)
        
        # Compute coverage
        cov = np.zeros(N_t, dtype=np.int32)
        block_subs_list = []
        for b in blocks:
            subs = [t_idx[s] for s in itertools.combinations(b, t)]
            block_subs_list.append(subs)
            for si in subs:
                cov[si] += 1
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < DEADLINE:
            improved = False
            iteration += 1
            
            # Try removing each block
            for i in range(n):
                if time.time() > DEADLINE:
                    return blocks
                subs = block_subs_list[i]
                can_remove = all(cov[si] > 1 for si in subs)
                if can_remove:
                    for si in subs:
                        cov[si] -= 1
                    blocks = blocks[:i] + blocks[i+1:]
                    block_subs_list = block_subs_list[:i] + block_subs_list[i+1:]
                    n = len(blocks)
                    improved = True
                    break
            
        return blocks

    # First run: deterministic greedy
    sol = run_greedy(randomize=False)
    sol = try_reduce(sol)
    sol = local_search(sol, max_iter=500)
    best_sol = sol
    best_len = len(sol)
    
    # Use remaining time for randomized restarts
    restart_count = 0
    while time.time() < DEADLINE - 5.0:
        seed = random.randint(0, 2**31)
        cand = run_greedy(randomize=True, seed=seed)
        cand = try_reduce(cand)
        cand = local_search(cand, max_iter=200)
        if len(cand) < best_len:
            best_sol = cand
            best_len = len(cand)
        restart_count += 1
    
    return best_sol
