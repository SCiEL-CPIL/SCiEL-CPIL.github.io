"""Covering design C(v,k,t) — Greedy Set Cover with Local Search and Randomized Restarts."""
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
    
    # 3. Precompute block -> t-subsets mapping (as list of lists for now, convert to arrays if needed)
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
        selected_indices = []
        
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
            
            selected_indices.append(best_block_idx)
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Vectorized update of gains for blocks covering this t-subset
                    block_gain[tsub_blocks_np[si]] -= 1
                        
        return selected_indices

    def try_reduce(block_indices):
        if not block_indices:
            return block_indices
        
        # Use precomputed block_tsubs
        n = len(block_indices)
        cov = np.zeros(N_t, dtype=np.int32)
        # Initialize coverage counts
        for idx in block_indices:
            for si in block_tsubs[idx]:
                cov[si] += 1
                
        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                subs = block_tsubs[block_indices[i]]
                for si in subs:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in subs:
                        cov[si] -= 1
                        
        return [idx for i, idx in enumerate(block_indices) if not removed[i]]

    def local_search(block_indices, max_iter=1000):
        """Try to improve by removing redundant ones."""
        if time.time() > DEADLINE:
            return block_indices
        
        n = len(block_indices)
        if n == 0:
            return block_indices
        
        # Compute coverage
        cov = np.zeros(N_t, dtype=np.int32)
        for idx in block_indices:
            for si in block_tsubs[idx]:
                cov[si] += 1
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < DEADLINE:
            improved = False
            iteration += 1
            
            # Try removing each block
            for i in range(n):
                if time.time() > DEADLINE:
                    return block_indices
                subs = block_tsubs[block_indices[i]]
                can_remove = True
                for si in subs:
                    if cov[si] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for si in subs:
                        cov[si] -= 1
                    block_indices = block_indices[:i] + block_indices[i+1:]
                    n = len(block_indices)
                    improved = True
                    break
            
        return block_indices

    # First run: deterministic greedy
    sol_indices = run_greedy(randomize=False)
    sol_indices = try_reduce(sol_indices)
    sol_indices = local_search(sol_indices, max_iter=500)
    best_sol_indices = sol_indices
    best_len = len(best_sol_indices)
    
    # Use remaining time for randomized restarts
    restart_count = 0
    while time.time() < DEADLINE - 5.0:
        seed = random.randint(0, 2**31)
        cand_indices = run_greedy(randomize=True, seed=seed)
        cand_indices = try_reduce(cand_indices)
        cand_indices = local_search(cand_indices, max_iter=200)
        if len(cand_indices) < best_len:
            best_sol_indices = cand_indices
            best_len = len(best_sol_indices)
        restart_count += 1
    
    # Convert indices back to blocks
    return [all_blocks[idx] for idx in best_sol_indices]
