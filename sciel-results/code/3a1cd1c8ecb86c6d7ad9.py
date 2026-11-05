import numpy as np
from scipy.sparse import csr_matrix
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    # ---------- Precomputation ----------
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # block -> list of t-subset indices it covers
    block_to_subsets = [[] for _ in range(num_blocks)]
    # subset -> list of block indices covering it
    subset_to_blocks = [[] for _ in range(num_t_subs)]
    
    rows = []
    cols = []
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            j = t_sub_index[ts]
            block_to_subsets[i].append(j)
            subset_to_blocks[j].append(i)
            rows.append(i)
            cols.append(j)
            
    data = np.ones(len(rows), dtype=np.int32)
    M = csr_matrix((data, (rows, cols)), shape=(num_blocks, num_t_subs))

    # Weights for subsets: 1 / frequency
    # Frequency of a subset is the number of blocks containing it
    # For C(v,k,t), freq is C(v-t, k-t)
    # Let's compute it from M to be safe, or just use the formula.
    # Formula: C(19-5, 8-5) = C(14, 3) = 364.
    # So all subsets have the same frequency?
    # Yes, in a symmetric design space, every t-subset is in C(v-t, k-t) blocks.
    # So weights are uniform. The standard greedy is already optimal in terms of "rarity".
    # So we stick to counting uncovered.
    
    def run_greedy_vectorized(seed, initial_uncovered=None):
        rng = np.random.default_rng(seed)
        if initial_uncovered is None:
            uncovered = np.ones(num_t_subs, dtype=np.int8)
        else:
            uncovered = initial_uncovered.copy()
            
        if not np.any(uncovered):
            return []
            
        selected = []
        
        # gains[b] = number of uncovered subsets in block b
        # gains = M_T.dot(uncovered)
        
        while True:
            if time.time() > deadline - 10.0:
                break
            
            # Calculate gains
            # M is sparse (num_blocks x num_t_subs)
            # uncovered is dense (num_t_subs,)
            # Result is dense (num_blocks,)
            gains = M.dot(uncovered).astype(np.int32)
            
            if gains.max() == 0:
                break
                
            # Add noise
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best = int(np.argmax(scores))
            
            if gains[best] == 0:
                break
                
            selected.append(best)
            
            # Update uncovered
            # Set uncovered[s] = 0 for all s in block_to_subsets[best]
            for s in block_to_subsets[best]:
                uncovered[s] = 0
                
        return selected

    def reduce_blocks(blocks, seed):
        rng = random.Random(seed)
        cur = list(blocks)
        if not cur:
            return cur
            
        cover = np.zeros(num_t_subs, dtype=np.int32)
        for b in cur:
            for s in block_to_subsets[b]:
                cover[s] += 1
                
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            # Find redundant blocks
            # A block is redundant if all its subsets have cover[s] > 1
            redundant = []
            for i, b in enumerate(cur):
                is_red = True
                for s in block_to_subsets[b]:
                    if cover[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant.append(i)
                    
            if not redundant:
                break
                
            # Remove one randomly
            pos = redundant[rng.randint(0, len(redundant) - 1)]
            b = cur[pos]
            for s in block_to_subsets[b]:
                cover[s] -= 1
            cur.pop(pos)
            improved = True
            
        return cur

    best = None
    best_len = float('inf')
    trial = 0
    
    while time.time() < deadline - 15.0:
        trial += 1
        
        # Strategy 1: Full Greedy + Reduce
        g = run_greedy_vectorized(trial)
        r = reduce_blocks(g, trial)
        if len(r) < best_len:
            best_len = len(r)
            best = r
            
        # Strategy 2: Perturb + Repair + Reduce
        if best is not None and len(best) > 10 and time.time() < deadline - 20.0:
            rng2 = random.Random(trial * 7 + 1)
            perturbed = list(best)
            # Remove a random number of blocks
            num_remove = rng2.randint(1, min(5, len(perturbed) - 1))
            rem_idx = rng2.sample(range(len(perturbed)), num_remove)
            rem_set = set(perturbed[i] for i in rem_idx)
            
            # Compute uncovered from perturbed minus removed
            cover = np.zeros(num_t_subs, dtype=np.int32)
            for b in perturbed:
                if b in rem_set:
                    continue
                for s in block_to_subsets[b]:
                    cover[s] += 1
                    
            unc_mask = (cover == 0).astype(np.int8)
            
            # Repair
            rep = run_greedy_vectorized(trial * 13 + 5, initial_uncovered=unc_mask)
            
            new_set = [b for b in perturbed if b not in rem_set]
            new_set.extend(rep)
            # De-duplicate
            new_set = list(dict.fromkeys(new_set))
            
            r2 = reduce_blocks(new_set, trial * 17 + 3)
            if len(r2) < best_len:
                best_len = len(r2)
                best = r2
                
    if best is None:
        best = reduce_blocks(run_greedy_vectorized(0), 0)
        
    return [list(all_blocks[i]) for i in best]
