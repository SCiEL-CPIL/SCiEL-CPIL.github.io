import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # safety margin before 150s kill
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Precompute for each 7-subset: which t-subsets it covers
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # nCr helper
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
    
    t_count = nCr(k, t)  # Number of t-subsets per k-block
    
    # Build mapping: for each t-subset, which k-subsets contain it
    t_to_blocks = [set() for _ in range(n_ts)]
    block_t_subsets = []
    
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        sub_indices = [idx_map[sub] for sub in subs]
        block_t_subsets.append(sub_indices)
        for si in sub_indices:
            t_to_blocks[si].add(i)
    
    # Convert to numpy arrays for speed
    block_t_subsets_arr = np.array(block_t_subsets, dtype=np.int32)
    
    # Build block lookup
    block_to_idx = {block: i for i, block in enumerate(all_ks)}
    
    # === PBD-inspired pool generation ===
    def generate_pbd_pool(seed):
        """Generate a structured block pool with balanced pair coverage."""
        rng = random.Random(seed)
        
        pair_count = np.zeros((v, v), dtype=np.int32)
        blocks = []
        n_blocks_target = 120
        
        # Generate random blocks, biasing toward balanced pair coverage
        attempts = 0
        while len(blocks) < n_blocks_target and attempts < n_blocks_target * 5:
            if time.time() - start_time > time_limit - 10:
                break
            attempts += 1
            block = tuple(sorted(rng.sample(range(v), k)))
            
            # Score: prefer blocks that cover under-represented pairs
            score = 0
            for i in range(k):
                for j in range(i+1, k):
                    score += pair_count[block[i]][block[j]]
            
            # Accept with probability based on score (lower score = better)
            # Use a simple threshold
            if len(blocks) < n_blocks_target // 2 or score < 50:
                if block not in set(tuple(b) for b in blocks):
                    blocks.append(block)
                    for i in range(k):
                        for j in range(i+1, k):
                            pair_count[block[i]][block[j]] += 1
        
        return blocks
    
    # === Greedy set cover ===
    def greedy_set_cover(seed, max_blocks=None):
        """Standard greedy set cover."""
        rng = random.Random(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        if max_blocks is None:
            max_blocks = 200
        
        while remaining > 0 and len(blocks_indices) < max_blocks:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute gain for each candidate block
            covered_in_blocks = covered[block_t_subsets_arr]
            gains = t_count - covered_in_blocks.sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 50:
                sample_size = min(50, len(max_gain_blocks))
                candidates = max_gain_blocks[rng.sample(range(len(max_gain_blocks)), sample_size)]
            else:
                candidates = max_gain_blocks
            
            # Pick the one that covers the most "rare" subsets
            best_candidate = candidates[0]
            best_rarity_score = -1
            
            for cand in candidates:
                subs = block_t_subsets_arr[cand]
                newly = ~covered[subs]
                if np.any(newly):
                    # Rarity: prefer subsets that appear in fewer k-subsets
                    rarity = np.sum(1.0 / (1.0 + np.array([len(t_to_blocks[s]) for s in subs[newly]])))
                else:
                    rarity = 0
                if rarity > best_rarity_score:
                    best_rarity_score = rarity
                    best_candidate = cand
            
            blocks_indices.append(best_candidate)
            subs = block_t_subsets_arr[best_candidate]
            newly = ~covered[subs]
            remaining -= np.sum(newly)
            covered[subs] = True
        
        return blocks_indices
    
    # === Greedy deletion ===
    def greedy_delete(pool_indices):
        """Remove redundant blocks from the pool."""
        if not pool_indices:
            return pool_indices
        
        cov_count = np.zeros(n_ts, dtype=np.int32)
        for bi in pool_indices:
            cov_count[block_t_subsets_arr[bi]] += 1
        
        # Verify full coverage
        if np.any(cov_count == 0):
            return None  # Invalid pool
        
        remaining = list(pool_indices)
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            new_remaining = []
            for bi in remaining:
                subs = block_t_subsets_arr[bi]
                if np.all(cov_count[subs] > 1):
                    cov_count[subs] -= 1
                    improved = True
                else:
                    new_remaining.append(bi)
            remaining = new_remaining
        
        return remaining
    
    # === Main search ===
    best_solution = None
    best_score = float('inf')
    
    # Phase 1: PBD-inspired pools
    for seed in range(15):
        if time.time() - start_time > time_limit - 10:
            break
        
        pool = generate_pbd_pool(seed)
        if not pool:
            continue
        
        pool_indices = [block_to_idx[b] for b in pool]
        
        # Check coverage and add missing
        cov_count = np.zeros(n_ts, dtype=np.int32)
        for bi in pool_indices:
            cov_count[block_t_subsets_arr[bi]] += 1
        
        if np.any(cov_count == 0):
            missing = np.where(cov_count == 0)[0]
            extra = []
            for mi in missing:
                if time.time() - start_time > time_limit - 5:
                    break
                candidates = list(t_to_blocks[mi])
                if not candidates:
                    continue
                best_cand = None
                best_gain = -1
                for cand in candidates:
                    subs = block_t_subsets_arr[cand]
                    gain = np.sum(cov_count[subs] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                if best_cand is not None and best_cand not in pool_indices and best_cand not in extra:
                    extra.append(best_cand)
                    cov_count[block_t_subsets_arr[best_cand]] += 1
            
            pool_indices.extend(extra)
        
        # Delete redundant blocks
        result = greedy_delete(pool_indices)
        if result is not None:
            score = len(result)
            if score < best_score:
                best_score = score
                best_solution = result
    
    # Phase 2: Greedy set cover with deletion
    for seed in range(10):
        if time.time() - start_time > time_limit - 10:
            break
        
        blocks = greedy_set_cover(seed)
        if not blocks:
            continue
        
        result = greedy_delete(blocks)
        if result is not None:
            score = len(result)
            if score < best_score:
                best_score = score
                best_solution = result
    
    # Phase 3: Additional deletion passes on best solution
    if best_solution is not None and time.time() - start_time < time_limit - 5:
        current = list(best_solution)
        for _ in range(3):
            if time.time() - start_time > time_limit - 3:
                break
            cov_count = np.zeros(n_ts, dtype=np.int32)
            for bi in current:
                cov_count[block_t_subsets_arr[bi]] += 1
            
            improved = True
            while improved and time.time() - start_time < time_limit - 3:
                improved = False
                new_current = []
                for bi in current:
                    subs = block_t_subsets_arr[bi]
                    if np.all(cov_count[subs] > 1):
                        cov_count[subs] -= 1
                        improved = True
                    else:
                        new_current.append(bi)
                current = new_current
        
        if len(current) < best_score:
            best_score = len(current)
            best_solution = current
    
    if best_solution is None:
        blocks = greedy_set_cover(0)
        if blocks:
            best_solution = blocks
        else:
            return []
    
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    # Final verification and fix
    final_cov = np.zeros(n_ts, dtype=np.int32)
    for i in best_solution:
        final_cov[block_t_subsets_arr[i]] += 1
    if np.any(final_cov == 0):
        missing = np.where(final_cov == 0)[0]
        for mi in missing:
            if time.time() - start_time > time_limit - 1:
                break
            for cand in t_to_blocks[mi]:
                if cand not in best_solution:
                    best_solution.append(cand)
                    final_cov[block_t_subsets_arr[cand]] += 1
                    break
        final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
