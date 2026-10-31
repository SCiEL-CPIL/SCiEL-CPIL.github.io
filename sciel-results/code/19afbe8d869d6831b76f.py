import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    # Fallback for non-standard parameters
    if v != 19 or k != 7 or t != 4:
        return _greedy_generic(v, k, t)

    # --- Precomputation ---
    # 1. Map 4-subsets to IDs
    map4 = {}
    id_counter = 0
    for comb in itertools.combinations(range(v), t):
        map4[comb] = id_counter
        id_counter += 1
    n_4s = id_counter  # 3876

    # 2. Generate candidate blocks using cyclic orbit structure
    # Z19 is prime, so all orbits of 7-subsets have size 19
    # Each orbit is determined by its "canonical" representative (min element = 0)
    
    random.seed(42)
    
    # Generate diverse base blocks (canonical forms with 0 as smallest element)
    all_base_blocks = set()
    
    # Structured bases with 0 included
    for _ in range(200):
        rest = random.sample(range(1, v), k - 1)
        base = tuple(sorted([0] + rest))
        all_base_blocks.add(base)
    
    # Also add some without 0, then normalize
    for _ in range(300):
        base = tuple(sorted(random.sample(range(v), k)))
        # Normalize: subtract min mod v, then sort
        min_val = base[0]
        normalized = tuple(sorted((b - min_val) % v for b in base))
        all_base_blocks.add(normalized)
    
    # Add structured patterns
    for start in range(v):
        base = tuple(sorted((start + i * 3) % v for i in range(k)))
        min_val = base[0]
        normalized = tuple(sorted((b - min_val) % v for b in base))
        all_base_blocks.add(normalized)
        base = tuple(sorted((start + i * 5) % v for i in range(k)))
        min_val = base[0]
        normalized = tuple(sorted((b - min_val) % v for b in base))
        all_base_blocks.add(normalized)
    
    # Generate all cyclic shifts of all base blocks
    candidate_blocks = set()
    for base in all_base_blocks:
        for shift in range(v):
            shifted = tuple(sorted((b + shift) % v for b in base))
            candidate_blocks.add(shifted)
    
    # Add purely random blocks for diversity
    for _ in range(500):
        block = tuple(sorted(random.sample(range(v), k)))
        candidate_blocks.add(block)
    
    candidate_blocks = sorted(candidate_blocks)
    n_candidates = len(candidate_blocks)
    
    # 3. For each candidate block, get list of 4-subset IDs it covers
    block_4s = []
    for block in candidate_blocks:
        ids = []
        for sub in itertools.combinations(block, t):
            ids.append(map4[sub])
        block_4s.append(np.array(ids, dtype=np.int32))
    
    block_4s_array = np.array(block_4s, dtype=np.int32)
    
    # 4. For each 4-subset, get list of candidate block indices covering it
    sub_blocks = [[] for _ in range(n_4s)]
    for b_idx, ids in enumerate(block_4s):
        for sid in ids:
            sub_blocks[sid].append(b_idx)
    
    # Build a set for fast membership testing
    candidate_set = set(range(n_candidates))
    
    # --- Multi-restart local search ---
    best_blocks = None
    best_len = float('inf')
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Different random seed per restart
        random.seed(42 + restart * 137)
        
        # Greedy initialization
        counts = np.zeros(n_4s, dtype=np.int32)
        current_blocks = []
        uncovered_mask = np.ones(n_4s, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() - start_time > time_limit - 15:
                break
            
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                break
            
            current_blocks.append(best_idx)
            counts[block_4s_array[best_idx]] += 1
            uncovered_mask = (counts == 0)
        
        # Local search
        no_improve_count = 0
        
        while time.time() - start_time < time_limit - 5:
            improved = False
            
            # 1. Remove redundant blocks
            if current_blocks:
                b_4s = block_4s_array[current_blocks]
                c_4s = counts[b_4s]
                red_scores = (c_4s > 1).sum(axis=1)
                
                max_red = np.max(red_scores)
                if max_red > 0:
                    candidates = np.where(red_scores == max_red)[0]
                    random.shuffle(candidates)
                    
                    for i in candidates:
                        b_idx = current_blocks[i]
                        counts[block_4s_array[b_idx]] -= 1
                        if np.any(counts == 0):
                            counts[block_4s_array[b_idx]] += 1
                        else:
                            current_blocks.remove(b_idx)
                            improved = True
                            break
            
            # 2. Try 1-swap: remove one, add one
            if not improved and current_blocks:
                for _ in range(50):
                    if time.time() - start_time > time_limit - 5:
                        break
                    
                    b_old = random.choice(current_blocks)
                    counts[block_4s_array[b_old]] -= 1
                    uncovered_now = (counts == 0)
                    
                    if np.any(uncovered_now):
                        unc_idx = np.where(uncovered_now)[0]
                        
                        # Find best replacement
                        candidates = set()
                        for uid in unc_idx:
                            candidates.update(sub_blocks[uid])
                        
                        if candidates:
                            best_candidate = -1
                            max_cov = -1
                            for cand in candidates:
                                if cand not in current_blocks:
                                    cov = np.sum(uncovered_now[block_4s_array[cand]])
                                    if cov > max_cov:
                                        max_cov = cov
                                        best_candidate = cand
                            
                            if best_candidate != -1:
                                current_blocks.remove(b_old)
                                current_blocks.append(best_candidate)
                                counts[block_4s_array[best_candidate]] += 1
                                
                                if not np.any(counts == 0):
                                    improved = True
                                else:
                                    counts[block_4s_array[best_candidate]] -= 1
                                    counts[block_4s_array[b_old]] += 1
                                    current_blocks.remove(best_candidate)
                                    current_blocks.append(b_old)
                    else:
                        current_blocks.remove(b_old)
                        improved = True
                        break
            
            # 3. Try 2-swap: remove two, add one
            if not improved and len(current_blocks) >= 2:
                for _ in range(30):
                    if time.time() - start_time > time_limit - 5:
                        break
                    
                    if len(current_blocks) < 2:
                        break
                    b_old1, b_old2 = random.sample(current_blocks, 2)
                    
                    counts[block_4s_array[b_old1]] -= 1
                    counts[block_4s_array[b_old2]] -= 1
                    uncovered_now = (counts == 0)
                    
                    if np.any(uncovered_now):
                        unc_idx = np.where(uncovered_now)[0]
                        
                        candidates = set()
                        for uid in unc_idx:
                            candidates.update(sub_blocks[uid])
                        
                        if candidates:
                            best_candidate = -1
                            max_cov = -1
                            for cand in candidates:
                                if cand not in current_blocks:
                                    cov = np.sum(uncovered_now[block_4s_array[cand]])
                                    if cov > max_cov:
                                        max_cov = cov
                                        best_candidate = cand
                            
                            if best_candidate != -1:
                                current_blocks.remove(b_old1)
                                current_blocks.remove(b_old2)
                                current_blocks.append(best_candidate)
                                counts[block_4s_array[best_candidate]] += 1
                                
                                if not np.any(counts == 0):
                                    improved = True
                                else:
                                    counts[block_4s_array[best_candidate]] -= 1
                                    counts[block_4s_array[b_old1]] += 1
                                    counts[block_4s_array[b_old2]] += 1
                                    current_blocks.remove(best_candidate)
                                    current_blocks.append(b_old1)
                                    current_blocks.append(b_old2)
                    else:
                        current_blocks.remove(b_old1)
                        current_blocks.remove(b_old2)
                        improved = True
                        break
            
            # Update best
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # If stuck, perturb
            if no_improve_count > 15 and current_blocks:
                n_remove = min(3, len(current_blocks))
                to_remove = random.sample(current_blocks, n_remove)
                for b in to_remove:
                    counts[block_4s_array[b]] -= 1
                    current_blocks.remove(b)
                
                # Greedy repair
                uncovered_mask = (counts == 0)
                while np.any(uncovered_mask):
                    if time.time() - start_time > time_limit - 5:
                        break
                    scores = uncovered_mask[block_4s_array].sum(axis=1)
                    best_idx = np.argmax(scores)
                    if scores[best_idx] == 0:
                        break
                    if best_idx not in current_blocks:
                        current_blocks.append(best_idx)
                        counts[block_4s_array[best_idx]] += 1
                        uncovered_mask = (counts == 0)
                    else:
                        break
                
                no_improve_count = 0
        
        # Ensure validity for this restart
        if best_blocks is None or len(current_blocks) < best_len:
            if np.all(counts >= 1):
                best_len = len(current_blocks)
                best_blocks = list(current_blocks)

    # Final validity check and repair
    if best_blocks is None:
        # Fallback: greedy from scratch
        counts = np.zeros(n_4s, dtype=np.int32)
        best_blocks = []
        uncovered_mask = np.ones(n_4s, dtype=bool)
        while np.any(uncovered_mask):
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            best_blocks.append(best_idx)
            counts[block_4s_array[best_idx]] += 1
            uncovered_mask = (counts == 0)
    else:
        # Verify and repair
        counts = np.zeros(n_4s, dtype=np.int32)
        for b_idx in best_blocks:
            counts[block_4s_array[b_idx]] += 1
        uncovered_mask = (counts == 0)
        while np.any(uncovered_mask):
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            if best_idx not in best_blocks:
                best_blocks.append(best_idx)
                counts[block_4s_array[best_idx]] += 1
                uncovered_mask = (counts == 0)
            else:
                break

    result = [list(candidate_blocks[i]) for i in best_blocks]
    return result

def _greedy_generic(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
