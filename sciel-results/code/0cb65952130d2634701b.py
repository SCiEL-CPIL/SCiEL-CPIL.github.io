import time
import sys
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights for all 2^N words
    weights = np.array([0] * max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
    
    def build_linear_code(seed_order=None):
        """Build a linear code using greedy basis selection."""
        basis = []
        current_code = [0]
        
        if seed_order is None:
            candidates = list(range(1, max_val))
        else:
            candidates = seed_order
        
        for g in candidates:
            if time.time() > deadline - 20:
                break
            if g == 0:
                continue
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                old_code = current_code
                current_code = []
                for c in old_code:
                    current_code.append(c)
                    current_code.append(c ^ g)
                if len(basis) >= 8:
                    break
        
        return current_code if len(current_code) >= 2 else None
    
    def nonlinear_expand(base_code, time_budget):
        """Expand a base code using batched vectorized greedy nonlinear addition."""
        base_set = set(base_code)
        base_arr = np.array(base_code, dtype=np.int32)
        
        # Compute min distance from each candidate to base code (vectorized)
        chunk_size = 8192
        candidates_by_score = defaultdict(list)
        
        for start in range(0, max_val, chunk_size):
            if time.time() > deadline - 15:
                break
            end = min(start + chunk_size, max_val)
            chunk = np.arange(start, end, dtype=np.int32)
            
            # Filter out base words
            mask = np.array([x not in base_set for x in chunk], dtype=bool)
            chunk = chunk[mask]
            
            if len(chunk) == 0:
                continue
            
            # Compute distances to all base words (vectorized)
            min_dists = np.full(len(chunk), max_val, dtype=np.int32)
            
            base_chunk_size = 256
            for bs in range(0, len(base_arr), base_chunk_size):
                be = min(bs + base_chunk_size, len(base_arr))
                base_sub = base_arr[bs:be]
                xor_matrix = chunk[:, None] ^ base_sub[None, :]
                dist_matrix = weights[xor_matrix]
                chunk_min = dist_matrix.min(axis=1)
                min_dists = np.minimum(min_dists, chunk_min)
            
            # Filter valid candidates (min_dist >= MIN_D)
            valid_mask = min_dists >= MIN_D
            valid_chunk = chunk[valid_mask]
            valid_scores = min_dists[valid_mask]
            
            for x, s in zip(valid_chunk, valid_scores):
                candidates_by_score[int(s)].append(int(x))
        
        if not candidates_by_score:
            return base_code[:]
        
        sorted_scores = sorted(candidates_by_score.keys(), reverse=True)
        
        best_result = base_code[:]
        best_size = len(base_code)
        
        # Multi-restart greedy with batched vectorized checks
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            existing_list = list(base_code)
            existing_set = set(base_code)
            existing_arr = np.array(base_code, dtype=np.int32)
            max_capacity = max_val  # pre-allocated max size
            existing_arr_full = np.zeros(max_capacity, dtype=np.int32)
            existing_arr_full[:len(base_code)] = base_arr
            cur_len = len(base_code)
            
            for s in sorted_scores:
                if time.time() > deadline - 5:
                    break
                group = candidates_by_score[s][:]
                random.shuffle(group)
                
                # Batch process: check all candidates in group against existing code
                if len(group) == 0:
                    continue
                
                group_arr = np.array(group, dtype=np.int32)
                
                # Compute min distance from each candidate to all existing codewords
                # Process in sub-batches to manage memory
                batch_size = 4096
                valid_indices = []
                
                for gb in range(0, len(group_arr), batch_size):
                    if time.time() > deadline - 5:
                        break
                    ge = min(gb + batch_size, len(group_arr))
                    sub_group = group_arr[gb:ge]
                    
                    # Compute distances: sub_group (batch) x existing (cur_len)
                    # Process existing in sub-chunks
                    min_dists_batch = np.full(len(sub_group), max_val, dtype=np.int32)
                    existing_chunk_size = 512
                    for ec in range(0, cur_len, existing_chunk_size):
                        ee = min(ec + existing_chunk_size, cur_len)
                        existing_sub = existing_arr_full[ec:ee]
                        xor_mat = sub_group[:, None] ^ existing_sub[None, :]
                        dist_mat = weights[xor_mat]
                        batch_min = dist_mat.min(axis=1)
                        min_dists_batch = np.minimum(min_dists_batch, batch_min)
                    
                    valid_in_batch = np.where(min_dists_batch >= MIN_D)[0]
                    for idx in valid_in_batch:
                        valid_indices.append(gb + idx)
                
                # Add all valid candidates
                for idx in valid_indices:
                    x = int(group_arr[idx])
                    if x in existing_set:
                        continue
                    existing_list.append(x)
                    existing_set.add(x)
                    existing_arr_full[cur_len] = x
                    cur_len += 1
            
            current_size = cur_len
            if current_size > best_size:
                best_size = current_size
                best_result = existing_list[:]
        
        return best_result
    
    # Phase 1: Try multiple linear code constructions
    base_codes = []
    
    # First, deterministic scan
    base_code = build_linear_code()
    if base_code and len(base_code) >= 256:
        base_codes.append(base_code)
    
    # Try a few random orderings if time permits
    if time.time() < deadline - 30:
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            random_order = list(range(1, max_val))
            random.shuffle(random_order)
            bc = build_linear_code(random_order)
            if bc and len(bc) >= 256:
                base_codes.append(bc)
    
    if not base_codes:
        # Fallback: return what we have
        return [0]
    
    # Phase 2: Nonlinear expansion for each base code
    best_overall = []
    best_overall_size = 0
    
    for i, bc in enumerate(base_codes):
        if time.time() > deadline - 10:
            break
        # Allocate remaining time
        remaining = deadline - 8 - time.time()
        if remaining < 2:
            break
        result = nonlinear_expand(bc, remaining)
        if len(result) > best_overall_size:
            best_overall_size = len(result)
            best_overall = result
    
    if not best_overall:
        best_overall = base_codes[0]
    
    return best_overall
