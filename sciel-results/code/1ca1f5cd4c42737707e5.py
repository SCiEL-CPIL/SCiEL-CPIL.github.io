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
    
    # Precompute weights for all 2^N words using numpy for speed
    all_words = np.arange(max_val, dtype=np.int32)
    weights = np.zeros(max_val, dtype=np.int32)
    for bit in range(N):
        mask = 1 << bit
        weights[all_words & mask] += 1
    
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
        """Expand a base code using multi-restart greedy nonlinear addition."""
        base_set = set(base_code)
        base_arr = np.array(base_code, dtype=np.int32)
        
        # Use numpy to efficiently find candidates with min distance >= MIN_D to all base words
        # For each candidate x, compute min weight of x^c for all c in base_code
        # We do this in batches to manage memory
        candidates = []
        min_distances = []
        
        batch_size = 1024
        for start in range(0, max_val, batch_size):
            if time.time() > deadline - 5:
                break
            end = min(start + batch_size, max_val)
            batch = np.arange(start, end, dtype=np.int32)
            
            # Skip words already in base code
            mask = np.ones(len(batch), dtype=bool)
            for b in base_code:
                mask[batch == b] = False
            if not np.any(mask):
                continue
            batch = batch[mask]
            
            if len(batch) == 0:
                continue
            
            # Compute min distance to base code for each word in batch
            # For efficiency, compute in sub-batches against base code
            min_d_batch = np.full(len(batch), max_val, dtype=np.int32)
            
            # Process base code in chunks
            base_chunk_size = 64
            for b_start in range(0, len(base_code), base_chunk_size):
                if time.time() > deadline - 5:
                    break
                b_end = min(b_start + base_chunk_size, len(base_code))
                base_chunk = base_arr[b_start:b_end]
                
                # Compute XOR distances: batch[:, None] ^ base_chunk[None, :]
                # This creates a (len(batch), len(base_chunk)) array
                xor_dists = batch[:, None] ^ base_chunk[None, :]
                # Compute weights
                w = np.zeros_like(xor_dists, dtype=np.int32)
                for bit in range(N):
                    m = 1 << bit
                    w[xor_dists & m] += 1
                
                # Min over base_chunk dimension
                batch_min = np.min(w, axis=1)
                min_d_batch = np.minimum(min_d_batch, batch_min)
            
            # Filter candidates with min distance >= MIN_D
            valid_mask = min_d_batch >= MIN_D
            if np.any(valid_mask):
                candidates.append(batch[valid_mask])
                min_distances.append(min_d_batch[valid_mask])
        
        if not candidates:
            return base_code[:]
        
        all_candidates = np.concatenate(candidates) if len(candidates) > 1 else candidates[0]
        all_min_dists = np.concatenate(min_distances) if len(min_distances) > 1 else min_distances[0]
        
        if len(all_candidates) == 0:
            return base_code[:]
        
        # Sort by min distance descending (prefer candidates that are far from base code)
        sort_idx = np.argsort(-all_min_dists)
        all_candidates = all_candidates[sort_idx]
        all_min_dists = all_min_dists[sort_idx]
        
        best_result = base_code[:]
        best_size = len(base_code)
        
        # Multi-restart greedy - use fewer restarts but more thorough
        num_restarts = 10
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Shuffle candidates for diversity
            shuffled_idx = np.random.permutation(len(all_candidates))
            shuffled_candidates = all_candidates[shuffled_idx]
            
            added_in_phase2 = []
            added_set = set()
            
            for x in shuffled_candidates:
                if time.time() > deadline - 5:
                    break
                
                x_int = int(x)
                if x_int in added_set:
                    continue
                
                # Check against added words only (base code already filtered)
                valid = True
                for c in added_in_phase2:
                    if weights[x_int ^ c] < MIN_D:
                        valid = False
                        break
                
                if valid:
                    added_in_phase2.append(x_int)
                    added_set.add(x_int)
            
            current_size = len(base_code) + len(added_in_phase2)
            if current_size > best_size:
                best_size = current_size
                best_result = base_code + added_in_phase2
        
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
