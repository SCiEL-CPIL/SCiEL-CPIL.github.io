import numpy as np
import itertools
import time
import random
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    # But keep it general enough
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for radius 1 (including self)
    neighbors = []
    neighbor_sets = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))
    
    # Convert to numpy arrays for fast operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Phase 1: Greedy Set Cover with randomized tie-breaking
    def greedy_cover(random_seed=0):
        rng = random.Random(random_seed)
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            
            # Compute gains: for each candidate, how many uncovered points it covers
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            
            for idx in uncovered_indices:
                gains[neighbors[idx]] += 1
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Randomized selection among top candidates
            top_candidates = np.where(gains == max_gain)[0]
            best_idx = top_candidates[rng.randint(0, len(top_candidates)-1)]
            
            code.append(int(best_idx))
            code_set.add(int(best_idx))
            uncovered[neighbors[best_idx]] = False
        
        return code, code_set
    
    # Phase 2: Local search with multiple strategies
    def local_search(code, code_set, time_budget=60):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        
        best_code = code[:]
        best_code_set = code_set.copy()
        best_size = len(code)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # --- Deletion Pass ---
            to_remove = []
            for c in code:
                if np.min(counts[neighbors[c]]) > 1:
                    to_remove.append(c)
                
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
            
            if len(code) < best_size:
                best_code = code[:]
                best_code_set = code_set.copy()
                best_size = len(code)
            
            # --- Swap Pass: Try to replace one codeword with another ---
            random.shuffle(code)
            
            for c in code:
                if time.time() > deadline - 5:
                    break
                    
                # Get critical points: points covered only by c
                crit_points = [i for i in neighbors[c] if counts[i] == 1]
                
                if not crit_points:
                    continue
                
                # Try to find a replacement that covers all critical points
                found_swap = False
                random.shuffle(crit_points)
                
                # Use the first critical point to generate candidates
                p = crit_points[0]
                cands = [x for x in neighbors[p] if x not in code_set]
                
                best_w = -1
                best_score = -1
                
                for w in cands:
                    # Check if w covers all critical points
                    valid = all(k in neighbor_sets[w] for k in crit_points)
                    if not valid:
                        continue
                    
                    # Score: maximize coverage of points with count==2
                    score = sum(1 for i in neighbors[w] if counts[i] == 2)
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    
                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors[best_w]] += 1
                    
                    improved = True
                    found_swap = True
                    break
            
            # --- 2-Swap Pass: Try to replace two codewords with one or two better ones ---
            if not improved and len(code) > 2:
                # Sample a few pairs
                samples = min(50, len(code))
                for _ in range(samples):
                    if time.time() > deadline - 5:
                        break
                    
                    i, j = random.sample(range(len(code)), 2)
                    c1, c2 = code[i], code[j]
                    
                    # Points that would become uncovered if we remove both
                    temp_counts = counts.copy()
                    temp_counts[neighbors[c1]] -= 1
                    temp_counts[neighbors[c2]] -= 1
                    
                    uncovered_after = [k for k in range(N) if temp_counts[k] == 0]
                    
                    if not uncovered_after:
                        # Both are redundant
                        code.remove(c1)
                        code.remove(c2)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        counts[neighbors[c1]] -= 1
                        counts[neighbors[c2]] -= 1
                        improved = True
                        break
                    
                    # Try to find one or two replacements
                    # First try one replacement
                    first_point = uncovered_after[0]
                    cands = [x for x in neighbors[first_point] if x not in code_set]
                    
                    best_single = -1
                    best_single_score = -1
                    
                    for w in cands:
                        if all(k in neighbor_sets[w] for k in uncovered_after):
                            score = sum(1 for i in neighbors[w] if counts[i] >= 2)
                            if score > best_single_score:
                                best_single_score = score
                                best_single = w
                    
                    if best_single != -1:
                        code.remove(c1)
                        code.remove(c2)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        counts[neighbors[c1]] -= 1
                        counts[neighbors[c2]] -= 1
                        
                        code.append(best_single)
                        code_set.add(best_single)
                        counts[neighbors[best_single]] += 1
                        
                        improved = True
                        break
        
        # Final cleanup
        counts = np.zeros(N, dtype=np.int32)
        for idx in best_code:
            counts[neighbors[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for c in list(best_code):
                if np.min(counts[neighbors[c]]) > 1:
                    best_code.remove(c)
                    best_code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    changed = True
        
        return best_code
    
    # Phase 3: Multi-start with different random seeds
    best_code = None
    best_size = float('inf')
    
    num_starts = 5
    for seed in range(num_starts):
        if time.time() > deadline - 40:
            break
        
        code, code_set = greedy_cover(random_seed=seed)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
        
        # Local search
        refined = local_search(code, code_set)
        
        if len(refined) < best_size:
            best_code = refined
            best_size = len(refined)
    
    # Final phase: More aggressive local search on the best solution
    if best_code is not None and time.time() < deadline - 10:
        best_code_set = set(best_code)
        final_code = local_search(best_code, best_code_set)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)
    
    if best_code is None:
        # Fallback: return empty or a trivial solution
        best_code = [0]
    
    return [list(words[i]) for i in best_code]
