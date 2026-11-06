import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    best_set = []
    best_size = 0
    
    def fast_greedy(noise=0.0, time_limit=None, seed=None):
        """Vectorized greedy using NumPy for scoring."""
        if time_limit is None:
            time_limit = deadline
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        chosen = []
        counts = defaultdict(int)
        
        # Precompute candidate arrays
        all_candidates = np.arange(1, N + 1)
        
        while True:
            if time.time() > time_limit:
                break
            
            if not chosen:
                # First element: pick randomly or greedily
                x = int(rng.choice(all_candidates))
                chosen.append(x)
                counts[2*x] = 1
                continue
            
            # Vectorized scoring
            chosen_arr = np.array(chosen)
            n_chosen = len(chosen)
            
            # Get current counts as array for vectorized access
            # We need counts for all possible sums
            max_sum = 2 * N
            counts_arr = np.zeros(max_sum + 1, dtype=np.int32)
            for s, c in counts.items():
                if s <= max_sum:
                    counts_arr[s] = c
            
            # For each candidate x, compute tightness
            # tightness(x) = number of sums s = x+y (y in chosen) with counts[s] == 1
            #                 + 1 if counts[2x] == 1
            # Valid if counts[2x] < 2 and counts[x+y] < 2 for all y in chosen
            
            # Build a matrix: for each candidate x and each chosen y, sum = x+y
            # candidates not in chosen
            chosen_set = set(chosen)
            candidates = [x for x in all_candidates if x not in chosen_set]
            
            if not candidates:
                break
            
            candidates_arr = np.array(candidates)
            
            # Sums matrix: [n_cands, n_chosen]
            sums = candidates_arr[:, None] + chosen_arr[None, :]
            # 2x sums
            double_sums = 2 * candidates_arr
            
            # Get counts for these sums
            # Clip to valid range
            sums_clipped = np.clip(sums, 0, max_sum)
            double_clipped = np.clip(double_sums, 0, max_sum)
            
            counts_sums = counts_arr[sums_clipped]
            counts_double = counts_arr[double_clipped]
            
            # Validity: no sum has count >= 2
            valid_mask = (counts_sums < 2) & (counts_double < 2)
            # valid_mask is [n_cands, n_chosen+1] effectively, but we need all to be valid
            # Actually counts_sums is [n_cands, n_chosen], counts_double is [n_cands]
            # Valid if all counts_sums < 2 AND counts_double < 2
            valid = np.all(counts_sums < 2, axis=1) & (counts_double < 2)
            
            if not np.any(valid):
                break
            
            # Tightness: count of sums with count == 1
            tight_sums = np.sum(counts_sums == 1, axis=1)
            tight_double = (counts_double == 1).astype(np.int32)
            tightness = tight_sums + tight_double
            
            # Add noise
            if noise > 0:
                tightness = tightness + rng.random(len(candidates)) * noise
            
            # Among valid, pick minimum tightness
            valid_indices = np.where(valid)[0]
            valid_tightness = tightness[valid_indices]
            
            min_idx = valid_indices[np.argmin(valid_tightness)]
            x = int(candidates_arr[min_idx])
            
            # Add x
            chosen.append(x)
            counts[2*x] = counts.get(2*x, 0) + 1
            for y in chosen[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        
        return chosen
    
    def build_counts(chosen):
        """Build counts dictionary from a set."""
        counts = defaultdict(int)
        for x in chosen:
            counts[2*x] = counts.get(2*x, 0) + 1
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    def local_search(initial_set, time_limit):
        """Efficient local search with vectorized checks."""
        chosen = list(initial_set)
        counts = build_counts(chosen)
        
        chosen_set = set(chosen)
        candidates = [x for x in range(1, N+1) if x not in chosen_set]
        
        if not candidates:
            return chosen, counts
        
        max_sum = 2 * N
        
        while time.time() < time_limit:
            # Try to add
            if not candidates:
                break
            
            # Vectorized add check
            chosen_arr = np.array(chosen)
            candidates_arr = np.array(candidates)
            
            counts_arr = np.zeros(max_sum + 1, dtype=np.int32)
            for s, c in counts.items():
                if s <= max_sum:
                    counts_arr[s] = c
            
            sums = candidates_arr[:, None] + chosen_arr[None, :]
            double_sums = 2 * candidates_arr
            
            sums_clipped = np.clip(sums, 0, max_sum)
            double_clipped = np.clip(double_sums, 0, max_sum)
            
            counts_sums = counts_arr[sums_clipped]
            counts_double = counts_arr[double_clipped]
            
            valid = np.all(counts_sums < 2, axis=1) & (counts_double < 2)
            
            if np.any(valid):
                # Pick a valid one (prefer low tightness)
                tight_sums = np.sum(counts_sums == 1, axis=1)
                tight_double = (counts_double == 1).astype(np.int32)
                tightness = tight_sums + tight_double
                
                valid_indices = np.where(valid)[0]
                min_idx = valid_indices[np.argmin(tightness[valid_indices])]
                x = int(candidates_arr[min_idx])
                
                # Add x
                counts[2*x] = counts.get(2*x, 0) + 1
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
                chosen_set.add(x)
                candidates.remove(x)
                continue
            
            # Stuck: try swaps
            # Try a few random swaps
            swap_done = False
            for _ in range(10):
                if time.time() > time_limit:
                    break
                if not candidates or not chosen:
                    break
                
                x_to_add = random.choice(candidates)
                y_to_remove = random.choice(chosen)
                
                # Check if removing y and adding x is valid
                # Temporarily remove y
                old_counts = dict(counts)
                
                # Remove y's contributions
                counts[2*y_to_remove] -= 1
                if counts[2*y_to_remove] == 0:
                    del counts[2*y_to_remove]
                for z in chosen:
                    if z != y_to_remove:
                        s = y_to_remove + z
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                
                # Check x
                valid = True
                if counts.get(2*x_to_add, 0) >= 2:
                    valid = False
                else:
                    for z in chosen:
                        if z != y_to_remove:
                            s = x_to_add + z
                            if counts.get(s, 0) >= 2:
                                valid = False
                                break
                
                if valid:
                    # Add x
                    counts[2*x_to_add] = counts.get(2*x_to_add, 0) + 1
                    for z in chosen:
                        if z != y_to_remove:
                            s = x_to_add + z
                            counts[s] = counts.get(s, 0) + 1
                    
                    chosen.remove(y_to_remove)
                    chosen.append(x_to_add)
                    chosen_set.remove(y_to_remove)
                    chosen_set.add(x_to_add)
                    candidates.remove(x_to_add)
                    candidates.append(y_to_remove)
                    swap_done = True
                    break
                else:
                    # Revert
                    counts.clear()
                    counts.update(old_counts)
            
            if not swap_done:
                # Try a more aggressive kick: remove a random element and refill
                if len(chosen) > 10:
                    y_to_remove = random.choice(chosen)
                    old_counts = dict(counts)
                    
                    counts[2*y_to_remove] -= 1
                    if counts[2*y_to_remove] == 0:
                        del counts[2*y_to_remove]
                    for z in chosen:
                        if z != y_to_remove:
                            s = y_to_remove + z
                            counts[s] -= 1
                            if counts[s] == 0:
                                del counts[s]
                    
                    chosen.remove(y_to_remove)
                    chosen_set.remove(y_to_remove)
                    candidates.append(y_to_remove)
                    
                    # Try to refill
                    refill_success = False
                    for x in candidates[:]:
                        if time.time() > time_limit:
                            break
                        if counts.get(2*x, 0) >= 2:
                            continue
                        valid = True
                        for z in chosen:
                            s = x + z
                            if counts.get(s, 0) >= 2:
                                valid = False
                                break
                        if valid:
                            counts[2*x] = counts.get(2*x, 0) + 1
                            for z in chosen:
                                s = x + z
                                counts[s] = counts.get(s, 0) + 1
                            chosen.append(x)
                            chosen_set.add(x)
                            candidates.remove(x)
                            refill_success = True
                            break
                    
                    if not refill_success and len(chosen) < len(initial_set):
                        # Revert
                        counts.clear()
                        counts.update(old_counts)
                        chosen.append(y_to_remove)
                        chosen_set.add(y_to_remove)
                        candidates.remove(y_to_remove)
                        break
            
            if time.time() > time_limit:
                break
        
        return chosen, counts
    
    # Main loop
    iteration = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Greedy with varying noise
        noise = random.uniform(0, 1.5)
        greedy_time = min(remaining * 0.3, 5.0)
        
        baseline = fast_greedy(noise=noise, time_limit=time.time() + greedy_time)
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = baseline
        
        # Local search refinement
        ls_remaining = deadline - time.time()
        if ls_remaining < 2:
            break
        
        ls_time = min(ls_remaining * 0.6, 10.0)
        start_ls = time.time() + ls_time
        
        refined, _ = local_search(best_set if len(best_set) >= len(baseline) else baseline, start_ls)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined
        
        iteration += 1
    
    return best_set
