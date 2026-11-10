import time
import random
import numpy as np
from collections import defaultdict

def _greedy_build(candidates, N=1000):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    chosen_set = set()
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    
    for x in candidates:
        if x in chosen_set or x < 1 or x > N:
            continue
        # Check if x can be added
        valid = True
        if counts[2 * x] >= 2:
            valid = False
        else:
            for y in chosen:
                if counts[x + y] >= 2:
                    valid = False
                    break
        if valid:
            chosen.append(x)
            chosen_set.add(x)
            counts[2 * x] += 1
            for y in chosen:
                if y != x:
                    counts[x + y] += 1
    
    return chosen, counts

def _local_search_numpy(chosen, counts, N=1000, time_limit=100.0, seed=42):
    """Fast NumPy-based local search with swaps and insertions."""
    start_time = time.time()
    rng = random.Random(seed)
    
    current_S = set(chosen)
    S_arr = np.array(sorted(chosen), dtype=np.int32)
    current_counts = counts.copy()
    best_S = frozenset(current_S)
    best_size = len(current_S)
    
    # Precompute: for each candidate y, how many conflicts would it create
    # This helps prioritize which y to try first
    
    def try_add_y(y):
        """Try adding y to current set. Returns (success, new_counts, new_S_arr)."""
        old_counts = current_counts.copy()
        
        # Tentative add
        current_counts[2 * y] += 1
        sums = y + S_arr
        current_counts[sums] += 1
        
        if np.any(current_counts > 2):
            current_counts = old_counts
            return False, None, None
        else:
            return True, current_counts, None
    
    def try_swap(x, y):
        """Try swapping x out and y in. Returns (success, new_counts, new_S_arr)."""
        old_counts = current_counts.copy()
        
        # Remove x
        current_counts[2 * x] -= 1
        mask = S_arr != x
        if np.any(mask):
            sums_x = x + S_arr[mask]
            current_counts[sums_x] -= 1
        
        # Update S_arr
        new_S_arr = S_arr[mask]
        
        # Add y
        current_counts[2 * y] += 1
        sums_y = y + new_S_arr
        current_counts[sums_y] += 1
        
        if np.any(current_counts > 2):
            current_counts = old_counts
            return False, None, None
        else:
            return True, current_counts, new_S_arr
    
    # Phase 1: Try to insert new elements
    insert_attempts = 0
    max_insert_attempts = 50000
    
    while time.time() - start_time < time_limit - 20 and insert_attempts < max_insert_attempts:
        # Generate candidates: prioritize values not in S
        candidates = [y for y in range(1, N + 1) if y not in current_S]
        
        if not candidates:
            break
        
        # Shuffle but with some structure: try nearby values first
        rng.shuffle(candidates)
        
        found = False
        for y in candidates:
            if time.time() - start_time > time_limit - 20:
                break
            
            success, new_counts, _ = try_add_y(y)
            if success:
                current_S.add(y)
                S_arr = np.append(S_arr, y)
                current_counts = new_counts
                insert_attempts += 1
                
                if len(current_S) > best_size:
                    best_S = frozenset(current_S)
                    best_size = len(current_S)
                
                found = True
                break
        
        if not found:
            # No more insertions possible, move to swaps
            break
    
    # Phase 2: Try swaps to potentially enable more insertions or just explore
    swap_attempts = 0
    max_swap_attempts = 100000
    
    while time.time() - start_time < time_limit - 5 and swap_attempts < max_swap_attempts:
        # Pick a random element to remove
        x = rng.choice(list(current_S))
        
        # Generate candidates to add
        candidates = [y for y in range(1, N + 1) if y not in current_S]
        if not candidates:
            break
        
        rng.shuffle(candidates)
        
        found_swap = False
        for y in candidates:
            if time.time() - start_time > time_limit - 5:
                break
            
            success, new_counts, new_S_arr = try_swap(x, y)
            if success:
                # Accept swap
                current_S.remove(x)
                current_S.add(y)
                S_arr = new_S_arr
                current_counts = new_counts
                swap_attempts += 1
                
                if len(current_S) > best_size:
                    best_S = frozenset(current_S)
                    best_size = len(current_S)
                
                found_swap = True
                break
        
        if not found_swap:
            # Try a few more random x before giving up
            for _ in range(5):
                if time.time() - start_time > time_limit - 5:
                    break
                x = rng.choice(list(current_S))
                candidates = [y for y in range(1, N + 1) if y not in current_S]
                if not candidates:
                    break
                rng.shuffle(candidates)
                for y in candidates[:50]:  # Limit candidates per x
                    success, new_counts, new_S_arr = try_swap(x, y)
                    if success:
                        current_S.remove(x)
                        current_S.add(y)
                        S_arr = new_S_arr
                        current_counts = new_counts
                        swap_attempts += 1
                        
                        if len(current_S) > best_size:
                            best_S = frozenset(current_S)
                            best_size = len(current_S)
                        
                        found_swap = True
                        break
                if found_swap:
                    break
        
        if not found_swap and swap_attempts > 10000:
            # Stuck, break out
            break
    
    return list(best_S), current_counts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    N = 1000
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Standard greedy (increasing order)
    candidates1 = list(range(1, N + 1))
    chosen1, counts1 = _greedy_build(candidates1)
    if len(chosen1) > best_size:
        best_result = chosen1
        best_size = len(chosen1)
    
    # Strategy 2: Descending order
    if time.time() < deadline - 30:
        candidates2 = list(range(N, 0, -1))
        chosen2, counts2 = _greedy_build(candidates2)
        if len(chosen2) > best_size:
            best_result = chosen2
            best_size = len(chosen2)
    
    # Strategy 3: Random orderings
    if time.time() < deadline - 30:
        rng = random.Random(123)
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            candidates = list(range(1, N + 1))
            random.Random(seed).shuffle(candidates)
            chosen, counts = _greedy_build(candidates)
            if len(chosen) > best_size:
                best_result = chosen
                best_size = len(chosen)
    
    # Strategy 4: Structured ordering (sparse then fill)
    if time.time() < deadline - 30:
        candidates = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
        chosen, counts = _greedy_build(candidates)
        if len(chosen) > best_size:
            best_result = chosen
            best_size = len(chosen)
    
    # Now apply local search to the best greedy result
    if time.time() < deadline - 10 and best_result:
        # Rebuild counts for the best result
        counts_best = np.zeros(2 * N + 1, dtype=np.int32)
        for i in range(len(best_result)):
            for j in range(i, len(best_result)):
                counts_best[best_result[i] + best_result[j]] += 1
        
        refined, _ = _local_search_numpy(best_result, counts_best, N, 
                                          time_limit=min(90, deadline - time.time() - 5),
                                          seed=42)
        if len(refined) > best_size:
            best_result = refined
            best_size = len(refined)
    
    # If we have time, try local search on other good greedy results
    if time.time() < deadline - 15:
        # Try a different greedy initialization
        rng = random.Random(999)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        chosen_alt, counts_alt = _greedy_build(candidates)
        
        if len(chosen_alt) >= best_size - 2:  # Close enough to be worth refining
            refined_alt, _ = _local_search_numpy(chosen_alt, counts_alt, N,
                                                  time_limit=min(30, deadline - time.time() - 5),
                                                  seed=777)
            if len(refined_alt) > best_size:
                best_result = refined_alt
                best_size = len(refined_alt)
    
    return best_result
