"""Optimized B2[2] set construction with local search."""
import numpy as np
import time

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    N = 1000
    MAX_REP = 2
    
    def count_sums(chosen):
        """Return array where arr[s] = number of representations s = a+b, a<=b, a,b in chosen."""
        if len(chosen) == 0:
            return np.zeros(2*N+1, dtype=np.int32)
        arr = np.array(chosen, dtype=np.int64)
        # All pairwise sums with a <= b
        # Use upper triangle
        i, j = np.triu_indices(len(arr))
        sums = arr[i] + arr[j]
        counts = np.bincount(sums, minlength=2*N+1)
        return counts.astype(np.int32)
    
    def max_rep(counts):
        return int(counts.max()) if len(counts) > 0 else 0
    
    def try_insert(chosen, x, counts):
        """Try to insert x into chosen. Returns (success, new_counts) or (False, None)."""
        arr = np.array(chosen, dtype=np.int64)
        if len(arr) == 0:
            new_counts = np.zeros(2*N+1, dtype=np.int32)
            new_counts[2*x] = 1
            return True, new_counts
        # New sums: x + each element in chosen, plus 2*x
        new_sums = arr + x
        new_sums = np.append(new_sums, 2*x)
        trial = counts.copy()
        trial[new_sums] += 1
        if trial.max() <= MAX_REP:
            return True, trial
        return False, None
    
    def try_remove(chosen, idx, counts):
        """Remove element at idx from chosen. Returns new_counts."""
        arr = np.array(chosen, dtype=np.int64)
        x = arr[idx]
        # Remove sums involving x
        # Sums: x + arr[j] for all j, but we need to be careful about double counting
        # Actually, the sums that involve x are: x + arr[j] for j != idx, and 2*x
        # But since we're removing x, we subtract 1 from each sum that had x
        # Sums involving x: for each y in chosen, x+y (where y != x counts once, y=x counts as 2*x)
        # More carefully: the pairs (a,b) with a<=b that include x are:
        #   (x, y) for y in chosen, y >= x  -> sum = x+y
        #   (y, x) is same as (x,y) since a<=b
        # So sums to decrement: x + y for y in chosen where y >= x, plus... 
        # Wait, let me think again. Pairs (a,b) with a<=b, a,b in chosen.
        # If we remove x, we remove all pairs where a=x or b=x.
        # Since a<=b, pairs with x are: (x, y) for y in chosen with y >= x.
        # That's it. Each such pair contributes one to count[x+y].
        # So we decrement count[x+y] for each y in chosen with y >= x.
        arr_sorted = np.sort(arr)
        # y values >= x in chosen
        mask = arr >= x
        ys = arr[mask]
        sums_to_decrement = x + ys
        new_counts = counts.copy()
        new_counts[sums_to_decrement] -= 1
        return new_counts
    
    # Phase 1: Greedy construction
    chosen = []
    counts = np.zeros(2*N+1, dtype=np.int32)
    
    for x in range(1, N+1):
        if time.time() > deadline - 30:
            break
        arr = np.array(chosen, dtype=np.int64) if chosen else np.array([], dtype=np.int64)
        if len(arr) == 0:
            new_counts = np.zeros(2*N+1, dtype=np.int32)
            new_counts[2*x] = 1
            chosen.append(x)
            counts = new_counts
            continue
        new_sums = arr + x
        new_sums = np.append(new_sums, 2*x)
        trial = counts.copy()
        trial[new_sums] += 1
        if trial.max() <= MAX_REP:
            chosen.append(x)
            counts = trial
    
    best_size = len(chosen)
    best_chosen = chosen[:]
    best_counts = counts.copy()
    
    # Phase 2: Local search with swap moves
    # Try to improve by swapping elements
    if time.time() < deadline - 10:
        # Build index
        chosen_set = set(chosen)
        
        # Try multiple rounds of swaps
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            chosen_list = chosen[:]
            counts_arr = counts.copy()
            
            for i in range(len(chosen_list)):
                if time.time() > deadline - 15:
                    break
                x_remove = chosen_list[i]
                # Try removing x_remove and inserting some other value
                new_counts_after_remove = try_remove(chosen_list, i, counts_arr)
                
                # Try inserting various candidates
                # Candidates: values not in chosen that could fit
                # Try a range of candidates around gaps
                candidates = []
                # Try values not in the set
                for x_try in range(1, N+1):
                    if x_try in chosen_set and x_try != x_remove:
                        continue
                    if x_try == x_remove:
                        continue
                    # Check if we can insert x_try after removing x_remove
                    arr_temp = np.array([v for j, v in enumerate(chosen_list) if j != i], dtype=np.int64)
                    if len(arr_temp) == 0:
                        trial = new_counts_after_remove.copy()
                        trial[2*x_try] = trial.get(2*x_try, 0) + 1 if hasattr(trial, 'get') else trial[2*x_try] + 1
                        if trial[2*x_try] <= MAX_REP:
                            # Accept swap
                            chosen_list[i] = x_try
                            chosen_set.remove(x_remove)
                            chosen_set.add(x_try)
                            counts_arr = trial
                            improved = True
                            break
                        continue
                    
                    new_sums = arr_temp + x_try
                    new_sums = np.append(new_sums, 2*x_try)
                    trial = new_counts_after_remove.copy()
                    trial[new_sums] += 1
                    if trial.max() <= MAX_REP:
                        # Accept swap
                        chosen_list[i] = x_try
                        chosen_set.remove(x_remove)
                        chosen_set.add(x_try)
                        counts_arr = trial
                        improved = True
                        break
            
            chosen = chosen_list
            counts = counts_arr
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = chosen[:]
                best_counts = counts.copy()
    
    return best_chosen
