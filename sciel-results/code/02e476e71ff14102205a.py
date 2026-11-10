import time
import random
import numpy as np

def _greedy_build(order, deadline, counts_arr, max_elements=60):
    """Fast greedy using numpy for vectorized checks."""
    chosen = []
    chosen_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if len(chosen) >= max_elements:
            break
        
        if chosen:
            new_sums = np.array(chosen) + x
            new_sums = np.append(new_sums, 2 * x)
            indices = new_sums - 1
            valid_indices = (indices >= 0) & (indices < len(counts_arr))
            if not np.all(valid_indices):
                continue
            if np.any(counts_arr[indices] >= 2):
                continue
        else:
            s = 2 * x
            idx = s - 1
            if not (0 <= idx < len(counts_arr)):
                continue
            if counts_arr[idx] >= 2:
                continue
        
        # Valid addition
        chosen.append(x)
        chosen_set.add(x)
        if chosen:
            new_sums = np.array(chosen) + x
            new_sums = np.append(new_sums, 2 * x)
        else:
            new_sums = np.array([2 * x])
        indices = new_sums - 1
        counts_arr[indices] += 1
    
    return chosen, counts_arr

def _local_search_sa(chosen, counts_arr, deadline, max_iter=5000):
    """Simulated annealing style local search with fast numpy checks."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    counts = counts_arr.copy()
    
    best_size = len(chosen)
    best_set = list(chosen)
    
    all_elements = set(range(1, 1001))
    candidates = list(all_elements - chosen_set)
    
    temp = 1.0
    cooling = 0.999
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        temp *= cooling
        
        if not candidates:
            break
        
        op = random.random()
        
        if op < 0.3 and len(chosen) < 60:
            x = random.choice(candidates)
            
            if chosen:
                new_sums = np.array(chosen) + x
                new_sums = np.append(new_sums, 2 * x)
                indices = new_sums - 1
                valid_indices = (indices >= 0) & (indices < len(counts))
                if not np.all(valid_indices):
                    continue
                if np.any(counts[indices] >= 2):
                    continue
            else:
                s = 2 * x
                idx = s - 1
                if not (0 <= idx < len(counts)):
                    continue
                if counts[idx] >= 2:
                    continue
            
            chosen.append(x)
            chosen_set.add(x)
            new_sums = np.array(chosen) + x
            new_sums = np.append(new_sums, 2 * x)
            indices = new_sums - 1
            counts[indices] += 1
            candidates.remove(x)
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = list(chosen)
        
        elif op < 0.6 and len(chosen) > 5:
            i = random.randint(0, len(chosen) - 1)
            x = chosen[i]
            
            removed_sums = []
            for j, y in enumerate(chosen):
                if j == i:
                    s = 2 * x
                else:
                    s = x + y
                removed_sums.append(s)
            
            for s in removed_sums:
                idx = s - 1
                if 0 <= idx < len(counts):
                    counts[idx] -= 1
            
            chosen.pop(i)
            chosen_set.discard(x)
            candidates.append(x)
            
            if random.random() < temp * 0.5:
                chosen.insert(i, x)
                chosen_set.add(x)
                for s in removed_sums:
                    idx = s - 1
                    if 0 <= idx < len(counts):
                        counts[idx] += 1
                candidates.remove(x)
        
        else:
            if not chosen or not candidates:
                continue
            
            i = random.randint(0, len(chosen) - 1)
            x = chosen[i]
            
            for _ in range(min(5, len(candidates))):
                y = random.choice(candidates)
                
                removed_sums = []
                for j, z in enumerate(chosen):
                    if j == i:
                        s = 2 * x
                    else:
                        s = x + z
                    removed_sums.append(s)
                
                can_remove = True
                for s in removed_sums:
                    idx = s - 1
                    if not (0 <= idx < len(counts)) or counts[idx] < 1:
                        can_remove = False
                        break
                
                if not can_remove:
                    continue
                
                for s in removed_sums:
                    idx = s - 1
                    counts[idx] -= 1
                chosen.pop(i)
                chosen_set.discard(x)
                
                if chosen:
                    new_sums = np.array(chosen) + y
                    new_sums = np.append(new_sums, 2 * y)
                    indices = new_sums - 1
                    valid_indices = (indices >= 0) & (indices < len(counts))
                    if not np.all(valid_indices):
                        chosen.insert(i, x)
                        chosen_set.add(x)
                        for s in removed_sums:
                            idx = s - 1
                            counts[idx] += 1
                        continue
                    if np.any(counts[indices] >= 2):
                        chosen.insert(i, x)
                        chosen_set.add(x)
                        for s in removed_sums:
                            idx = s - 1
                            counts[idx] += 1
                        continue
                else:
                    s = 2 * y
                    idx = s - 1
                    if not (0 <= idx < len(counts)) or counts[idx] >= 2:
                        chosen.insert(i, x)
                        chosen_set.add(x)
                        for s in removed_sums:
                            idx = s - 1
                            counts[idx] += 1
                        continue
                
                chosen.append(y)
                chosen_set.add(y)
                if chosen:
                    new_sums = np.array(chosen) + y
                    new_sums = np.append(new_sums, 2 * y)
                else:
                    new_sums = np.array([2 * y])
                indices = new_sums - 1
                counts[indices] += 1
                
                candidates.remove(y)
                candidates.append(x)
                
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_set = list(chosen)
                
                break
    
    return best_set, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_set = []
    best_size = 0
    
    counts_arr_size = 2001
    
    counts1 = np.zeros(counts_arr_size, dtype=np.int32)
    s1, _ = _greedy_build(range(1, 1001), deadline, counts1)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    counts2 = np.zeros(counts_arr_size, dtype=np.int32)
    s2, _ = _greedy_build(range(1000, 0, -1), deadline, counts2)
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0:
        order = list(range(1, 1001))
        rng.shuffle(order)
        
        counts_r = np.zeros(counts_arr_size, dtype=np.int32)
        s, _ = _greedy_build(order, deadline, counts_r)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        
        if time.time() < deadline - 10.0:
            counts_ls = np.zeros(counts_arr_size, dtype=np.int32)
            for i, a in enumerate(best_set):
                for b in best_set[i:]:
                    s_val = a + b
                    idx = s_val - 1
                    if 0 <= idx < len(counts_ls):
                        counts_ls[idx] += 1
            
            improved, _ = _local_search_sa(best_set, counts_ls, deadline, max_iter=2000)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
        
        if best_size >= 55:
            break
    
    return best_set
