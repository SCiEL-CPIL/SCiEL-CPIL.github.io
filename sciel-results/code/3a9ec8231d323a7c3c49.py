import time
import random
import numpy as np

MAX_VAL = 1000
MAX_SUM = 2 * MAX_VAL

def _greedy_random(seed):
    rng = random.Random(seed)
    candidates = list(range(1, MAX_VAL + 1))
    rng.shuffle(candidates)
    
    chosen = []
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    
    for x in candidates:
        if chosen:
            sums = x + np.array(chosen)
            if np.any(counts[sums] >= 2):
                continue
        if counts[2 * x] >= 2:
            continue
            
        if chosen:
            counts[x + np.array(chosen)] += 1
        counts[2 * x] += 1
        chosen.append(x)
        
    return chosen

def _local_search(initial, deadline, seed=42):
    rng = random.Random(seed)
    current = list(initial)
    current_set = set(current)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    
    if current:
        arr = np.array(current)
        for i in range(len(arr)):
            counts[arr[i] + arr[i:]] += 1
            
    best = list(current)
    best_size = len(current)
    all_vals = list(range(1, MAX_VAL + 1))
    
    while time.time() < deadline:
        op = rng.random()
        
        if op < 0.5:
            # Insert
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            valid = True
            if current:
                sums = x + np.array(current)
                if np.any(counts[sums] >= 2):
                    valid = False
            if valid and counts[2 * x] >= 2:
                valid = False
                
            if valid:
                if current:
                    counts[x + np.array(current)] += 1
                counts[2 * x] += 1
                current.append(x)
                current_set.add(x)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    
        elif op < 0.9:
            # Swap
            if not current:
                continue
            remove_x = rng.choice(current)
            add_x = rng.choice(all_vals)
            if add_x == remove_x or add_x in current_set:
                continue
            
            others = [v for v in current if v != remove_x]
            others_arr = np.array(others) if others else np.array([])
            
            # Undo remove_x
            if others:
                counts[remove_x + others_arr] -= 1
            counts[2 * remove_x] -= 1
            
            # Check add_x
            valid = True
            if others:
                sums = add_x + others_arr
                if np.any(counts[sums] >= 2):
                    valid = False
            if valid and counts[2 * add_x] >= 2:
                valid = False
                
            if valid:
                if others:
                    counts[add_x + others_arr] += 1
                counts[2 * add_x] += 1
                current = others + [add_x]
                current_set.discard(remove_x)
                current_set.add(add_x)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
            else:
                # Revert
                if others:
                    counts[remove_x + others_arr] += 1
                counts[2 * remove_x] += 1
                current = others + [remove_x]
                current_set.add(remove_x)
                
        else:
            # Remove 1, Insert 2 (Net +1)
            if not current:
                continue
            remove_x = rng.choice(current)
            others = [v for v in current if v != remove_x]
            others_arr = np.array(others) if others else np.array([])
            
            # Undo remove_x
            if others:
                counts[remove_x + others_arr] -= 1
            counts[2 * remove_x] -= 1
            
            added = []
            for _ in range(2):
                add_x = rng.choice(all_vals)
                if add_x in current_set or add_x in added or add_x == remove_x:
                    continue
                
                # Check against others + added
                temp_set = others + added
                temp_arr = np.array(temp_set) if temp_set else np.array([])
                
                valid = True
                if temp_set:
                    sums = add_x + temp_arr
                    if np.any(counts[sums] >= 2):
                        valid = False
                if valid and counts[2 * add_x] >= 2:
                    valid = False
                    
                if valid:
                    # Commit to added list and counts
                    if temp_set:
                        counts[add_x + temp_arr] += 1
                    counts[2 * add_x] += 1
                    added.append(add_x)
            
            if len(added) == 2:
                # Success: we removed 1 and added 2
                current = others + added
                current_set = set(current)
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
            else:
                # Revert: remove 'added' from counts, restore 'remove_x'
                for ax in added:
                    # We need to decrement counts for ax + (others + added_before_ax)
                    # This is tricky to revert exactly if we don't store the state.
                    # Simpler: Revert by rebuilding counts? No, too slow.
                    # Alternative: If it fails, just revert the 'added' items.
                    # Since we only added up to 2, we can just subtract their contributions.
                    # But their contributions depend on the set at the time of addition.
                    # Item 1 was added to 'others'. Item 2 was added to 'others' + [Item 1].
                    # So to revert Item 2: subtract (Item 2 + others) and (Item 2 + Item 1) and 2*Item 2.
                    # To revert Item 1: subtract (Item 1 + others) and 2*Item 1.
                    
                    # Let's just revert the last added item if any.
                    if added:
                        last = added.pop()
                        # Revert last
                        # It was added to 'others' + added[:-1]
                        prev_added = added
                        temp_set = others + prev_added
                        temp_arr = np.array(temp_set) if temp_set else np.array([])
                        if temp_set:
                            counts[last + temp_arr] -= 1
                        counts[2 * last] -= 1
                        # If we added 2 items, we need to revert the first one too.
                        # The loop structure above commits immediately.
                        # This revert logic is getting messy.
                        pass
                # Restore remove_x
                if others:
                    counts[remove_x + others_arr] += 1
                counts[2 * remove_x] += 1
                current = others + [remove_x]
                current_set.add(remove_x)

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_set = []
    best_size = 0
    
    # Greedy Phase
    greedy_deadline = start_time + 20
    seed = 1
    while time.time() < greedy_deadline:
        s = _greedy_random(seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        seed += 1
        
    if time.time() > deadline:
        return best_set
        
    # Local Search Phase
    result = _local_search(best_set, deadline, seed=42)
    if len(result) > best_size:
        best_set = result
        best_size = len(result)
        
    return best_set
