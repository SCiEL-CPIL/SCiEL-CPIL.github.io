import time
import random
import math
import numpy as np

N = 64
TIME_LIMIT = 290 # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Grid points as indices 0..N^2-1
    # x = idx % N, y = idx // N
    all_indices = list(range(N * N))
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def idx_to_pt(idx):
        return (idx % N, idx // N)
    
    def dist2_idx(i, j):
        x1, y1 = i % N, i // N
        x2, y2 = j % N, j // N
        dx = x1 - x2
        dy = y1 - y2
        return dx * dx + dy * dy

    def rebuild_dist_from(selected):
        dist_from = []
        for i in range(len(selected)):
            s = set()
            pi = selected[i]
            for j in range(len(selected)):
                if i == j: continue
                pj = selected[j]
                s.add(dist2_idx(pi, pj))
            dist_from.append(s)
        return dist_from

    def try_add(p, selected, dist_from):
        d_p = []
        for pt in selected:
            d_p.append(dist2_idx(p, pt))
            
        if len(set(d_p)) != len(d_p):
            return False
            
        for i in range(len(selected)):
            if d_p[i] in dist_from[i]:
                return False
                
        selected.append(p)
        new_dist_set = set(d_p)
        dist_from.append(new_dist_set)
        for i in range(len(selected) - 1):
            dist_from[i].add(d_p[i])
            
        return True

    def try_remove(idx, selected, dist_from):
        p = selected[idx]
        
        # Remove distances involving p from other points' sets
        for j in range(len(selected)):
            if j == idx: continue
            d = dist2_idx(p, selected[j])
            dist_from[j].discard(d)
            
        selected.pop(idx)
        dist_from.pop(idx)

    def local_search(initial_set, initial_dist):
        selected = list(initial_set)
        dist_from = [set(s) for s in initial_dist]
        
        current_size = len(selected)
        local_best = list(selected)
        local_best_size = current_size
        selected_set = set(selected)
        
        max_iterations = 10000
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # 1. Try to add a point (Net +1)
            added = False
            for _ in range(100):
                p = rng.choice(all_indices)
                if p in selected_set:
                    continue
                if try_add(p, selected, dist_from):
                    selected_set.add(p)
                    current_size += 1
                    added = True
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best = list(selected)
                    break
            
            if added:
                continue

            # 2. Try 1-Remove-2-Add (Net +1)
            # Remove a random point u, then try to add two points v, w
            if current_size > 0:
                idx_u = rng.randint(0, current_size - 1)
                u = selected[idx_u]
                
                # Save state for rollback if we don't achieve net +1
                selected_backup = list(selected)
                dist_from_backup = [set(s) for s in dist_from]
                selected_set_backup = set(selected_set)
                size_backup = current_size
                
                # Remove u
                try_remove(idx_u, selected, dist_from)
                selected_set.discard(u)
                current_size -= 1
                
                added_v = False
                added_w = False
                v_idx = -1
                w_idx = -1
                
                # Try to add v
                for _ in range(50):
                    v = rng.choice(all_indices)
                    if v in selected_set or v == u:
                        continue
                    if try_add(v, selected, dist_from):
                        selected_set.add(v)
                        current_size += 1
                        added_v = True
                        v_idx = len(selected) - 1
                        break
                
                if added_v:
                    # Try to add w
                    for _ in range(50):
                        w = rng.choice(all_indices)
                        if w in selected_set or w == u:
                            continue
                        if try_add(w, selected, dist_from):
                            selected_set.add(w)
                            current_size += 1
                            added_w = True
                            break
                
                if added_w:
                    # Net +1 achieved. Keep the change.
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best = list(selected)
                else:
                    # Rollback to state before removing u
                    selected = selected_backup
                    dist_from = dist_from_backup
                    selected_set = selected_set_backup
                    current_size = size_backup
                    continue # Try next move type

            # 3. Try 1-Swap (Net 0) - Escape local minima
            swapped = False
            if current_size > 0:
                idx_u = rng.randint(0, current_size - 1)
                u = selected[idx_u]
                
                try_remove(idx_u, selected, dist_from)
                selected_set.discard(u)
                current_size -= 1
                
                # Try to add a random point
                for _ in range(100):
                    v = rng.choice(all_indices)
                    if v in selected_set:
                        continue
                    if try_add(v, selected, dist_from):
                        selected_set.add(v)
                        current_size += 1
                        swapped = True
                        break
                
                if not swapped:
                    # Re-add u
                    try_add(u, selected, dist_from)
                    selected_set.add(u)
                    current_size += 1
                else:
                    continue

            # 4. Try 2-Swap (Net 0)
            if current_size >= 2:
                idx_u = rng.randint(0, current_size - 1)
                idx_v = rng.randint(0, current_size - 1)
                if idx_u == idx_v:
                    idx_v = (idx_v + 1) % current_size
                    
                u = selected[idx_u]
                v = selected[idx_v]
                
                # Save state
                selected_backup = list(selected)
                dist_from_backup = [set(s) for s in dist_from]
                selected_set_backup = set(selected_set)
                size_backup = current_size
                
                # Remove u
                try_remove(idx_u, selected, dist_from)
                selected_set.discard(u)
                current_size -= 1
                
                # Adjust idx_v
                if idx_v > idx_u:
                    idx_v -= 1
                    
                # Remove v
                try_remove(idx_v, selected, dist_from)
                selected_set.discard(v)
                current_size -= 1
                
                # Try to add w and x
                swap2_success = False
                for _ in range(50):
                    w = rng.choice(all_indices)
                    x = rng.choice(all_indices)
                    if w in selected_set or x in selected_set or w == x:
                        continue
                    
                    if try_add(w, selected, dist_from):
                        selected_set.add(w)
                        current_size += 1
                        if try_add(x, selected, dist_from):
                            selected_set.add(x)
                            current_size += 1
                            swap2_success = True
                            break
                        else:
                            idx_w = selected.index(w)
                            try_remove(idx_w, selected, dist_from)
                            selected_set.discard(w)
                            current_size -= 1
                
                if swap2_success:
                    continue
                else:
                    selected = selected_backup
                    dist_from = dist_from_backup
                    selected_set = selected_set_backup
                    current_size = size_backup
            
        return local_best, local_best_size

    # Main Loop: Multiple Restarts
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # 1. Generate initial set via Randomized Greedy
        selected = []
        dist_from = []
        candidates = list(all_indices)
        rng.shuffle(candidates)
        
        for idx in candidates:
            p = idx
            if try_add(p, selected, dist_from):
                pass
            if time.time() > deadline - 5:
                break
        
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(selected)
        
        # 2. Local Search
        if time.time() < deadline - 20:
            dist_from = rebuild_dist_from(selected)
            
            local_best, local_best_size = local_search(selected, dist_from)
            
            if local_best_size > best_size:
                best_size = local_best_size
                best_set = list(local_best)

    # Convert indices back to (x, y) tuples
    return [idx_to_pt(idx) for idx in best_set]

# Test/Verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
