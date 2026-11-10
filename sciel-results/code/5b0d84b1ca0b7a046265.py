import time
import random
import math

N = 64
TIME_LIMIT = 290 # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def rebuild_dist_from(selected):
        dist_from = []
        for i in range(len(selected)):
            s = set()
            pi = selected[i]
            for j in range(len(selected)):
                if i == j: continue
                pj = selected[j]
                s.add(dist2(pi, pj))
            dist_from.append(s)
        return dist_from

    def try_add(p, selected, dist_from):
        d_p = []
        for pt in selected:
            d_p.append(dist2(p, pt))
            
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
        # We iterate over all other points j
        for j in range(len(selected)):
            if j == idx: continue
            d = dist2(p, selected[j])
            dist_from[j].discard(d)
            
        selected.pop(idx)
        dist_from.pop(idx)

    def tabu_search(initial_set, initial_dist, max_iterations=2000, tabu_tenure=50):
        selected = list(initial_set)
        dist_from = [set(s) for s in initial_dist]
        
        current_size = len(selected)
        local_best = list(selected)
        local_best_size = current_size
        selected_set = set(selected)
        
        # Tabu list: store (removed_point, added_point)
        # We use a dictionary to count tabu status (time when it becomes free)
        tabu_list = {} 
        # To avoid storing huge tuples, we can use indices or just the points.
        # Points are tuples of ints, hashable.
        
        # We will store the move (p_out, p_in) and the iteration it becomes non-tabu
        # Actually, simpler: store a set of (p_out, p_in) that are tabu.
        # But we need to age them.
        # Let's use a list of (iteration, p_out, p_in) and prune.
        tabu_moves = [] # (iteration, p_out, p_in)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Prune tabu list
            # Keep only moves that are still tabu
            # Tabu tenure is tabu_tenure
            while tabu_moves and tabu_moves[0][0] + tabu_tenure <= iteration:
                tabu_moves.pop(0)
            
            tabu_set = set((m[1], m[2]) for m in tabu_moves)
            
            # 1. Try to grow
            added = False
            # Try a few random candidates
            for _ in range(20):
                p = rng.choice(all_points)
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
            
            # 2. Try 1-Swap with Tabu
            swapped = False
            if current_size > 0:
                # Pick a random point to remove
                idx_u = rng.randint(0, current_size - 1)
                u = selected[idx_u]
                
                # Try several candidates for v
                for _ in range(20):
                    v = rng.choice(all_points)
                    if v in selected_set:
                        continue
                    if v == u:
                        continue
                    
                    # Check if move (u, v) is tabu
                    if (u, v) in tabu_set:
                        # Aspiration criterion: if this move leads to a new best, accept it
                        # But we don't know if it leads to a new best without trying.
                        # Since size doesn't change, it can't be a "new best" in size.
                        # So we just skip tabu moves.
                        continue
                    
                    # Try the swap
                    try_remove(idx_u, selected, dist_from)
                    selected_set.discard(u)
                    current_size -= 1
                    
                    if try_add(v, selected, dist_from):
                        selected_set.add(v)
                        current_size += 1
                        swapped = True
                        
                        # Add to tabu list
                        tabu_moves.append((iteration, u, v))
                        
                        # We don't update local_best here as size is same
                        break
                    
                    # If add failed, restore u
                    if not try_add(u, selected, dist_from):
                         # Should not happen
                         pass
                    selected_set.add(u)
                    current_size += 1
                    
                    if swapped:
                        break
                
                if swapped:
                    continue

            # 3. If stuck, try a 2-Swap (non-tabu logic for simplicity, or just random)
            if current_size >= 2:
                idx_u = rng.randint(0, current_size - 1)
                idx_v = rng.randint(0, current_size - 1)
                if idx_u == idx_v:
                    idx_v = (idx_v + 1) % current_size
                    
                u = selected[idx_u]
                v = selected[idx_v]
                
                selected_backup = list(selected)
                dist_from_backup = [set(s) for s in dist_from]
                selected_set_backup = set(selected_set)
                size_backup = current_size
                
                try_remove(idx_u, selected, dist_from)
                selected_set.discard(u)
                current_size -= 1
                
                if idx_v > idx_u:
                    idx_v -= 1
                    
                try_remove(idx_v, selected, dist_from)
                selected_set.discard(v)
                current_size -= 1
                
                swap2_success = False
                for _ in range(10):
                    w = rng.choice(all_points)
                    x = rng.choice(all_points)
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
        candidates = list(range(len(all_points)))
        rng.shuffle(candidates)
        
        for idx in candidates:
            p = all_points[idx]
            if try_add(p, selected, dist_from):
                pass # Added
            if time.time() > deadline - 5:
                break
        
        # Now selected is a maximal set (or close to it).
        current_size = len(selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(selected)
        
        # 2. Local Search (Tabu)
        if time.time() < deadline - 20:
            # Rebuild dist_from to be safe
            dist_from = rebuild_dist_from(selected)
            
            local_best, local_best_size = tabu_search(selected, dist_from)
            
            if local_best_size > best_size:
                best_size = local_best_size
                best_set = list(local_best)

    return best_set

# Test/Verify
if __name__ == "__main__":
    res = construct()
    print(len(res))
