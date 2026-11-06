import time
import random

N = 64
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

def squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = []
    best_size = 0
    
    def try_add(p_idx, s_indices, dist_sets):
        p = ALL_POINTS[p_idx]
        seen_dists = set()
        for i in range(len(s_indices)):
            q_idx = s_indices[i]
            q = ALL_POINTS[q_idx]
            d = squared_dist(p, q)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            if d in dist_sets[i]:
                return False
        return True

    def add_point(p_idx, s_indices, dist_sets):
        p = ALL_POINTS[p_idx]
        new_dists = set()
        for i in range(len(s_indices)):
            q_idx = s_indices[i]
            q = ALL_POINTS[q_idx]
            d = squared_dist(p, q)
            new_dists.add(d)
            dist_sets[i].add(d)
        s_indices.append(p_idx)
        dist_sets.append(new_dists)

    def remove_point(idx_to_remove, s_indices, dist_sets):
        q_idx = s_indices[idx_to_remove]
        q = ALL_POINTS[q_idx]
        
        for j in range(len(s_indices)):
            if j == idx_to_remove:
                continue
            p_idx = s_indices[j]
            p = ALL_POINTS[p_idx]
            d = squared_dist(q, p)
            dist_sets[j].remove(d)
            
        dist_sets.pop(idx_to_remove)
        s_indices.pop(idx_to_remove)

    # Phase 1: Greedy
    num_greedy = 2000
    indices_pool = list(range(NUM_POINTS))
    
    for _ in range(num_greedy):
        if time.time() > deadline:
            break
        random.shuffle(indices_pool)
        
        s_indices = []
        dist_sets = []
        
        for idx in indices_pool:
            if try_add(idx, s_indices, dist_sets):
                add_point(idx, s_indices, dist_sets)
                
        if len(s_indices) > best_size:
            best_size = len(s_indices)
            best_set = s_indices[:]

    # Phase 2: Local Search
    if best_size > 0:
        s_indices = best_set[:]
        
        # Rebuild dist_sets
        dist_sets = [set() for _ in s_indices]
        for i in range(len(s_indices)):
            pi = ALL_POINTS[s_indices[i]]
            for j in range(len(s_indices)):
                if i == j: continue
                pj = ALL_POINTS[s_indices[j]]
                d = squared_dist(pi, pj)
                dist_sets[i].add(d)
                
        # Setup inactive structure
        active_set = set(s_indices)
        inactive_list = [i for i in range(NUM_POINTS) if i not in active_set]
        pos_in_inactive = {i: idx for idx, i in enumerate(inactive_list)}
        
        def remove_from_inactive(p_idx):
            pos = pos_in_inactive[p_idx]
            last_idx = len(inactive_list) - 1
            last_item = inactive_list[last_idx]
            
            if pos != last_idx:
                inactive_list[pos] = last_item
                pos_in_inactive[last_item] = pos
            inactive_list.pop()
            del pos_in_inactive[p_idx]
            
        def add_to_inactive(p_idx):
            inactive_list.append(p_idx)
            pos_in_inactive[p_idx] = len(inactive_list) - 1

        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            if not inactive_list:
                break
                
            # 1. Try Add
            p_idx = random.choice(inactive_list)
            if try_add(p_idx, s_indices, dist_sets):
                add_point(p_idx, s_indices, dist_sets)
                remove_from_inactive(p_idx)
                active_set.add(p_idx)
                no_improve_count = 0
                if len(s_indices) > best_size:
                    best_size = len(s_indices)
                    best_set = s_indices[:]
                continue
                
            # 2. Try Swap
            if s_indices:
                q_pos = random.randint(0, len(s_indices) - 1)
                q_idx = s_indices[q_pos]
                
                # Remove q
                remove_point(q_pos, s_indices, dist_sets)
                active_set.remove(q_idx)
                add_to_inactive(q_idx)
                
                # Try to add a random inactive point
                # We might pick q_idx again, which is fine (revert)
                # Or pick another
                p_idx = random.choice(inactive_list)
                
                if try_add(p_idx, s_indices, dist_sets):
                    add_point(p_idx, s_indices, dist_sets)
                    remove_from_inactive(p_idx)
                    active_set.add(p_idx)
                    no_improve_count = 0
                    if len(s_indices) > best_size:
                        best_size = len(s_indices)
                        best_set = s_indices[:]
                else:
                    # Revert
                    add_point(q_idx, s_indices, dist_sets)
                    remove_from_inactive(q_idx)
                    active_set.add(q_idx)
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                # Reset to best
                s_indices = best_set[:]
                dist_sets = [set() for _ in s_indices]
                for i in range(len(s_indices)):
                    pi = ALL_POINTS[s_indices[i]]
                    for j in range(len(s_indices)):
                        if i == j: continue
                        pj = ALL_POINTS[s_indices[j]]
                        d = squared_dist(pi, pj)
                        dist_sets[i].add(d)
                        
                active_set = set(s_indices)
                inactive_list = [i for i in range(NUM_POINTS) if i not in active_set]
                pos_in_inactive = {i: idx for idx, i in enumerate(inactive_list)}
                no_improve_count = 0

    return [ALL_POINTS[i] for i in best_set]
