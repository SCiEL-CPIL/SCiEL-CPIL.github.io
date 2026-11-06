import random
import time
import sys

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    
    # Precompute coordinates
    coords = [(x, y) for y in range(N) for x in range(N)]
    
    def sq_dist(i, j):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        dx = x1 - x2
        dy = y1 - y2
        return dx*dx + dy*dy

    def try_greedy_with_swaps(seed):
        random.seed(seed)
        candidates = list(range(TOTAL_POINTS))
        random.shuffle(candidates)
        
        selected = []
        selected_set = set()
        # dist_counts[q] = dict {distance: count}
        dist_counts = {}
        
        # 1. Greedy Addition
        for p in candidates:
            if p in selected_set:
                continue
            
            # Check if p can be added
            # Cost 1: Apex p
            hist = {}
            for q in selected:
                d = sq_dist(p, q)
                hist[d] = hist.get(d, 0) + 1
            
            valid = True
            for d, c in hist.items():
                if c > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Cost 2: Apex q in selected
            for q in selected:
                d = sq_dist(p, q)
                # If d is already in dist_counts[q], then there is an r with dist(q,r)=d
                # So (q, p, r) is isosceles
                if d in dist_counts[q]:
                    valid = False
                    break
            
            if valid:
                selected.append(p)
                selected_set.add(p)
                dist_counts[p] = hist
                for q in selected:
                    if q == p: continue
                    d = sq_dist(p, q)
                    if d in dist_counts[q]:
                        dist_counts[q][d] += 1
                    else:
                        dist_counts[q][d] = 1
        
        # 2. Local Search with Swaps to improve structure
        # We try to swap points to make it "easier" to add more points later?
        # Or just to find a different valid set of same size.
        # Since we want to maximize size, we should try to ADD points after swaps.
        
        # We will do a loop of: Try Add -> If Fail, Try Swap -> Repeat
        
        # We need a list of unselected points
        unselected = [p for p in range(TOTAL_POINTS) if p not in selected_set]
        random.shuffle(unselected)
        
        # Time limit for this restart
        restart_deadline = time.time() + 10
        
        while time.time() < restart_deadline and len(unselected) > 0:
            # Try to add a random unselected point
            added_any = False
            # Try a few random candidates
            for _ in range(10):
                if not unselected:
                    break
                idx = random.randint(0, len(unselected) - 1)
                p = unselected[idx]
                
                # Check validity of adding p
                hist = {}
                for q in selected:
                    d = sq_dist(p, q)
                    hist[d] = hist.get(d, 0) + 1
                
                valid = True
                for d, c in hist.items():
                    if c > 1:
                        valid = False
                        break
                
                if valid:
                    for q in selected:
                        d = sq_dist(p, q)
                        if d in dist_counts[q]:
                            valid = False
                            break
                
                if valid:
                    # Add p
                    selected.append(p)
                    selected_set.add(p)
                    dist_counts[p] = hist
                    for q in selected:
                        if q == p: continue
                        d = sq_dist(p, q)
                        if d in dist_counts[q]:
                            dist_counts[q][d] += 1
                        else:
                            dist_counts[q][d] = 1
                    unselected.pop(idx)
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # If no add, try a swap
            # Pick random p_in from unselected, p_out from selected
            if not unselected or not selected:
                break
            
            p_in = random.choice(unselected)
            p_out = random.choice(selected)
            
            # Check if swapping p_out with p_in results in a valid set
            # This is equivalent to: Remove p_out, then try to add p_in.
            
            # 1. Remove p_out from data structures
            # We need to revert dist_counts
            # This is expensive. 
            # Alternative: Just simulate the check.
            # Check if p_in is valid in (selected \ {p_out}).
            
            # Apex p_in check:
            hist_in = {}
            for q in selected:
                if q == p_out:
                    continue
                d = sq_dist(p_in, q)
                hist_in[d] = hist_in.get(d, 0) + 1
            
            valid_swap = True
            for d, c in hist_in.items():
                if c > 1:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Apex q check for q in selected \ {p_out}
                for q in selected:
                    if q == p_out:
                        continue
                    d = sq_dist(p_in, q)
                    # We need to know if d is in dist_counts[q] EXCLUDING p_out
                    # dist_counts[q][d] is the count including p_out?
                    # Yes.
                    # If dist_counts[q][d] > 1, then there is another r != p_out with same dist.
                    # If dist_counts[q][d] == 1, then the only point with that dist is p_out.
                    # So if we remove p_out, the count becomes 0.
                    # So we should check:
                    # if d in dist_counts[q] and dist_counts[q][d] > 1:
                    #    valid = False
                    # elif d in dist_counts[q] and dist_counts[q][d] == 1:
                    #    # Only p_out had this dist. After removal, count is 0. OK.
                    #    pass
                    # else:
                    #    # d not present. OK.
                    #    pass
                    
                    if d in dist_counts[q]:
                        if dist_counts[q][d] > 1:
                            valid_swap = False
                            break
            
            if valid_swap:
                # Perform Swap
                # Remove p_out
                selected.remove(p_out)
                selected_set.remove(p_out)
                del dist_counts[p_out]
                for q in selected:
                    d = sq_dist(p_out, q)
                    dist_counts[q][d] -= 1
                    if dist_counts[q][d] == 0:
                        del dist_counts[q][d]
                
                # Add p_in
                selected.append(p_in)
                selected_set.add(p_in)
                dist_counts[p_in] = hist_in
                for q in selected:
                    if q == p_in: continue
                    d = sq_dist(p_in, q)
                    if d in dist_counts[q]:
                        dist_counts[q][d] += 1
                    else:
                        dist_counts[q][d] = 1
                
                # Update unselected
                unselected.remove(p_in)
                unselected.append(p_out)
                random.shuffle(unselected)
        
        return selected

    restart_count = 0
    while time.time() < deadline:
        result = try_greedy_with_swaps(restart_count)
        if len(result) > len(best_set):
            best_set = result
        restart_count += 1

    return [coords[i] for i in best_set]
