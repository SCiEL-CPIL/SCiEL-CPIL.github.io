import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    aps_containing = [[] for _ in range(n)]
    
    for d in range(1, n):
        max_start = n - 3*d - 1
        for start in range(max_start + 1):
            idx1 = start
            idx2 = start + d
            idx3 = start + 2*d
            idx4 = start + 3*d
            ap = (idx1, idx2, idx3, idx4)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                aps_containing[idx].append(ap_idx)
                
    def get_distinct_count(ap, colors):
        c1 = colors[ap[0]]
        c2 = colors[ap[1]]
        c3 = colors[ap[2]]
        c4 = colors[ap[3]]
        return len({c1, c2, c3, c4})
        
    def is_bad_ap(ap_idx, colors):
        return get_distinct_count(aps[ap_idx], colors) < 3
        
    def find_solution(num_colors):
        colors = [random.randint(0, num_colors - 1) for _ in range(n)]
        
        # Initial bad APs
        bad_aps = set()
        for i in range(len(aps)):
            if is_bad_ap(i, colors):
                bad_aps.add(i)
                
        if not bad_aps:
            return colors
            
        # Local Search
        no_improve_count = 0
        max_no_improve = 500
        steps = 0
        max_steps = 500000
        
        while bad_aps and steps < max_steps and time.time() < deadline:
            steps += 1
            if time.time() > deadline:
                break
                
            # Pick a random bad AP
            ap_idx = random.choice(list(bad_aps))
            ap = aps[ap_idx]
            
            # Pick a random position in the AP to change
            pos_idx = random.randint(0, 3)
            pos = ap[pos_idx]
            
            old_color = colors[pos]
            new_color = random.randint(0, num_colors - 1)
            
            # Ensure new color is different
            while new_color == old_color:
                new_color = random.randint(0, num_colors - 1)
                
            colors[pos] = new_color
            
            # Update bad_aps
            for affected_idx in aps_containing[pos]:
                was_bad = affected_idx in bad_aps
                is_bad_now = is_bad_ap(affected_idx, colors)
                
                if was_bad and not is_bad_now:
                    bad_aps.remove(affected_idx)
                    no_improve_count = 0
                elif not was_bad and is_bad_now:
                    bad_aps.add(affected_idx)
                    no_improve_count += 1
                else:
                    if is_bad_now:
                        no_improve_count += 1
                        
            if no_improve_count > max_no_improve:
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5)
                colors = [random.randint(0, num_colors - 1) for _ in range(n)]
                bad_aps = set()
                for i in range(len(aps)):
                    if is_bad_ap(i, colors):
                        bad_aps.add(i)
                steps = 0
                
        if not bad_aps:
            return colors
        return None

    # Try 3 colors
    sol = find_solution(3)
    if sol:
        return sol
        
    # Try 4 colors
    if time.time() < deadline:
        sol = find_solution(4)
        if sol:
            return sol
            
    # Fallback: Return a random 3-coloring (likely invalid, but better than crash)
    # Actually, if it's invalid, score is 0.
    # But we should try to return *something* valid if possible.
    # Given the high probability of 3-coloring existing, this is unlikely.
    
    return [0]*n
