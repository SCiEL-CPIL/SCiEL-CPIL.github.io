import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            if dd <= N:
                aps.append((a - 1, b - 1, c - 1, dd - 1))
                
    ap_info = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in range(4):
            ap_info[ap[pos]].append((idx, pos))

    def solve_return(k_colors, time_limit):
        colors = [-1] * N
        unassigned = set(range(N))
        
        def get_valid_colors(x):
            valid = []
            for c in range(k_colors):
                ok = True
                for ap_idx, pos in ap_info[x]:
                    ap = aps[ap_idx]
                    known = []
                    for p in range(4):
                        if p == pos:
                            known.append(c)
                        else:
                            known.append(colors[ap[p]])
                    
                    if all(v != -1 for v in known):
                        if len(set(known)) < 3:
                            ok = False
                            break
                    else:
                        vals = [v for v in known if v != -1]
                        if len(vals) == 3:
                            if len(set(vals)) == 1:
                                ok = False
                                break
                if ok:
                    valid.append(c)
            return valid

        def dfs():
            if time.time() > time_limit:
                return False
            if not unassigned:
                return True
            
            # Fail-first: pick x with fewest valid colors
            min_count = k_colors + 1
            best_x = -1
            best_valid = []
            
            for x in unassigned:
                valid = get_valid_colors(x)
                if len(valid) < min_count:
                    min_count = len(valid)
                    best_x = x
                    best_valid = valid
                    if min_count == 0:
                        return False
                    if min_count == 1:
                        break # Can't get better than 1
            
            if best_x == -1:
                return False

            # Randomize order of best_valid
            random.shuffle(best_valid)
            
            for c in best_valid:
                colors[best_x] = c
                unassigned.remove(best_x)
                
                if dfs():
                    return True
                
                colors[best_x] = -1
                unassigned.add(best_x)
            
            return False

        if dfs():
            return colors
        return None

    # Try 3 colors
    t_start = time.time()
    limit_3 = t_start + 10.0
    sol = solve_return(3, limit_3)
    if sol:
        return sol

    # Try 4 colors
    t_start = time.time()
    limit_4 = t_start + (deadline - t_start)
    sol = solve_return(4, limit_4)
    if sol:
        return sol

    # Fallback
    return [i % 10 for i in range(N)]
