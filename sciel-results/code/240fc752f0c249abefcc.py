import random
import time

N = 64

def construct():
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    best_size = 0
    best_set = []
    
    start_time = time.time()
    deadline = start_time + 295
    
    for seed in range(2000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        candidates = all_points[:]
        random.shuffle(candidates)
        
        S = []
        dist_sets = {}
        in_set = set()
        
        # Greedy construction
        for p in candidates:
            px, py = p
            valid = True
            dists = set()
            for a in S:
                ax, ay = a
                dx = px - ax
                dy = py - ay
                d2 = dx * dx + dy * dy
                if d2 in dists or d2 in dist_sets[a]:
                    valid = False
                    break
                dists.add(d2)
            if valid:
                S.append(p)
                dist_sets[p] = dists
                in_set.add(p)
                for a in S[:-1]:
                    ax, ay = a
                    dx = px - ax
                    dy = py - ay
                    d2 = dx * dx + dy * dy
                    dist_sets[a].add(d2)
        
        # Try to add more points
        if time.time() < deadline:
            for p in candidates:
                if p in in_set:
                    continue
                px, py = p
                valid = True
                dists = set()
                for a in S:
                    ax, ay = a
                    dx = px - ax
                    dy = py - ay
                    d2 = dx * dx + dy * dy
                    if d2 in dists or d2 in dist_sets[a]:
                        valid = False
                        break
                    dists.add(d2)
                if valid:
                    S.append(p)
                    dist_sets[p] = dists
                    in_set.add(p)
                    for a in S[:-1]:
                        ax, ay = a
                        dx = px - ax
                        dy = py - ay
                        d2 = dx * dx + dy * dy
                        dist_sets[a].add(d2)
        
        # Swap local search
        if time.time() < deadline:
            for _ in range(30):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    R = S[i]
                    S.pop(i)
                    del dist_sets[R]
                    in_set.remove(R)
                    for a in S:
                        ax, ay = a
                        dx = R[0] - ax
                        dy = R[1] - ay
                        d2 = dx * dx + dy * dy
                        dist_sets[a].discard(d2)
                    
                    added = 0
                    for p in candidates:
                        if p in in_set:
                            continue
                        px, py = p
                        valid = True
                        dists = set()
                        for a in S:
                            ax, ay = a
                            dx = px - ax
                            dy = py - ay
                            d2 = dx * dx + dy * dy
                            if d2 in dists or d2 in dist_sets[a]:
                                valid = False
                                break
                            dists.add(d2)
                        if valid:
                            S.append(p)
                            dist_sets[p] = dists
                            in_set.add(p)
                            for a in S[:-1]:
                                ax, ay = a
                                dx = px - ax
                                dy = py - ay
                                d2 = dx * dx + dy * dy
                                dist_sets[a].add(d2)
                            added += 1
                        if added >= 2:
                            break
                    
                    if added >= 2:
                        improved = True
                    else:
                        S.insert(i, R)
                        dists = set()
                        for a in S:
                            if a == R:
                                continue
                            ax, ay = a
                            dx = R[0] - ax
                            dy = R[1] - ay
                            d2 = dx * dx + dy * dy
                            dists.add(d2)
                            dist_sets[a].add(d2)
                        dist_sets[R] = dists
                        in_set.add(R)
                
                if not improved:
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    return best_set
