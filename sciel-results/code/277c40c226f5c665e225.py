import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 30
    RANGE = 30

    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    inits.append(gen_grid(15, 2, -6, 0))
    inits.append(gen_grid(2, 15, 0, -6))
    
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_overall_score = 0
    best_points = inits[0]

    def get_score(points):
        counts = Counter()
        for i in range(N-2):
            x1, y1 = points[i]
            for j in range(i+1, N-1):
                x2, y2 = points[j]
                for k in range(j+1, N):
                    x3, y3 = points[k]
                    d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                    if d > 0:
                        counts[d] += 1
        return max(counts.values()) if counts else 0

    for init_pts in inits:
        if len(set(init_pts)) != N:
            continue
        for p in init_pts:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                break
        else:
            s = get_score(init_pts)
            if s > best_overall_score:
                best_overall_score = s
                best_points = list(init_pts)

    K = 5
    stuck_limit = 5000
    
    for init_pts in inits:
        if time.time() > deadline - 5.0:
            break
            
        current_points = list(init_pts)
        if len(set(current_points)) != N:
            continue
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                break
        else:
            areas = Counter()
            for i in range(N-2):
                x1, y1 = current_points[i]
                for j in range(i+1, N-1):
                    x2, y2 = current_points[j]
                    for k in range(j+1, N):
                        x3, y3 = current_points[k]
                        d = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                        if d > 0:
                            areas[d] += 1
            
            curr_max = max(areas.values()) if areas else 0
            
            if curr_max > best_overall_score:
                best_overall_score = curr_max
                best_points = list(current_points)
            
            no_improve = 0
            
            while time.time() < deadline - 2.0 and no_improve < stuck_limit:
                idx = random.randint(0, N-1)
                old_pt = current_points[idx]
                
                candidates = []
                for _ in range(K * 2):
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    if dx == 0 and dy == 0: continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                        if new_pt not in current_points:
                            candidates.append(new_pt)
                            if len(candidates) >= K:
                                break
                while len(candidates) < K:
                    new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                    if new_pt not in current_points and new_pt != old_pt:
                        candidates.append(new_pt)
                
                best_candidate_score = curr_max - 1
                best_candidate_pt = None
                
                for new_pt in candidates:
                    # Apply move
                    current_points[idx] = new_pt
                    
                    for j in range(N):
                        if j == idx: continue
                        x2, y2 = current_points[j]
                        for k in range(j+1, N):
                            if k == idx: continue
                            x3, y3 = current_points[k]
                            d_old = abs((x2-old_pt[0])*(x3[1]-old_pt[1]) - (x3[0]-old_pt[0])*(x2[1]-old_pt[1]))
                            d_new = abs((x2-new_pt[0])*(x3[1]-new_pt[1]) - (x3[0]-new_pt[0])*(x2[1]-new_pt[1]))
                            
                            if d_old > 0:
                                areas[d_old] -= 1
                                if areas[d_old] == 0:
                                    del areas[d_old]
                            if d_new > 0:
                                areas[d_new] += 1
                    
                    new_max = max(areas.values()) if areas else 0
                    
                    # Revert move
                    current_points[idx] = old_pt
                    for j in range(N):
                        if j == idx: continue
                        x2, y2 = current_points[j]
                        for k in range(j+1, N):
                            if k == idx: continue
                            x3, y3 = current_points[k]
                            d_old = abs((x2-old_pt[0])*(x3[1]-old_pt[1]) - (x3[0]-old_pt[0])*(x2[1]-old_pt[1]))
                            d_new = abs((x2-new_pt[0])*(x3[1]-new_pt[1]) - (x3[0]-new_pt[0])*(x2[1]-new_pt[1]))
                            
                            if d_old > 0:
                                areas[d_old] += 1
                            if d_new > 0:
                                areas[d_new] -= 1
                                if areas[d_new] == 0:
                                    del areas[d_new]
                    
                    if new_max > best_candidate_score:
                        best_candidate_score = new_max
                        best_candidate_pt = new_pt

                if best_candidate_pt is not None:
                    current_points[idx] = best_candidate_pt
                    for j in range(N):
                        if j == idx: continue
                        x2, y2 = current_points[j]
                        for k in range(j+1, N):
                            if k == idx: continue
                            x3, y3 = current_points[k]
                            d_old = abs((x2-old_pt[0])*(x3[1]-old_pt[1]) - (x3[0]-old_pt[0])*(x2[1]-old_pt[1]))
                            d_new = abs((x2-best_candidate_pt[0])*(x3[1]-best_candidate_pt[1]) - (x3[0]-best_candidate_pt[0])*(x2[1]-best_candidate_pt[1]))
                            
                            if d_old > 0:
                                areas[d_old] -= 1
                                if areas[d_old] == 0:
                                    del areas[d_old]
                            if d_new > 0:
                                areas[d_new] += 1
                    
                    curr_max = best_candidate_score
                    
                    if curr_max > best_overall_score:
                        best_overall_score = curr_max
                        best_points = list(current_points)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

    if len(best_points) != N or len(set(best_points)) != N:
         return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
