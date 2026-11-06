import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Initial configuration: 6x5 grid centered in [-30, 30]
    x_coords = [-10, -6, -2, 2, 6, 10]
    y_coords = [-8, -4, 0, 4, 8]
    pts = [(x, y) for x in x_coords for y in y_coords]
    
    def get_det(p1, p2, p3):
        return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
    
    def compute_full_score(points):
        areas = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d = get_det(points[i], points[j], points[k])
                    if d > 0:
                        areas.append(d)
        if not areas:
            return 0
        return max(Counter(areas).values())
    
    current_score = compute_full_score(pts)
    
    improved = True
    passes = 0
    max_passes = 100
    
    while improved and passes < max_passes and time.time() < deadline - 5:
        improved = False
        passes += 1
        
        for i in range(30):
            if time.time() > deadline - 5:
                break
                
            old_pt = pts[i]
            
            # Compute base counter: areas of all triples NOT involving i
            base_counter = Counter()
            indices = [j for j in range(30) if j != i]
            for idx_a in range(len(indices)):
                a = indices[idx_a]
                for idx_b in range(idx_a + 1, len(indices)):
                    b = indices[idx_b]
                    for idx_c in range(idx_b + 1, len(indices)):
                        c = indices[idx_c]
                        d = get_det(pts[a], pts[b], pts[c])
                        if d > 0:
                            base_counter[d] += 1
            
            best_new_pt = None
            best_new_score = current_score
            
            # Try candidate positions
            for x in range(-30, 31):
                if time.time() > deadline - 2:
                    break
                for y in range(-30, 31):
                    if (x, y) == old_pt:
                        continue
                    # Check if (x,y) is already in pts
                    is_dup = False
                    for p in pts:
                        if p == (x, y):
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    # Calculate areas involving i at (x,y)
                    new_areas = []
                    for j in range(30):
                        if j == i: continue
                        for k in range(j+1, 30):
                            if k == i: continue
                            d = get_det((x, y), pts[j], pts[k])
                            if d > 0:
                                new_areas.append(d)
                    
                    # Merge
                    temp_counter = base_counter.copy()
                    for d in new_areas:
                        temp_counter[d] += 1
                    
                    s = max(temp_counter.values()) if temp_counter else 0
                    
                    if s > best_new_score:
                        best_new_score = s
                        best_new_pt = (x, y)
            
            if best_new_pt is not None:
                pts[i] = best_new_pt
                current_score = best_new_score
                improved = True
    
    return pts
