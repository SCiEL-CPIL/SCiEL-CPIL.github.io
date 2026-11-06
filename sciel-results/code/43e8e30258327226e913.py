import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    def generate_pool():
        pts = set()
        for x in range(-20, 21):
            for y in range(-20, 21):
                pts.add((x, y))
        for _ in range(200):
            pts.add((random.randint(-100, 100), random.randint(-100, 100)))
        return list(pts)
    
    pool = generate_pool()
    
    def collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])
    
    def check_no_four(pts):
        lines = {}
        for p1, p2 in itertools.combinations(pts, 2):
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = p2[0]*p1[1] - p1[0]*p2[1]
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            if A == 0 and B == 0:
                continue
            g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g; B //= g; C //= g
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(p1)
            lines[key].add(p2)
        
        for pts_on_line in lines.values():
            if len(pts_on_line) >= 4:
                return False
        return True

    def count_triples(pts):
        pts_list = list(pts)
        count = 0
        for p1, p2, p3 in itertools.combinations(pts_list, 3):
            if collinear(p1, p2, p3):
                count += 1
        return count

    def calculate_score(pts):
        n = len(pts)
        pts_list = list(pts)
        idx_map = {p: i for i, p in enumerate(pts_list)}
        
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(pts_list[i], pts_list[j], pts_list[k]):
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        # Min Hitting Set B&B
        # Greedy Upper Bound
        active_triples = set(triples)
        greedy_size = 0
        temp_active = set(active_triples)
        while temp_active:
            # Count occurrences
            counts = {}
            for t in temp_active:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
            # Pick max
            v_max = max(counts, key=counts.get)
            greedy_size += 1
            # Remove triples containing v_max
            temp_active = {t for t in temp_active if v_max not in t}
            
        best_k = greedy_size
        
        def search(active_triples, current_size):
            nonlocal best_k
            if current_size >= best_k:
                return
            if not active_triples:
                if current_size < best_k:
                    best_k = current_size
                return
                
            # Pick a triple
            t = next(iter(active_triples))
            v1, v2, v3 = t
            
            # Option 1: Include v1
            new_active1 = {tt for tt in active_triples if v1 not in tt}
            search(new_active1, current_size + 1)
            
            # Option 2: Exclude v1 -> Include v2 or v3
            # Sub-option 2a: Include v2
            new_active2 = {tt for tt in active_triples if v2 not in tt}
            search(new_active2, current_size + 1)
            
            # Sub-option 2b: Exclude v2 -> Include v3
            new_active3 = {tt for tt in active_triples if v3 not in tt}
            search(new_active3, current_size + 1)

        search(active_triples, 0)
        return n - best_k

    # Greedy Construction
    current_pts = set()
    random.shuffle(pool)
    
    for p in pool:
        if len(current_pts) >= 20:
            break
        new_pts = current_pts | {p}
        if check_no_four(new_pts):
            current_pts.add(p)
            
    while len(current_pts) < 20:
        p = random.choice(pool)
        if p not in current_pts:
             new_pts = current_pts | {p}
             if check_no_four(new_pts):
                 current_pts.add(p)
                 
    best_score = calculate_score(current_pts)
    best_pts = set(current_pts)
    
    if time.time() - start_time > deadline - 30:
        return [[x, y] for x, y in best_pts]
        
    steps = 0
    max_steps = 1000
    while time.time() - start_time < deadline - 10 and steps < max_steps:
        steps += 1
        improved = False
        pts_list = list(best_pts)
        random.shuffle(pts_list)
        
        for p_old in pts_list:
            if time.time() - start_time > deadline - 10:
                break
            rest = best_pts - {p_old}
            
            best_candidate = None
            best_candidate_triples = -1
            for _ in range(50):
                p_new = random.choice(pool)
                if p_new in rest:
                    continue
                new_set = rest | {p_new}
                if not check_no_four(new_set):
                    continue
                t_count = count_triples(new_set)
                if t_count > best_candidate_triples:
                    best_candidate_triples = t_count
                    best_candidate = p_new
                    
            if best_candidate:
                new_set = rest | {best_candidate}
                s = calculate_score(new_set)
                if s < best_score:
                    best_score = s
                    best_pts = new_set
                    improved = True
                    break
                elif s == best_score and random.random() < 0.1:
                    best_pts = new_set
                    improved = True
                    break
                    
        if not improved:
            break
            
    return [[x, y] for x, y in best_pts]
# EVOLVE-BLOCK-END
