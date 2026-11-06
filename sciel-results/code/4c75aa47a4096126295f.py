import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_line_id(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(points):
        lines = defaultdict(set)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                lid = get_line_id(points[i], points[j])
                lines[lid].add(i)
                lines[lid].add(j)
        for lid, pts_in_line in lines.items():
            if len(pts_in_line) >= 4:
                return True
        return False

    def max_no_3_collinear(points):
        n = len(points)
        if n <= 2:
            return n
            
        all_lines = defaultdict(int) 
        for i in range(n):
            for j in range(i + 1, n):
                lid = get_line_id(points[i], points[j])
                all_lines[lid] |= (1 << i)
                all_lines[lid] |= (1 << j)
                
        line_id_pair = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    line_id_pair[i][j] = get_line_id(points[i], points[j])
                    
        best = 0
        full_mask = (1 << n) - 1
        
        # Sort indices by descending line count for better pruning
        point_line_count = [0]*n
        for i in range(n):
            lines_seen = set()
            for j in range(n):
                if i != j:
                    lines_seen.add(line_id_pair[i][j])
            point_line_count[i] = len(lines_seen)
            
        indices = list(range(n))
        indices.sort(key=lambda x: -point_line_count[x])
        
        # Remap masks to sorted order? 
        # To keep it simple and fast, we stick to original indices but 
        # the pruning `current_len + (n - start_idx) <= best` is weak if 
        # we don't sort the indices used in the loop.
        # Let's permute the points themselves to make the B&B faster.
        
        # Create a permutation
        perm = indices
        inv_perm = [0]*n
        for i, idx in enumerate(perm):
            inv_perm[idx] = i
            
        # Remap all_lines to new indices
        new_all_lines = defaultdict(int)
        for lid, mask in all_lines.items():
            new_mask = 0
            for i in range(n):
                if mask & (1 << i):
                    new_mask |= (1 << inv_perm[i])
            new_all_lines[lid] = new_mask
            
        # Remap line_id_pair
        new_line_id_pair = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    orig_i = perm[i]
                    orig_j = perm[j]
                    new_line_id_pair[i][j] = line_id_pair[orig_i][orig_j]

        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = S_mask.bit_count()
            
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include
            i = start_idx
            valid = True
            temp_mask = S_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                lid = new_line_id_pair[i][j]
                M = new_all_lines[lid]
                if (M & S_mask).bit_count() > 1:
                    valid = False
                    break
                temp_mask ^= lsb
                
            if valid:
                backtrack(S_mask | (1 << i), start_idx + 1)
                
        backtrack(0, 0)
        return best

    def count_triples(points):
        lines = defaultdict(set)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                lid = get_line_id(points[i], points[j])
                lines[lid].add(i)
                lines[lid].add(j)
        triples = 0
        for lid, s in lines.items():
            if len(s) == 3:
                triples += 1
        return triples

    def simulated_annealing(initial_points, max_iters=2000):
        current_points = initial_points[:]
        current_triples = count_triples(current_points)
        best_points = current_points[:]
        best_triples = current_triples
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            
            # Move to a nearby random point
            new_x = old_p[0] + random.randint(-5, 5)
            new_y = old_p[1] + random.randint(-5, 5)
            
            # Keep within bounds
            new_x = max(-100, min(100, new_x))
            new_y = max(-100, min(100, new_y))
            
            new_p = (new_x, new_y)
            
            # Check collision with other points
            if any(new_p == current_points[k] for k in range(20) if k != idx):
                continue
                
            # Check 4-collinear
            temp_points = current_points[:]
            temp_points[idx] = new_p
            if has_4_collinear(temp_points):
                continue
                
            new_triples = count_triples(temp_points)
            
            delta = new_triples - current_triples
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points = temp_points
                current_triples = new_triples
                
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = current_points[:]
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_points, best_triples

    best_config = None
    best_score = 21
    best_triples_global = -1
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Perturbed 5x5 Grid
    grid_points = [(x, y) for x in range(5) for y in range(5)]
    for _ in range(5):
        pts = grid_points[:]
        random.shuffle(pts)
        # Remove 5 points randomly
        selected = pts[:20]
        # Scale to larger range to allow more movement? 
        # Or keep small to maintain structure?
        # Let's scale by 10 to allow SA to move more effectively
        scaled = [(x*10, y*10) for x, y in selected]
        initial_configs.append(scaled)
        
    # 2. Random Cluster
    for _ in range(5):
        pts = []
        for _ in range(20):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = (x, y)
            if p in pts:
                continue
            pts.append(p)
        initial_configs.append(pts)
        
    # Run SA on each
    for init_pts in initial_configs:
        if time.time() > deadline - 10:
            break
            
        # Ensure no 4-collinear in init (grid subset might have 4-collinear if not careful? 
        # 5x5 grid has rows of 5. Removing 5 points randomly might leave a row of 4.
        # So we must filter or repair.)
        if has_4_collinear(init_pts):
            # Try to repair by moving points
            repaired = init_pts[:]
            for i in range(20):
                if has_4_collinear(repaired):
                    # Move point i randomly until valid
                    for _ in range(100):
                        new_x = random.randint(-100, 100)
                        new_y = random.randint(-100, 100)
                        if (new_x, new_y) in repaired:
                            continue
                        repaired[i] = (new_x, new_y)
                        if not has_4_collinear(repaired):
                            break
            if has_4_collinear(repaired):
                continue
            init_pts = repaired

        sa_points, sa_triples = simulated_annealing(init_pts, max_iters=1000)
        
        if not has_4_collinear(sa_points):
            score = max_no_3_collinear(sa_points)
            
            if score < best_score:
                best_score = score
                best_config = [list(p) for p in sa_points]
                
            if best_score <= 10:
                break

    # Fallback if SA fails
    if best_config is None:
        # Greedy fallback
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p in pts:
                continue
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts.append(p)
        best_config = [list(p) for p in pts]

    return best_config
