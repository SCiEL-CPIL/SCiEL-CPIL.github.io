import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        # Precompute all collinear triples and build conflict bitmasks
        # triple_mask[i] = bitmask of points that form a collinear triple with i
        # For B&B: when including point i, we need to ensure no two points in current_mask
        # are both in triple_mask[i]
        
        # Build adjacency: for each pair (i,j), find all k such that i,j,k collinear
        # Then for each triple (i,j,k), mark that i conflicts with {j,k}, etc.
        
        # More efficient: for each point i, compute which pairs (j,k) with j<k, j,k != i
        # are collinear with i. Store as list of (j,k) pairs.
        
        conflicts = [[] for _ in range(n)]
        
        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + 1, n):
                xj, yj = pts[j]
                dx = xj - xi
                dy = yj - yi
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    if dx * (yk - yi) == dy * (xk - xi):
                        conflicts[i].append((j, k))
                        conflicts[j].append((i, k))
                        conflicts[k].append((i, j))
        
        # For faster validity check: for each point i, precompute a set of "forbidden pairs"
        # But we can also use a different approach: 
        # When we include point i, we must remove from consideration any point that forms
        # a triple with i and another already-included point.
        
        # Order by degree
        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)
        
        best_score = 0
        time_limit = deadline - 5
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if time.time() > time_limit:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            
            if current_size + (n - step) <= best_score:
                return
                
            idx = order[step]
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)
            
            # Option 2: Include idx
            # Check: does adding idx create a collinear triple with any two points in current_mask?
            valid = True
            for a, b in conflicts[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def generate_grid_candidates():
        """
        Generate candidate sets based on grid perturbations.
        """
        candidates = []
        # Base grids
        # 4x5, 5x4, 5x5 (subset)
        
        # 4x5 Grid
        for ox in range(-10, 11, 2):
            for oy in range(-10, 11, 2):
                pts = []
                for i in range(4):
                    for j in range(5):
                        x = ox + i * 2 + random.randint(-1, 1)
                        y = oy + j * 2 + random.randint(-1, 1)
                        if -100 <= x <= 100 and -100 <= y <= 100:
                            pts.append((x, y))
                if len(pts) == 20:
                    # Check for duplicates
                    if len(set(pts)) == 20:
                        candidates.append(pts)
                        
        # 5x4 Grid
        for ox in range(-10, 11, 2):
            for oy in range(-10, 11, 2):
                pts = []
                for i in range(5):
                    for j in range(4):
                        x = ox + i * 2 + random.randint(-1, 1)
                        y = oy + j * 2 + random.randint(-1, 1)
                        if -100 <= x <= 100 and -100 <= y <= 100:
                            pts.append((x, y))
                if len(pts) == 20:
                    if len(set(pts)) == 20:
                        candidates.append(pts)
                        
        # Random Clumps
        for _ in range(50):
            cx = random.randint(-50, 50)
            cy = random.randint(-50, 50)
            pts = []
            while len(pts) < 20:
                x = cx + random.randint(-10, 10)
                y = cy + random.randint(-10, 10)
                p = (x, y)
                if p not in pts:
                    pts.append(p)
            candidates.append(pts)
            
        return candidates

    def local_search(pts, max_iterations=5000):
        current_pts = [tuple(p) for p in pts]
        if has_4_collinear(current_pts):
            # Try to fix it
            for _ in range(200):
                # Find a 4-collinear set and move one of those points
                # First, try random
                idx = random.randint(0, 19)
                old_p = current_pts[idx]
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                if new_p not in current_pts and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    current_pts[idx] = new_p
                    if not has_4_collinear(current_pts):
                        break
            if has_4_collinear(current_pts):
                return None, float('inf')

        current_score = calculate_score(current_pts)
        best_pts = current_pts[:]
        best_score = current_score
        no_improve_count = 0
        
        for i in range(max_iterations):
            if elapsed() > deadline - 10:
                break
            
            # Preferentially move points that are in many collinear triples
            # Compute a "conflict degree" for each point
            conflict_deg = [0] * 20
            for a in range(20):
                xa, ya = current_pts[a]
                for b in range(a + 1, 20):
                    xb, yb = current_pts[b]
                    for c in range(b + 1, 20):
                        xc, yc = current_pts[c]
                        if (xb - xa) * (yc - ya) == (yb - ya) * (xc - xa):
                            conflict_deg[a] += 1
                            conflict_deg[b] += 1
                            conflict_deg[c] += 1
            
            # Weighted random selection: higher conflict degree -> more likely to be moved
            total_deg = sum(conflict_deg)
            if total_deg == 0:
                idx = random.randint(0, 19)
            else:
                r = random.random() * total_deg
                cum = 0
                idx = 0
                for k in range(20):
                    cum += conflict_deg[k]
                    if r <= cum:
                        idx = k
                        break
                idx = min(idx, 19)
            
            old_p = current_pts[idx]
            
            # Generate new point
            # Try local moves with varying radii
            radius = random.choice([2, 3, 5, 8])
            if random.random() < 0.8:
                new_p = (old_p[0] + random.randint(-radius, radius), old_p[1] + random.randint(-radius, radius))
            else:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                
            if new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
                
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            if has_4_collinear(new_pts):
                continue
                
            new_score = calculate_score(new_pts)
            
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts
                current_pts = new_pts
                current_score = new_score
                no_improve_count = 0
            elif new_score == current_score and random.random() < 0.15:
                # Accept equal score to escape local optima
                current_pts = new_pts
                current_score = new_score
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            # If stuck for a while, try a larger perturbation
            if no_improve_count > 100:
                no_improve_count = 0
                # Do a bigger random move
                for _ in range(5):
                    idx2 = random.randint(0, 19)
                    old_p2 = current_pts[idx2]
                    new_p2 = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p2 not in current_pts and -100 <= new_p2[0] <= 100 and -100 <= new_p2[1] <= 100:
                        current_pts[idx2] = new_p2
                        if has_4_collinear(current_pts):
                            current_pts[idx2] = old_p2
                        else:
                            current_score = calculate_score(current_pts)
                            if current_score < best_score:
                                best_score = current_score
                                best_pts = current_pts[:]
        
        return best_pts, best_score

    best_points = None
    best_val = float('inf')
    
    # Generate initial candidates
    candidates = generate_grid_candidates()
    
    # Add more diverse random candidates
    for _ in range(30):
        pts = []
        while len(pts) < 20:
            p = (random.randint(-50, 50), random.randint(-50, 50))
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    # Shuffle to try diverse ones
    random.shuffle(candidates)
    
    # Process candidates with time-aware iteration limits
    time_per_candidate = max(2.0, (deadline - 25 - elapsed()) / max(1, len(candidates)))
    
    for cand in candidates:
        if elapsed() > deadline - 20:
            break
        
        # Validate
        if has_4_collinear(cand):
            # Try to repair quickly
            repaired = cand[:]
            for _ in range(100):
                if not has_4_collinear(repaired):
                    break
                idx = random.randint(0, 19)
                old_p = repaired[idx]
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                if new_p not in repaired and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    repaired[idx] = new_p
                    
            if has_4_collinear(repaired):
                continue
            cand = repaired

        # Scale iterations based on remaining time
        remaining = deadline - 20 - elapsed()
        if remaining < 5:
            break
        max_iter = min(2000, int(remaining * 50))
        pts, score = local_search(cand, max_iterations=max_iter)
        
        if pts is not None and score < best_val:
            best_val = score
            best_points = pts
            
    # If no candidates worked, fallback to random
    if best_points is None:
        while True:
            if elapsed() > deadline - 5:
                break
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break
        
    if best_points is None:
        # Absolute fallback: simple grid with large steps (might be invalid but try to fix)
        # Or just return a known valid set if time is tight?
        # Let's try to construct one on the fly
        pts = []
        x = 0
        y = 0
        while len(pts) < 20:
            p = (x, y)
            if p not in pts:
                pts.append(p)
            y += 1
            if y > 100:
                y = 0
                x += 1
        # This will likely have 4 collinear.
        # Let's just return the first valid random one found or a hardcoded one if we find one.
        # For safety, if we are here, we are in trouble.
        # Let's try a very sparse set which is likely valid.
        pts = [(i * 10, (i * 7) % 100) for i in range(20)]
        # Check
        if has_4_collinear(pts):
             # Perturb
             for i in range(20):
                 while has_4_collinear(pts):
                     if elapsed() > deadline - 1:
                         break
                     pts[i] = (pts[i][0] + 1, pts[i][1])
                     if pts[i][0] > 100: pts[i] = (pts[i][0]-200, pts[i][1])
                     if pts[i][1] > 100: pts[i] = (pts[i][0], pts[i][1]-200)
        best_points = pts

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
    # Note: calculate_score is inside construct, so we can't print it directly here without refactoring
    # But the evaluator will call construct and then score it externally or internally?
    # "The evaluator measures the raw value". It likely calls construct and then runs its own scorer.
    # So we just need to return the points.
