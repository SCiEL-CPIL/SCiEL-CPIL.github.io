import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    def check_four_collinear(points):
        """Check if any 4 points are collinear. Returns True if invalid."""
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) != dy * (x3 - x1):
                        continue
                    for l in range(k + 1, n):
                        x4, y4 = points[l]
                        if dx * (y4 - y1) == dy * (x4 - x1):
                            return True
        return False

    def is_valid(points):
        """Check if points are valid: 20 distinct, in bounds, no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        return not check_four_collinear(points)

    def compute_score(points):
        """
        Compute the size of the largest subset with no 3 collinear
        using branch and bound with bitmasks.
        """
        n = len(points)
        
        # Precompute: for each pair (i,j), which points k are collinear with them
        # forbidden[i] = bitmask of points that cannot be added if i is in selected
        # Actually: if we have selected set S, and we want to add i,
        # we need no j,k in S such that i,j,k collinear.
        
        # collinear_with[i] = list of (j, k) pairs where i,j,k are collinear
        # Better: conflict[i] = bitmask of all points that form a collinear triple with i and some other selected point
        # This is tricky. Let me use a different approach.
        
        # For each point i, compute the set of pairs (j,k) such that i,j,k collinear
        # When adding i, we need: for all j in selected, no k in selected (k != j) with i,j,k collinear
        
        # Precompute: lines[i] = set of j (j != i) such that there exists k with i,j,k collinear
        # Actually, let me precompute: for each i, the set of points that are "blocked" when i is added given current selected
        
        # Simpler: precompute collinear_triples
        # triple_conflict[i] = bitmask of points j such that there exists k with i,j,k collinear
        # No, that's not right either.
        
        # Let me think differently:
        # A set S is valid if no 3 points in S are collinear.
        # We build S incrementally. When adding i to S, we check:
        # For each pair (j, k) in S, i is not collinear with j and k.
        # Equivalently: for each j in S, i is not in the set of points collinear with j and any other point in S.
        
        # Precompute: for each pair (j, k), the set of points collinear with them
        # collinear[j][k] = bitmask of points m (m != j, k) such that j,k,m collinear
        
        n2 = n
        collinear = [[0] * n2 for _ in range(n2)]
        for i in range(n2):
            x1, y1 = points[i]
            for j in range(i + 1, n2):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                mask = 0
                for k in range(n2):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        mask |= (1 << k)
                collinear[i][j] = mask
                collinear[j][i] = mask
        
        best = [0]
        
        def backtrack(start, selected_mask, size):
            if size > best[0]:
                best[0] = size
            
            # Upper bound
            remaining = 0
            m = selected_mask
            while m:
                m &= m - 1
            for i in range(start, n2):
                if not (selected_mask & (1 << i)):
                    remaining += 1
            
            if size + remaining <= best[0]:
                return
            
            # Try adding each point from start onwards
            for i in range(start, n2):
                if selected_mask & (1 << i):
                    continue
                
                # Check if i can be added
                valid = True
                # For each j in selected, check if there's k in selected such that i,j,k collinear
                temp = selected_mask
                while temp:
                    lsb = temp & (-temp)
                    j = lsb.bit_length() - 1
                    temp ^= lsb
                    # Check if any k in selected is collinear with i and j
                    # collinear[i][j] gives points collinear with i and j
                    if collinear[i][j] & selected_mask:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                new_mask = selected_mask | (1 << i)
                backtrack(i + 1, new_mask, size + 1)
        
        backtrack(0, 0, 0)
        return best[0]

    def random_valid_point(existing):
        """Generate a random point that doesn't create 4 collinear with existing."""
        while time.time() < deadline - 5.0:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if any(p[0] == x and p[1] == y for p in existing):
                continue
            test = existing + [[x, y]]
            if not check_four_collinear(test):
                return [x, y]
        return None

    # Generate initial configuration
    random.seed(12345)
    pts = []
    max_tries = 10000
    tries = 0
    while len(pts) < 20 and tries < max_tries:
        tries += 1
        p = random_valid_point(pts)
        if p is not None:
            pts.append(p)
    
    if len(pts) < 20:
        # Fallback: use a structured set
        pts = []
        for i in range(20):
            x = i - 10
            y = (i * i) % 21 - 10
            if not any(p[0] == x and p[1] == y for p in pts):
                if not check_four_collinear(pts + [[x, y]]):
                    pts.append([x, y])
        # If still not enough, add more
        while len(pts) < 20:
            for x in range(-100, 101):
                for y in range(-100, 101):
                    if len(pts) >= 20:
                        break
                    if not any(p[0] == x and p[1] == y for p in pts):
                        if not check_four_collinear(pts + [[x, y]]):
                            pts.append([x, y])
                if len(pts) >= 20:
                    break
            if len(pts) >= 20:
                break
    
    if len(pts) != 20:
        # Last resort fallback
        pts = [[i, i * i % 101 - 50] for i in range(-10, 10)]
        # This might not be valid, but it's a fallback
    
    best_pts = [p[:] for p in pts]
    best_score = compute_score(best_pts)
    current_pts = [p[:] for p in pts]
    current_score = best_score
    
    # Simulated annealing
    T = 8.0
    cooling = 0.998
    
    while time.time() < deadline - 10.0:
        idx = random.randint(0, 19)
        old_p = current_pts[idx]
        
        # Generate new point
        new_p = None
        for _ in range(50):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if old_p[0] == x and old_p[1] == y:
                continue
            if any(current_pts[i][0] == x and current_pts[i][1] == y for i in range(20) if i != idx):
                continue
            test = [p[:] for p in current_pts]
            test[idx] = [x, y]
            if not check_four_collinear(test):
                new_p = [x, y]
                break
        
        if new_p is None:
            T *= cooling
            continue
        
        new_pts = [p[:] for p in current_pts]
        new_pts[idx] = new_p
        new_score = compute_score(new_pts)
        
        if new_score < current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score < best_score:
                best_pts = [p[:] for p in new_pts]
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta == 0:
                if random.random() < 0.3:
                    current_pts = new_pts
                    current_score = new_score
            else:
                if T > 0.01:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
        
        T *= cooling
    
    return best_pts
