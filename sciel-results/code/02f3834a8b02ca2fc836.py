import time
import random
import math

def construct():
    deadline = time.time() + 140.0  # Safety margin
    n = 20
    
    def is_valid_full(points):
        if len(points) != 20:
            return False
        if len(set(map(tuple, points))) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        # Check no 4 collinear
        for i in range(20):
            x1, y1 = points[i]
            for j in range(i+1, 20):
                x2, y2 = points[j]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for k in range(j+1, 20):
                    x3, y3 = points[k]
                    if dx12 * (y3 - y1) != dy12 * (x3 - x1):
                        continue
                    for l in range(k+1, 20):
                        x4, y4 = points[l]
                        if dx12 * (y4 - y1) == dy12 * (x4 - x1):
                            return False
        return True

    def is_valid_move(points, idx, new_pt):
        # points: list of 20 points
        # idx: index of point to replace
        # new_pt: [x, y]
        x, y = new_pt
        if x < -100 or x > 100 or y < -100 or y > 100:
            return False
        
        # Check distinctness
        for i in range(20):
            if i == idx:
                continue
            if points[i][0] == x and points[i][1] == y:
                return False
        
        # Check no 4 collinear involving new_pt
        # We need to check if there are 3 other points collinear with new_pt
        # such that all 4 are in the set.
        # Since the original set had no 4 collinear, any new 4-collinear set
        # must include new_pt.
        # So we check all triples from the other 19 points.
        # If new_pt, p_a, p_b, p_c are collinear, return False.
        
        other_pts = [points[i] for i in range(20) if i != idx]
        
        for i in range(19):
            x1, y1 = other_pts[i]
            dx1 = x1 - x
            dy1 = y1 - y
            for j in range(i+1, 19):
                x2, y2 = other_pts[j]
                dx2 = x2 - x
                dy2 = y2 - y
                # Check if (x1,y1), (x2,y2) are collinear with (x,y)
                # Cross product: dx1*dy2 - dy1*dx2 == 0
                if dx1 * dy2 != dy1 * dx2:
                    continue
                # They are collinear with new_pt.
                # Now check if any k > j is also collinear.
                for k in range(j+1, 19):
                    x3, y3 = other_pts[k]
                    dx3 = x3 - x
                    dy3 = y3 - y
                    if dx1 * dy3 == dy1 * dx3:
                        return False
        return True

    def get_score(points):
        """
        Calculate the size of the largest subset with no 3 collinear.
        Uses Branch and Bound.
        """
        n = len(points)
        # Precompute collinear triples
        # collinear[i] is a set of indices j such that i, j, k are collinear for some k?
        # Better: For each pair (i, j), store the set of k collinear with i, j.
        # If we select i and j, we cannot select any k in collinear[i][j].
        
        collinear = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        collinear[i][j].add(k)
                        collinear[j][i].add(k)
        
        best = [0]
        
        def backtrack(start, selected, size):
            if size > best[0]:
                best[0] = size
            
            # Upper bound
            if size + (n - start) <= best[0]:
                return
            
            # Heuristic: Try to pick points that block the most future picks?
            # Or just standard B&B.
            
            # Optimization: If we pick i, we must exclude all j in selected that are collinear with i?
            # No, selected is the set of points already picked.
            # If we pick i, we must ensure i is not collinear with any pair in selected?
            # No, the condition is "no 3 collinear".
            # So if we pick i, and we already have j, k in selected, then i, j, k must not be collinear.
            
            # Standard check:
            # If we add i to selected, for every j in selected, i must not be in collinear[j][i]?
            # Actually, if i and j are in selected, and there exists k in selected such that i,j,k collinear, invalid.
            # So when adding i, we check if for any j in selected, i is in collinear[j][i] AND there is another k in selected?
            # No.
            # Condition: No 3 points in `selected + [i]` are collinear.
            # This is equivalent to: For all pairs (a, b) in selected, i is not collinear with a and b.
            # AND for all a in selected, there is no b, c in selected such that a, b, c collinear (already guaranteed by previous steps).
            # AND for all pairs (a, i), there is no b in selected such that a, i, b collinear.
            
            # So, when adding i:
            # 1. For each j in selected, if i is in collinear[j][i], then we need to check if there is any k in selected (k != j) such that i, j, k are collinear?
            #    Wait, collinear[j][i] contains all k collinear with j and i.
            #    So if i is in collinear[j][i], it means there EXISTS some k.
            #    But we need to know if that k is in `selected`.
            #    So: For each j in selected, if collinear[j][i] intersects selected (excluding i, which is not in selected yet), then invalid.
            
            valid = True
            for j in selected:
                # Check if any point in selected (other than j) is collinear with i and j
                # collinear[i][j] is the set of points collinear with i and j.
                # If any k in selected is in collinear[i][j], then i, j, k are collinear.
                # Since i is not in selected, we just check intersection of selected and collinear[i][j].
                if selected and collinear[i][j] & set(selected):
                    valid = False
                    break
            
            if not valid:
                return
            
            # Also, we must ensure that adding i doesn't create a triple with two NEW points?
            # No, we only add one point at a time.
            
            # Wait, the check above is:
            # For each j in selected, is there a k in selected such that i, j, k are collinear?
            # This covers all triples involving i and 2 points from selected.
            # Triples involving i and 1 point from selected? No, that's not a triple.
            # Triples involving i and 0 points from selected? No.
            
            # So the check is correct.
            
            selected.append(i)
            backtrack(i+1, selected, size+1)
            selected.pop()
            
            # Optional: Branching heuristic
            # We iterate i from start to n.
            # We could reorder i to try "critical" points first?
            # For now, keep it simple.
            
            for i in range(start, n):
                # If i is already in selected? No, we only pick from start onwards.
                # But we need to check if i can be added.
                
                # Check validity of adding i
                can_add = True
                for j in selected:
                    if collinear[i][j] & set(selected):
                        can_add = False
                        break
                
                if can_add:
                    selected.append(i)
                    backtrack(i+1, selected, size+1)
                    selected.pop()
        
        backtrack(0, [], 0)
        return best[0]

    # Initialize
    random.seed(42)
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Generate initial valid config
    pts = []
    while len(pts) < 20:
        p = random_point()
        # Check if adding p to pts keeps it valid (no 4 collinear, distinct)
        # For initialization, just check full validity of pts + [p]
        test_pts = pts + [p]
        if len(set(map(tuple, test_pts))) == len(test_pts):
            # Check no 4 collinear in test_pts
            valid = True
            for i in range(len(test_pts)):
                x1, y1 = test_pts[i]
                for j in range(i+1, len(test_pts)):
                    x2, y2 = test_pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    for k in range(j+1, len(test_pts)):
                        x3, y3 = test_pts[k]
                        if dx * (y3 - y1) == dy * (x3 - x1):
                            for l in range(k+1, len(test_pts)):
                                x4, y4 = test_pts[l]
                                if dx * (y4 - y1) == dy * (x4 - x1):
                                    valid = False
                                    break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if valid:
                pts.append(p)
                
    if len(pts) != 20:
        # Fallback
        pts = [[x, x*x] for x in range(-10, 10)]
        
    best_pts = [p[:] for p in pts]
    best_score = get_score(best_pts)
    current_pts = [p[:] for p in pts]
    current_score = best_score
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        idx = random.randint(0, 19)
        new_p = random_point()
        
        if not is_valid_move(current_pts, idx, new_p):
            continue
            
        # Calculate new score
        # To speed up, we could copy and modify, then calc score
        new_pts = [p[:] for p in current_pts]
        new_pts[idx] = new_p
        new_score = get_score(new_pts)
        
        if new_score < current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score < best_score:
                best_pts = [p[:] for p in new_pts]
                best_score = new_score
        else:
            # Accept with probability
            delta = new_score - current_score
            if delta == 0:
                # Accept to explore
                if random.random() < 0.5:
                    current_pts = new_pts
                    current_score = new_score
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
                    
        T *= cooling
        
    return best_pts
