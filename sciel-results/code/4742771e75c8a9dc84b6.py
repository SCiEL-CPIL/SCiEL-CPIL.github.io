import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def calculate_score(points):
        """Exact max subset with no 3 collinear using B&B with bitmasks."""
        n = len(points)
        
        # Precompute: for each point i, bitmask of points collinear with i
        # Actually, we need: for each pair (i,j), which points k are collinear with i,j
        # For B&B: when we add point x to set S, we need x not collinear with any pair in S
        
        # Better approach: precompute conflict masks
        # conflict[i] = set of j such that there exists a line through i and j with >= 3 points total
        # No, that's not quite right either.
        
        # Let's think differently. The constraint is: no 3 points in S are collinear.
        # Equivalently: for any line L, |S ∩ L| <= 2.
        
        # Precompute all lines that contain at least 3 points, with their point sets.
        lines = []  # list of frozensets of point indices
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Find all points on this line
                pts_on_line = set()
                for k in range(n):
                    if are_collinear(p1, p2, points[k]):
                        pts_on_line.add(k)
                if len(pts_on_line) >= 3:
                    line_key = frozenset(pts_on_line)
                    if line_key not in seen_lines:
                        seen_lines.add(line_key)
                        lines.append(line_key)
        
        # Now the problem is: max subset S such that for every line L in lines, |S ∩ L| <= 2.
        # This is a packing problem.
        
        # For B&B, we use:
        # - Include/exclude each point
        # - Prune if adding all remaining points can't beat best
        
        # Precompute for each point, which lines it's on
        point_lines = [set() for _ in range(n)]
        for li, line in enumerate(lines):
            for pt in line:
                point_lines[pt].add(li)
        
        best_score = 0
        current_size = 0
        # Track for each line how many points from S are on it
        line_counts = [0] * len(lines)
        
        def can_add(pt):
            """Check if adding pt to current S keeps |S ∩ L| <= 2 for all lines."""
            for li in point_lines[pt]:
                if line_counts[li] >= 2:
                    return False
            return True
        
        def backtrack(idx):
            nonlocal best_score, current_size
            
            if idx == n:
                if current_size > best_score:
                    best_score = current_size
                return
            
            # Upper bound: current_size + (n - idx)
            if current_size + (n - idx) <= best_score:
                return
            
            # Option 1: skip point idx
            backtrack(idx + 1)
            
            # Option 2: include point idx
            if can_add(idx):
                # Add idx
                for li in point_lines[idx]:
                    line_counts[li] += 1
                current_size += 1
                backtrack(idx + 1)
                current_size -= 1
                for li in point_lines[idx]:
                    line_counts[li] -= 1
        
        backtrack(0)
        return best_score

    def generate_structured_points():
        """Generate points with controlled collinearity."""
        # Use a grid-like structure but perturb to avoid 4 collinear
        # Strategy: place points on a 5x4 grid with small perturbations
        points = []
        seen = set()
        
        # Base grid positions
        base_positions = []
        for row in range(4):
            for col in range(5):
                x = -20 + col * 10 + random.randint(-2, 2)
                y = -15 + row * 10 + random.randint(-2, 2)
                if x < -100: x = -100
                if x > 100: x = 100
                if y < -100: y = -100
                if y > 100: y = 100
                if (x, y) not in seen:
                    seen.add((x, y))
                    base_positions.append([x, y])
        
        # Ensure exactly 20 unique points
        while len(base_positions) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                base_positions.append([x, y])
        
        return base_positions[:20]

    def local_search_improve(points):
        """Improve by moving points to break collinear triples."""
        best_points = [p[:] for p in points]
        best_score = calculate_score(best_points)
        
        if best_score <= 8:
            return best_points, best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(20):
                if time.time() > deadline:
                    return best_points, best_score
                old_point = best_points[i][:]
                # Try targeted moves: move to a position that breaks existing collinear triples
                # First, find which triples point i is part of
                triples_with_i = []
                for j in range(20):
                    if j == i:
                        continue
                    for k in range(j + 1, 20):
                        if k == i:
                            continue
                        if are_collinear(best_points[i], best_points[j], best_points[k]):
                            triples_with_i.append((j, k))
                
                # Try random moves and targeted moves
                candidates = []
                # Random candidates
                for _ in range(15):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    candidates.append([x, y])
                
                # Targeted: try to move perpendicular to existing lines through i
                for (j, k) in triples_with_i[:5]:
                    # Move point i away from line jk
                    pj, pk = best_points[j], best_points[k]
                    dx = pk[0] - pj[0]
                    dy = pk[1] - pj[1]
                    # Perpendicular direction
                    px = -dy
                    py = dx
                    # Normalize and scale
                    norm = (px * px + py * py) ** 0.5
                    if norm > 0:
                        px = int(px / norm * random.randint(5, 20))
                        py = int(py / norm * random.randint(5, 20))
                        new_x = best_points[i][0] + px
                        new_y = best_points[i][1] + py
                        if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                            candidates.append([new_x, new_y])
                
                for new_point in candidates:
                    if time.time() > deadline:
                        return best_points, best_score
                    # Check distinctness
                    if any(new_point == p for idx, p in enumerate(best_points) if idx != i):
                        continue
                    test_points = best_points[:]
                    test_points[i] = new_point
                    if has_four_collinear(test_points):
                        continue
                    score = calculate_score(test_points)
                    if score < best_score:
                        best_score = score
                        best_points = test_points
                        improved = True
                        break
                    elif score == best_score and random.random() < 0.1:
                        # Occasional sideways move to escape local minima
                        best_points = test_points
                        improved = True
                        break
        
        return best_points, best_score

    # Main search loop
    best_points = None
    best_score = 21
    
    # Phase 1: Structured random restarts
    for restart in range(80):
        if time.time() > deadline:
            break
        
        if restart % 2 == 0:
            points = generate_structured_points()
        else:
            points = []
            seen = set()
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    seen.add((x, y))
                    points.append([x, y])
        
        if has_four_collinear(points):
            continue
        
        score = calculate_score(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
        
        # Quick local search on promising starts
        if score <= 12 and time.time() < deadline:
            improved_points, improved_score = local_search_improve(points)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Phase 2: More intensive local search on best found
    if best_points and time.time() < deadline - 20:
        best_points, best_score = local_search_improve(best_points)
    
    # Fallback
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
