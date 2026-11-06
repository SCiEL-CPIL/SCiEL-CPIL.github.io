import itertools
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate 5x5 grid points
    points = [(i, j) for i in range(5) for j in range(5)]
    point_indices = list(range(25))
    
    # 2. Find all lines with >= 4 points in the full grid
    # A line is defined by a set of points.
    # We iterate over all pairs of points to define a line, then collect all points on that line.
    # To avoid duplicates, we can use a canonical representation of the line (e.g., normalized slope/intercept or set of indices).
    
    lines = []
    seen_lines = set()
    
    for p1 in range(25):
        for p2 in range(p1 + 1, 25):
            x1, y1 = points[p1]
            x2, y2 = points[p2]
            
            # Calculate line equation Ax + By + C = 0
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize the line coefficients to handle duplicates
            # Find GCD of A, B, C to normalize
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(A, gcd(B, C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            # Ensure a canonical sign (e.g., A > 0, or A == 0 and B > 0)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            line_key = (A, B, C)
            
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            # Find all points on this line
            line_pts = []
            for p3 in range(25):
                x3, y3 = points[p3]
                if A * x3 + B * y3 + C == 0:
                    line_pts.append(p3)
            
            if len(line_pts) >= 4:
                lines.append(frozenset(line_pts))
                
    # 3. Find a set of 5 points to remove that hits all lines in 'lines'
    # We use backtracking to find hitting sets of size 5.
    # We want to find multiple valid sets to compare scores.
    
    valid_removals = []
    
    def backtrack(idx, current_removal, remaining_lines):
        if time.time() > deadline - 10:
            return
        
        if not remaining_lines:
            valid_removals.append(set(current_removal))
            return
            
        if len(current_removal) == 5:
            return
            
        # Pick a line from remaining_lines (heuristic: shortest line first? or just first)
        # To speed up, pick a line with minimum points not yet removed?
        # Actually, we just need to hit all.
        line = remaining_lines[0]
        rest_lines = remaining_lines[1:]
        
        # We must remove at least one point from 'line'
        # Points in 'line' that are not already removed
        candidates = [p for p in line if p not in current_removal]
        
        if not candidates:
            return # Cannot hit this line
        
        for p in candidates:
            new_removal = list(current_removal) + [p]
            # Filter remaining lines: remove any line that is hit by p
            new_remaining = [l for l in rest_lines if p not in l]
            backtrack(idx + 1, new_removal, new_remaining)
            
    backtrack(0, [], lines)
    
    # If no valid removal found (unlikely), fallback to parabola
    if not valid_removals:
        return [[x, x * x] for x in range(-10, 10)]
        
    # 4. For each valid removal, calculate the score of the remaining 20 points
    # Score: Largest subset with no 3 collinear.
    
    def calculate_score(removal_set):
        # Get the 20 points
        remaining_indices = [i for i in range(25) if i not in removal_set]
        
        # Precompute all collinear triples in these 20 points
        # A triple is collinear if (p1, p2, p3) are collinear.
        # We can check collinearity using cross product.
        
        triples = []
        for i in range(20):
            for j in range(i + 1, 20):
                for k in range(j + 1, 20):
                    p1 = points[remaining_indices[i]]
                    p2 = points[remaining_indices[j]]
                    p3 = points[remaining_indices[k]]
                    # Collinear if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
                    if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0]):
                        triples.append((i, j, k))
                        
        # Now find max subset of indices 0..19 such that no triple from 'triples' is fully included.
        # This is the maximum independent set in a 3-uniform hypergraph.
        # Branch and Bound.
        
        best_score = 0
        
        def bbb(current_set, start_idx, current_size):
            nonlocal best_score
            if time.time() > deadline - 5:
                return
                
            if current_size > best_score:
                best_score = current_size
                
            # Pruning: if current_size + (19 - start_idx + 1) <= best_score, return
            if current_size + (20 - start_idx) <= best_score:
                return
                
            if start_idx == 20:
                return
                
            # Option 1: Exclude start_idx
            bbb(current_set, start_idx + 1, current_size)
            
            # Option 2: Include start_idx
            # Check if including start_idx creates a collinear triple with any two points in current_set
            # We need to know which triples involve start_idx.
            # For efficiency, precompute for each point, the list of triples containing it.
            # Or simply check all triples in 'triples' that contain start_idx.
            
            # Let's optimize: Precompute conflict graph?
            # Actually, checking all triples is O(T) which is small.
            
            valid = True
            for t in triples:
                if start_idx in t:
                    # Check if the other two are in current_set
                    others = [x for x in t if x != start_idx]
                    if others[0] in current_set and others[1] in current_set:
                        valid = False
                        break
                        
            if valid:
                new_set = set(current_set)
                new_set.add(start_idx)
                bbb(new_set, start_idx + 1, current_size + 1)
                
        bbb(set(), 0, 0)
        return best_score
        
    best_config = None
    min_score = 21
    
    # Sort valid removals? Maybe try a few.
    # To be safe, try the first 10 valid removals found.
    for rem in valid_removals[:10]:
        if time.time() > deadline - 20:
            break
        s = calculate_score(rem)
        if s < min_score:
            min_score = s
            best_config = rem
            
    if best_config is None:
        return [[x, x * x] for x in range(-10, 10)]
        
    # Construct final points
    final_points = []
    for i in range(25):
        if i not in best_config:
            final_points.append(list(points[i]))
            
    return final_points
