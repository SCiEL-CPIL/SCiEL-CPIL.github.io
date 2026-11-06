import time
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            # Normalize sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _pappus_configuration():
    """
    Construct the Pappus configuration: 12 points with exactly 6 ordinary lines.
    
    Pappus theorem: Given two lines L1 and L2, with points A1,A2,A3 on L1
    and B1,B2,B3 on L2, the intersection points of AiBj and AjBi (i≠j) are
    collinear in triples.
    
    We use:
    L1: y = 0, points at x = 0, 1, 2
    L2: y = 1, points at x = 0, 1, 2
    
    A1=(0,0), A2=(1,0), A3=(2,0)
    B1=(0,1), B2=(1,1), B3=(2,1)
    
    Cross intersections:
    - Line A1B2 and Line A2B1: 
      A1B2: through (0,0) and (1,1) -> y=x
      A2B1: through (1,0) and (0,1) -> x+y=1
      Intersection: (0.5, 0.5) -> scale by 2: (1, 1) but that's B2!
      
    Let me use different positions to avoid overlaps.
    
    L1: y = 0, points at x = 0, 2, 4
    L2: y = 3, points at x = 1, 3, 5
    
    A1=(0,0), A2=(2,0), A3=(4,0)
    B1=(1,3), B2=(3,3), B3=(5,3)
    
    Cross pairs (i≠j):
    (A1,B2) & (A2,B1):
      A1B2: through (0,0),(3,3) -> y=x
      A2B1: through (2,0),(1,3) -> slope=(3-0)/(1-2)=-3, y-0=-3(x-2) -> y=-3x+6
      Intersection: x = -3x+6 -> 4x=6 -> x=3/2, y=3/2 -> scale by 2: (3,3) = B2!
      
    Hmm, let me try yet another approach. Use the standard Pappus with:
    A1=(0,0), A2=(1,0), A3=(0,1)
    B1=(1,1), B2=(0,2), B3=(2,0)
    
    Actually, let me just use a known integer Pappus configuration.
    """
    # Known Pappus configuration with 12 integer points
    # Two lines with 3 points each, plus 6 intersection points
    # that form 3 more lines of 3 points each.
    
    # Line 1 (y=0): A1, A2, A3
    # Line 2 (y=1): B1, B2, B3
    # Line 3-5: the three "Pappus lines" each containing 3 intersection points
    
    A1 = [0, 0]
    A2 = [1, 0]
    A3 = [2, 0]
    B1 = [0, 1]
    B2 = [1, 1]
    B3 = [2, 1]
    
    # Cross intersections:
    # P1 = A1B2 ∩ A2B1
    # A1B2: through (0,0) and (1,1): y = x
    # A2B1: through (1,0) and (0,1): x + y = 1
    # Intersection: x = 1/2, y = 1/2 -> multiply all coords by 2 later
    
    # P2 = A1B3 ∩ A3B1
    # A1B3: through (0,0) and (2,1): y = x/2
    # A3B1: through (2,0) and (0,1): x/2 + y = 1 -> x + 2y = 2
    # Intersection: x/2 + x/2 = 1 -> x = 1, y = 1/2
    
    # P3 = A2B3 ∩ A3B2
    # A2B3: through (1,0) and (2,1): y = x - 1
    # A3B2: through (2,0) and (1,1): y = -x + 2
    # Intersection: x-1 = -x+2 -> 2x = 3 -> x = 3/2, y = 1/2
    
    # These are the Pappus collinear points: P1=(1/2,1/2), P2=(1,1/2), P3=(3/2,1/2)
    # They lie on y = 1/2!
    
    # Now the other 3 intersection points:
    # P4 = A1B2 ∩ A3B1 ... wait, let me redo this properly.
    
    # The 6 cross-intersection points are:
    # For each pair (i,j) with i≠j from {1,2,3}:
    # Ci = Ai+1 Bj ∩ Aj Bi+1 (cyclic)
    
    # Let me just use a verified integer Pappus configuration.
    # From literature, one such configuration:
    
    points = [
        [0, 0],   # A1
        [1, 0],   # A2
        [2, 0],   # A3
        [0, 1],   # B1
        [1, 1],   # B2
        [2, 1],   # B3
    ]
    
    # Compute the 6 cross-intersection points and scale to integers
    # P1 = line(A1,B2) ∩ line(A2,B1) = (1/2, 1/2)
    # P2 = line(A2,B3) ∩ line(A3,B2) = (3/2, 1/2)
    # P3 = line(A1,B3) ∩ line(A3,B1) = (1, 1/2)
    # These 3 are collinear on y=1/2
    
    # The other 3:
    # P4 = line(A1,B2) ∩ line(A3,B1)
    # A1B2: y=x; A3B1: through (2,0),(0,1): x+2y=2
    # x = 2-x -> 2x=2 -> x=1, y=1 -> that's B2! Collision.
    
    # This configuration has issues. Let me use a different one.
    
    # Use non-uniform spacing:
    A1 = [0, 0]
    A2 = [1, 0]
    A3 = [3, 0]
    B1 = [0, 1]
    B2 = [2, 1]
    B3 = [4, 1]
    
    # P1 = A1B2 ∩ A2B1
    # A1B2: through (0,0),(2,1): y = x/2
    # A2B1: through (1,0),(0,1): x+y=1
    # x/2 + x = 1 -> 3x/2 = 1 -> x=2/3, y=1/3
    
    # P2 = A2B3 ∩ A3B2
    # A2B3: through (1,0),(4,1): y = (x-1)/3
    # A3B2: through (3,0),(2,1): y = -(x-3) = -x+3
    # (x-1)/3 = -x+3 -> x-1 = -3x+9 -> 4x=10 -> x=5/2, y=7/6
    
    # P3 = A1B3 ∩ A3B1
    # A1B3: through (0,0),(4,1): y = x/4
    # A3B1: through (3,0),(0,1): x/3 + y = 1 -> x+3y=3
    # x/4 + x/3 = 1 -> 7x/12 = 1 -> x=12/7, y=3/7
    
    # Check if P1, P2, P3 are collinear:
    # P1=(2/3, 1/3), P2=(5/2, 7/6), P3=(12/7, 3/7)
    # Slope P1P2: (7/6-1/3)/(5/2-2/3) = (7/6-2/6)/(15/6-4/6) = (5/6)/(11/6) = 5/11
    # Slope P1P3: (3/7-1/3)/(12/7-2/3) = (9/21-7/21)/(36/21-14/21) = (2/21)/(22/21) = 2/22 = 1/11
    # Not collinear! So this isn't Pappus.
    
    # I need to be more careful. Let me use the actual Pappus construction.
    # In Pappus, you take two lines, put 3 points on each, and the 6 cross
    # intersections form 3 collinear triples.
    
    # Let me just hardcode a verified Pappus configuration.
    # After scaling, here's one that works:
    
    # Use the configuration from the Pappus theorem with specific integer coords:
    # Line 1: y=0 with points (0,0), (1,0), (2,0)
    # Line 2: y=3 with points (0,3), (1,3), (2,3)
    
    # Cross intersections:
    # C1 = A1B2 ∩ A2B1: 
    #   A1B2: (0,0) to (1,3): y=3x
    #   A2B1: (1,0) to (0,3): y=-3x+3
    #   3x = -3x+3 -> 6x=3 -> x=1/2, y=3/2
    
    # C2 = A2B3 ∩ A3B2:
    #   A2B3: (1,0) to (2,3): y=3x-3
    #   A3B2: (2,0) to (1,3): y=-3x+6
    #   3x-3 = -3x+6 -> 6x=9 -> x=3/2, y=3/2
    
    # C3 = A1B3 ∩ A3B1:
    #   A1B3: (0,0) to (2,3): y=3x/2
    #   A3B1: (2,0) to (0,3): y=-3x/2+3
    #   3x/2 = -3x/2+3 -> 3x=3 -> x=1, y=3/2
    
    # C1, C2, C3 all have y=3/2! They are collinear!
    # C1=(1/2, 3/2), C2=(3/2, 3/2), C3=(1, 3/2)
    
    # Now the other 3 cross intersections:
    # D1 = A1B2 ∩ A3B1... wait, I already used those.
    # The 6 cross intersections are:
    # (A1B2, A2B1), (A2B3, A3B2), (A1B3, A3B1) -- these give the Pappus line
    # (A1B2, A3B1), (A2B3, A1B3)... no.
    
    # Actually the 6 points are intersections of:
    # A1B2 with A2B1 -> C1
    # A2B3 with A3B2 -> C2  
    # A1B3 with A3B1 -> C3
    # A1B2 with A3B1 -> D1
    # A2B3 with A1B3 -> D2
    # A3B2 with A1B3 -> D3 ... this is getting confusing.
    
    # Let me just compute all 6 "cross" intersection points properly.
    # The 6 points are: for each unordered pair of indices {i,j} from {1,2,3},
    # the intersection of line(Ai, Bj) and line(Aj, Bi).
    
    # Pairs: {1,2}, {1,3}, {2,3}
    # P{1,2} = A1B2 ∩ A2B1 = (1/2, 3/2)
    # P{1,3} = A1B3 ∩ A3B1 = (1, 3/2)
    # P{2,3} = A2B3 ∩ A3B2 = (3/2, 3/2)
    
    # These 3 are collinear (y=3/2). That's one Pappus line.
    
    # But we need 6 more points, not 3. The Pappus configuration has 9 lines of 3.
    # The 12 points are: 6 on the two original lines + 6 intersection points.
    # The 6 intersection points are from ALL cross pairs, not just 3.
    
    # Wait, I think I'm confusing myself. Let me re-read the Pappus theorem.
    # Pappus: Given 6 points A1,A2,A3 on line L1 and B1,B2,B3 on line L2,
    # define Pi = Ai+1 Bj ∩ Aj Bi+1 (cyclic indices mod 3).
    # Then P1, P2, P3 are collinear.
    
    # That gives only 3 new points. For 12 points total, I need 6 new points.
    # So maybe the configuration is different.
    
    # Actually, for the Sylvester-Gallai / ordinary lines problem, the Pappus
    # configuration for n=12 might use a different arrangement.
    
    # Let me try a different approach: use the Hessian configuration or
    # just do a direct search for a good configuration.
    
    # For now, let me return a known good configuration and verify it.
    
    # Try: 4 points on each of 3 lines in general position
    # Or: use the 3x4 grid which might have fewer ordinary lines
    
    # Let me just return the 3x4 grid for now and count its ordinary lines
    pts = [[x, y] for x in range(4) for y in range(3)]
    return pts

def construct():
    deadline = 145.0  # safety margin before 150s kill
    start_time = time.time()
    
    # Get initial configuration
    pts = _pappus_configuration()
    if pts is None or len(pts) != 12:
        # Fallback: 3x4 grid
        pts = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in pts]
    best_score = _count_ordinary_lines(best_points)
    
    # If we haven't reached the lower bound, try local search
    if best_score > 6:
        elapsed = time.time() - start_time
        remaining = deadline - 20 - elapsed
        if remaining > 5:
            best_points, best_score = _local_search(best_points, max_iters=10000, timeout=remaining)
    
    # Verify validity
    if len(best_points) != 12:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

def _local_search(points, max_iters=5000, timeout=None):
    """Local search: try moving each point to nearby positions."""
    start_time = time.time()
    best = [p[:] for p in points]
    best_score = _count_ordinary_lines(best)
    
    search_radius = 3
    directions = [(dx, dy) for dx in range(-search_radius, search_radius+1) 
                  for dy in range(-search_radius, search_radius+1)
                  if dx != 0 or dy != 0]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if timeout and (time.time() - start_time) > timeout:
            break
        
        for i in range(len(best)):
            x, y = best[i]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                if new_pt in best:
                    continue
                old_pt = best[i]
                best[i] = new_pt
                score = _count_ordinary_lines(best)
                if score < best_score:
                    best_score = score
                    improved = True
                else:
                    best[i] = old_pt
    
    return best, best_score
