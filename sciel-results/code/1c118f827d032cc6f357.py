import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) = a + b*alpha
    def to_idx(x):
        return x[0] + x[1] * p

    def from_idx(idx):
        return (idx % p, idx // p)

    def field_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)

    def field_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)

    # Precompute addition and multiplication tables
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        xi = from_idx(i)
        for j in range(q):
            xj = from_idx(j)
            add_table[i, j] = to_idx(field_add(xi, xj))
            mul_table[i, j] = to_idx(field_mul(xi, xj))

    # Build line structure
    # 50 directions: (1, m) for m in F_49 (49 dirs) and (0, 1) (1 dir)
    # Total lines: 50 * 49 = 2450

    line_points = []
    point_to_lines = defaultdict(set)

    # Direction (1, m), m_idx in 0..48
    for m_idx in range(q):
        m = from_idx(m_idx)
        for c_idx in range(q):
            c = from_idx(c_idx)
            line_id = m_idx * q + c_idx
            pts = []
            for x_idx in range(q):
                x = from_idx(x_idx)
                mx = field_mul(m, x)
                y = field_add(mx, c)
                y_idx = to_idx(y)
                point_idx = x_idx * q + y_idx
                pts.append(point_idx)
                point_to_lines[point_idx].add(line_id)
            line_points.append(pts)

    # Direction (0, 1): vertical lines x = c
    for c_idx in range(q):
        line_id = q * q + c_idx
        pts = []
        for y_idx in range(q):
            point_idx = c_idx * q + y_idx
            pts.append(point_idx)
            point_to_lines[point_idx].add(line_id)
        line_points.append(pts)

    # Now model as max-flow / min-vertex-cover problem
    # We want to find the largest set C of excluded points such that
    # every point has at least one "clean" line (a line with all other points in N)
    
    # Reformulation: A line L is "covered" by C if L contains at least one point from C.
    # For point x to be valid, at least one of the 50 lines through x must NOT be covered by C.
    # Equivalently: the set of lines through x that are covered by C must have size < 50.
    
    # This is equivalent to: we need to select lines to "keep clean" such that:
    # 1. Every point x has at least one clean line through it
    # 2. If line L is clean, then no point on L (other than potentially x itself) is in C
    #    Actually: if L is clean, then ALL points on L except possibly the anchor are in N
    
    # Better formulation using bipartite graph:
    # Left nodes: points (2401 nodes) - these are candidates for C
    # Right nodes: lines (2450 nodes)
    # Edge from point p to line L if p lies on L
    # 
    # We want to find a set C of points and a set of lines S (the "clean" lines) such that:
    # - Every point p has at least one line through p in S
    # - If line L is in S, then NO point on L is in C (all points on L are in N)
    # - Maximize |C| = 2401 - |N|
    #
    # This is a maximum independent set in a bipartite graph where:
    # - Left: points
    # - Right: lines  
    # - Edge: point lies on line
    # - Independent set = set of nodes with no edges between them
    # 
    # But wait, that's not quite right. Let me think differently.
    
    # Alternative: Use the duality between vertex cover and matching in bipartite graphs.
    # 
    # Consider this: We want to choose which lines to "sacrifice" (make non-clean).
    # A point x is "covered" if ALL 50 lines through x are sacrificed.
    # We want to minimize the number of covered points (or maximize uncovered points in N).
    #
    # Actually, let's use a different approach: local search with constraint satisfaction.
    # 
    # Key insight: The problem is to find the maximum size set C such that
    # for every point x, not all 50 lines through x contain a point of C.
    #
    # This is equivalent to: find minimum set of lines such that every point
    # is incident to at least one line in this set, AND no two points on the same
    # selected line can both be in C... 
    #
    # Let me try a direct max-flow formulation:
    # 
    # Source -> point nodes (capacity 1)
    # Point node -> line node (capacity 1) if point on line
    # Line node -> sink (capacity 48)  [each line can "absorb" up to 48 points]
    #
    # Wait, this doesn't capture the constraint correctly.
    
    # Let me try a different structural approach: use simulated annealing / local search
    # on the set C, with a clever initialization and move set.
    
    # Initialize C using a structured pattern
    # For F_49^2, we can try: C = {(x,y) : Tr(x) + N(y) = 0} where Tr(x)=2*x_a, N(y)=y_a^2-3*y_b^2
    # This gives about 2401/7 = 343 points
    
    def in_unital(point_idx):
        x_idx = point_idx // q
        y_idx = point_idx % q
        xa = x_idx % p
        ya = y_idx % p
        yb = y_idx // p
        return (2 * xa + ya * ya - 3 * yb * yb) % p == 0

    # Build initial C from unital
    C = set()
    for point_idx in range(total_points):
        if in_unital(point_idx):
            C.add(point_idx)

    # Precompute affected points for each point
    affected = []
    for x_idx in range(total_points):
        aff = []
        for line_id in point_to_lines[x_idx]:
            for z in line_points[line_id]:
                if z != x_idx:
                    aff.append(z)
        affected.append(aff)

    # Clean line count: for each point, how many of its 50 lines have NO point in C
    # Initially compute for the unital-based C
    def compute_clean_counts(C_set):
        clean_counts = np.zeros(total_points, dtype=np.int32)
        for point_idx in range(total_points):
            count = 0
            for line_id in point_to_lines[point_idx]:
                # Check if any point on this line (other than point_idx) is in C
                has_c = False
                for z in line_points[line_id]:
                    if z != point_idx and z in C_set:
                        has_c = True
                        break
                if not has_c:
                    count += 1
            clean_counts[point_idx] = count
        return clean_counts

    clean_counts = compute_clean_counts(C)

    # Verify: all points should have at least 1 clean line
    # If not, adjust C

    # Local search: try to add points to C (increasing |C|)
    # A point x can be added to C if after adding, all points still have >= 1 clean line
    # Adding x to C: the 50 lines through x become "dirty" (if they weren't already)
    # For each such line L, the other 48 points on L lose L as a clean line
    
    # More efficient: track for each line whether it's clean (no C-points on it except possibly the point we're considering)
    
    # Let's use a different representation:
    # line_has_c[line_id] = True if line has at least one point in C
    # point_clean_count[point_idx] = number of lines through point_idx that are clean
    
    def build_line_state(C_set):
        line_has_c = np.zeros(len(line_points), dtype=bool)
        for point_idx in C_set:
            for line_id in point_to_lines[point_idx]:
                line_has_c[line_id] = True
        return line_has_c

    def compute_point_clean_counts(line_has_c):
        counts = np.zeros(total_points, dtype=np.int32)
        for point_idx in range(total_points):
            count = 0
            for line_id in point_to_lines[point_idx]:
                if not line_has_c[line_id]:
                    count += 1
            counts[point_idx] = count
        return counts

    line_has_c = build_line_state(C)
    point_clean_counts = compute_point_clean_counts(line_has_c)

    # Now local search: try to add points to C
    # Priority: points that are on many dirty lines (adding them doesn't make new lines dirty)
    
    def can_add_point(x, line_has_c_arr, point_clean_counts_arr):
        # Check if adding x to C keeps all points with >= 1 clean line
        # Adding x makes all 50 lines through x dirty
        # For each such line L, the 48 other points on L lose L
        # So for each affected point z, check if it still has >= 1 clean line after losing these lines
        
        # The lines through x that are currently clean will become dirty
        newly_dirty_lines = []
        for line_id in point_to_lines[x]:
            if not line_has_c_arr[line_id]:
                newly_dirty_lines.append(line_id)
        
        if not newly_dirty_lines:
            return True  # All lines through x are already dirty, adding x doesn't change anything
        
        # For each affected point, check if it loses all its clean lines
        # Affected points: all z != x on any of the newly_dirty_lines
        affected_set = set()
        for line_id in newly_dirty_lines:
            for z in line_points[line_id]:
                if z != x:
                    affected_set.add(z)
        
        for z in affected_set:
            # Count how many clean lines z has that are NOT in newly_dirty_lines
            remaining_clean = 0
            for line_id in point_to_lines[z]:
                if not line_has_c_arr[line_id] and line_id not in newly_dirty_lines:
                    remaining_clean += 1
            if remaining_clean < 1:
                return False
        return True

    # This is O(2400 * 50) per check, which is ~120k operations
    # With 2401 points to try, that's ~288M operations per pass - too slow in Python
    
    # Use numpy for speed
    def can_add_point_fast(x, line_has_c_arr, point_clean_counts_arr):
        # Get lines through x that are currently clean
        lines_through_x = list(point_to_lines[x])
        clean_lines_through_x = [lid for lid in lines_through_x if not line_has_c_arr[lid]]
        
        if not clean_lines_through_x:
            return True
        
        # For each affected point, check remaining clean lines
        # Use a set for O(1) lookup
        clean_lines_set = set(clean_lines_through_x)
        
        # Collect affected points
        affected = set()
        for lid in clean_lines_through_x:
            for z in line_points[lid]:
                if z != x:
                    affected.add(z)
        
        for z in affected:
            # Count clean lines through z that are not in clean_lines_through_x
            remaining = 0
            for lid in point_to_lines[z]:
                if not line_has_c_arr[lid] and lid not in clean_lines_set:
                    remaining += 1
                    if remaining >= 1:
                        break
            if remaining < 1:
                return False
        return True

    # This is still slow. Let me precompute more aggressively.
    
    # Precompute for each point z, the list of lines through z
    # And for each pair of points (x, z) where z is on a line through x, 
    # precompute which lines through z are also lines through x
    
    # Actually, let's just do a simple greedy with numpy optimization
    # and accept that we'll get a smaller C than optimal
    
    # Greedy: repeatedly add the point that allows the most additional points to be added
    # But for now, just do a simple pass: try to add each point if possible
    
    t_start = time.time()
    added_count = 0
    for x in range(total_points):
        if time.time() > deadline - safety_margin:
            break
        if x in C:
            continue
        if can_add_point_fast(x, line_has_c, point_clean_counts):
            C.add(x)
            # Update line_has_c
            for lid in point_to_lines[x]:
                line_has_c[lid] = True
            # Update point_clean_counts
            for lid in point_to_lines[x]:
                if not line_has_c[lid]:  # This is now True, so skip
                    pass
                for z in line_points[lid]:
                    if z != x:
                        point_clean_counts[z] -= 1
            added_count += 1
    
    # Try to add more points in multiple passes
    for _ in range(5):
        if time.time() > deadline - safety_margin:
            break
        progress = False
        for x in range(total_points):
            if time.time() > deadline - safety_margin:
                break
            if x in C:
                continue
            if can_add_point_fast(x, line_has_c, point_clean_counts):
                C.add(x)
                for lid in point_to_lines[x]:
                    line_has_c[lid] = True
                for lid in point_to_lines[x]:
                    for z in line_points[lid]:
                        if z != x:
                            point_clean_counts[z] -= 1
                progress = True
        if not progress:
            break

    # Build the Nikodym set N = all points not in C
    N_points = []
    for point_idx in range(total_points):
        if point_idx not in C:
            x_idx = point_idx // q
            y_idx = point_idx % q
            xa = x_idx % p
            xb = x_idx // p
            ya = y_idx % p
            yb = y_idx // p
            N_points.append([[xa, xb], [ya, yb]])

    if not N_points:
        N_points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
