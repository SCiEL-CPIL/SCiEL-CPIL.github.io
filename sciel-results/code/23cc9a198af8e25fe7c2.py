import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)

    def build_triple_graph(points):
        """Build graph where vertices are collinear triples; edge = share a point."""
        n = len(points)
        line_to_pts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_to_pts:
                    line_to_pts[key] = []
                line_to_pts[key].append(j)
        
        # Find all lines with >= 3 points
        lines_with_3 = []
        for key, pt_list in line_to_pts.items():
            if len(pt_list) >= 2:  # pt_list has indices of other points on same line as first
                pass
        
        # Better: collect all points per line
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        # Check for 4-collinear
        for key, pts in line_points.items():
            if len(pts) >= 4:
                return None, None  # Invalid
        
        # Generate all triples
        triples = []
        triple_id = 0
        triple_to_points = {}
        for key, pts in line_points.items():
            if len(pts) >= 3:
                pts_list = sorted(pts)
                for combo in itertools.combinations(pts_list, 3):
                    triples.append(combo)
                    triple_to_points[triple_id] = set(combo)
                    triple_id += 1
        
        if not triples:
            return [], set()
        
        # Build conflict graph: two triples conflict if they share a point
        num_triples = len(triples)
        conflicts = [set() for _ in range(num_triples)]
        for i in range(num_triples):
            for j in range(i + 1, num_triples):
                if triple_to_points[i] & triple_to_points[j]:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        return conflicts, set(range(num_triples))

    def exact_max_independent_set(conflicts, vertices):
        """Exact MIS via branch and bound with greedy coloring upper bound."""
        if not vertices:
            return 0
        
        vertices = sorted(vertices)
        n = len(vertices)
        
        # Greedy coloring for upper bound
        def greedy_color_bound(verts):
            """Return upper bound on MIS size using greedy coloring."""
            if not verts:
                return 0
            color_classes = []
            uncolored = set(verts)
            while uncolored:
                color_class = set()
                # Greedily pick vertices that don't conflict with each other
                for v in sorted(uncolored):
                    if not (color_class & conflicts[v]):
                        color_class.add(v)
                color_classes.append(color_class)
                uncolored -= color_class
            return len(color_classes)
        
        best = [0]
        
        def branch(remaining, current_size):
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound via greedy coloring
            ub = current_size + greedy_color_bound(remaining)
            if ub <= best[0]:
                return
            
            # Pick vertex with highest degree in remaining subgraph
            remaining_list = sorted(remaining)
            if not remaining_list:
                return
            
            # Use the first vertex (or highest degree)
            v = remaining_list[0]
            
            # Branch 1: include v
            new_remaining = remaining - {v} - conflicts[v]
            branch(new_remaining, current_size + 1)
            
            # Branch 2: exclude v
            branch(remaining - {v}, current_size)
        
        branch(set(vertices), 0)
        return best[0]

    def get_score(points):
        """Return the maximum subset size with no three collinear.
        Lower is better. Returns negative if invalid (4+ collinear)."""
        n = len(points)
        
        # Check for 4+ collinear
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        for key, pts in line_points.items():
            if len(pts) >= 4:
                return -1000000
        
        # Find all collinear triples
        triples = []
        for key, pts in line_points.items():
            if len(pts) >= 3:
                pts_list = sorted(pts)
                for combo in itertools.combinations(pts_list, 3):
                    triples.append(set(combo))
        
        if not triples:
            return n  # All points can be selected
        
        # Build conflict graph on triples
        num_triples = len(triples)
        conflicts = [set() for _ in range(num_triples)]
        for i in range(num_triples):
            for j in range(i + 1, num_triples):
                if triples[i] & triples[j]:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Max independent set of triples = max number of triples we can "hit"
        # Actually we want: largest subset of points with NO three collinear
        # This is equivalent to: min number of points to remove to destroy all triples
        # Which is: n - (max independent set in the hypergraph dual)
        
        # Alternative: The answer is n - (minimum vertex cover of the 3-uniform hypergraph)
        # For small instances, we can use: answer = n - max_independent_set_of_triples
        # Wait, that's not right either.
        
        # Let me think again. We want the largest subset S of points such that no 3 points in S are collinear.
        # Equivalently: remove the minimum number of points so that no triple remains.
        # This is the minimum vertex cover of the 3-uniform hypergraph defined by triples.
        # By complement: max independent set in the hypergraph = n - min vertex cover
        # But hypergraph independent set means no edge is fully contained.
        
        # For our purposes with small n, let's use a different approach:
        # The score = n - (minimum number of points to remove to break all collinear triples)
        
        # Use branch and bound on point removal
        best_removal = [n]
        
        def can_break_all(removed, idx):
            """Check if all triples are broken given removed set."""
            for t in triples:
                if not (t & removed):
                    return False
            return True
        
        def count_remaining_triples(removed):
            count = 0
            for t in triples:
                if not (t & removed):
                    count += 1
            return count
        
        def branch_removal(removed, idx, remaining_needed):
            if idx >= n:
                if can_break_all(removed, idx):
                    if len(removed) < best_removal[0]:
                        best_removal[0] = len(removed)
                return
            
            # Upper bound: even if we remove all remaining points, can we beat best?
            if len(removed) + (n - idx) <= best_removal[0] and not can_break_all(removed, idx):
                # Actually this bound isn't tight. Let's just proceed.
                pass
            
            # Check if already broken
            if can_break_all(removed, idx):
                if len(removed) < best_removal[0]:
                    best_removal[0] = len(removed)
                return
            
            # Pick a triple that's not yet broken
            target_triple = None
            for t in triples:
                if not (t & removed):
                    target_triple = t
                    break
            
            if target_triple is None:
                if len(removed) < best_removal[0]:
                    best_removal[0] = len(removed)
                return
            
            # Try removing each point in target_triple
            for p in sorted(target_triple):
                if p not in removed:
                    branch_removal(removed | {p}, idx + 1, remaining_needed)
        
        branch_removal(set(), 0, 0)
        
        return n - best_removal[0]

    def generate_line_packed_config(seed_val):
        """Generate a configuration using line-packing heuristic.
        Place points on a small number of lines to maximize triple density."""
        random.seed(seed_val)
        
        # Strategy: pick 3-4 lines, place 5-7 points on each, ensuring no 4 collinear
        # This creates many triples on those lines
        
        num_lines = random.randint(3, 5)
        points = []
        used_positions = set()
        
        for _ in range(num_lines):
            # Pick a random line (random direction and offset)
            # Use small slopes to keep points in range
            slope_choices = [
                (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2),
                (3, 1), (1, 3), (3, -1), (1, -3)
            ]
            dx, dy = random.choice(slope_choices)
            # Normalize
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Pick an offset
            c = random.randint(-80, 80)
            
            # Place points on this line: c = dy*x - dx*y => y = (dy*x - c) / dx if dx != 0
            # Or x = (dx*y + c) / dy if dy != 0
            
            num_on_line = random.randint(4, 6)
            line_points = []
            
            if dx == 0:
                # Vertical line: x = -c/dy... wait, c = dy*x - dx*y = dy*x
                # So x = c/dy
                x_val = c // dy if dy != 0 else 0
                if -100 <= x_val <= 100:
                    for k in range(num_on_line):
                        y_val = random.randint(-100, 100)
                        pos = (x_val, y_val)
                        if pos not in used_positions:
                            line_points.append(pos)
                            used_positions.add(pos)
            elif dy == 0:
                # Horizontal line: c = -dx*y => y = -c/dx
                y_val = -c // dx if dx != 0 else 0
                if -100 <= y_val <= 100:
                    for k in range(num_on_line):
                        x_val = random.randint(-100, 100)
                        pos = (x_val, y_val)
                        if pos not in used_positions:
                            line_points.append(pos)
                            used_positions.add(pos)
            else:
                # General line: c = dy*x - dx*y
                # Pick x values and compute y
                for k in range(num_on_line):
                    x_val = random.randint(-90, 90)
                    # y = (dy*x - c) / dx
                    num = dy * x_val - c
                    if num % dx == 0:
                        y_val = num // dx
                        if -100 <= y_val <= 100:
                            pos = (x_val, y_val)
                            if pos not in used_positions:
                                line_points.append(pos)
                                used_positions.add(pos)
            
            points.extend(line_points)
        
        # Fill remaining points randomly
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            pos = (x, y)
            if pos not in used_positions:
                points.append(pos)
                used_positions.add(pos)
        
        # Truncate to 20 if needed
        points = points[:20]
        
        # Ensure no 4 collinear
        for _ in range(50):
            line_points_check = {}
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    key = line_key(points[i], points[j])
                    if key not in line_points_check:
                        line_points_check[key] = set()
                    line_points_check[key].add(i)
                    line_points_check[key].add(j)
            
            has_four = False
            for key, pts in line_points_check.items():
                if len(pts) >= 4:
                    has_four = True
                    # Move one point off this line
                    idx = random.choice(list(pts))
                    points[idx] = (random.randint(-100, 100), random.randint(-100, 100))
                    break
            
            if not has_four:
                break
        
        # Ensure distinct
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        while len(unique_points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                unique_points.append((x, y))
                seen.add((x, y))
        
        return [list(p) for p in unique_points[:20]]

    def local_search_line_packed(points, time_limit):
        """Local search that tries to increase triple density while maintaining validity."""
        current_pts = [p[:] for p in points]
        
        def quick_score(pts):
            """Quick validity check and triple count."""
            line_counts = {}
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    key = line_key(pts[i], pts[j])
                    if key not in line_counts:
                        line_counts[key] = set()
                    line_counts[key].add(i)
                    line_counts[key].add(j)
            
            for key, pts_in_line in line_counts.items():
                if len(pts_in_line) >= 4:
                    return -1000000
            
            triples = 0
            for key, pts_in_line in line_counts.items():
                cnt = len(pts_in_line)
                if cnt >= 3:
                    triples += cnt * (cnt - 1) * (cnt - 2) // 6
            
            return triples
        
        current_score = quick_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        T = 5.0
        cooling = 0.98
        
        while time.time() < time_limit and T > 0.1:
            i = random.randint(0, 19)
            old_p = current_pts[i]
            
            # Move point to create or join a collinear triple
            if random.random() < 0.8:
                # Pick two other points and try to align
                others = [x for x in range(20) if x != i]
                j, k = random.sample(others, 2)
                p_j, p_k = current_pts[j], current_pts[k]
                dx = p_k[0] - p_j[0]
                dy = p_k[1] - p_j[1]
                g = gcd(abs(dx), abs(dy))
                if g > 0:
                    dx_n = dx // g
                    dy_n = dy // g
                    # Find integer points on this line near p_j
                    found = False
                    for t in range(-8, 9):
                        if t == 0:
                            continue
                        new_p = [p_j[0] + t * dx_n, p_j[1] + t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in current_pts and new_p != old_p:
                                current_pts[i] = new_p
                                found = True
                                break
                    if not found:
                        continue
                else:
                    continue
            else:
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p == old_p or new_p in current_pts:
                    continue
                if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                    continue
                current_pts[i] = new_p
            
            new_score = quick_score(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = pow(2.71828, -abs(delta) / T)
                    if random.random() < prob:
                        current_score = new_score
                else:
                    current_score = new_score
            
            T *= cooling
        
        return best_pts

    # Main search loop
    best_overall = None
    best_overall_score = -1

    # Try multiple line-packed configurations
    for seed_val in range(100):
        if time.time() > deadline - 10:
            break
        
        pts = generate_line_packed_config(seed_val)
        
        # Quick check
        quick = get_score(pts)
        if quick < 0:
            continue
        
        # Refine with local search
        search_end = min(time.time() + 3, deadline - 5)
        if search_end - time.time() < 1:
            search_end = deadline - 2
        
        refined = local_search_line_packed(pts, search_end)
        score = get_score(refined)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = refined
    
    if best_overall is None:
        # Fallback: simple valid configuration
        best_overall = [[i, 0] for i in range(20)]
    
    return best_overall
