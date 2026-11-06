import time
import random
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Define the grid
    # Use a smaller grid to increase density of collinear triples
    # Points must be distinct integers with |x|, |y| <= 100
    # Let's use [-30, 30] for x and y to get a dense set
    grid_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # 2. Precompute lines
    # A line is determined by its normal vector (A, B, C) for Ax + By + C = 0
    # We normalize the normal vector to avoid duplicates
    # gcd(A, B) must be 1. C is determined by the first point.
    
    lines = defaultdict(set)
    
    # For each pair of points, determine the line they form
    # This is O(N^2) which is 61^2 ~ 3721 pairs, very fast
    for i in range(len(grid_points)):
        p1 = grid_points[i]
        for j in range(i + 1, len(grid_points)):
            p2 = grid_points[j]
            # Vector p1->p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normal vector (A, B) = (dy, -dx)
            A = dy
            B = -dx
            
            # Normalize normal vector
            if A == 0 and B == 0:
                continue
            g = __import__('math').gcd(abs(A), abs(B))
            A //= g
            B //= g
            
            # Ensure canonical form: A > 0 or (A == 0 and B > 0)
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
            
            # Calculate C such that A*x1 + B*y1 + C = 0
            C = -(A * p1[0] + B * p1[1])
            
            line_key = (A, B, C)
            lines[line_key].add(p1)
            lines[line_key].add(p2)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Filter lines that have at least 3 points
    # Also, we need to know for each point, which lines it belongs to
    point_lines = defaultdict(list)
    valid_lines = {}
    for key, pts in lines.items():
        if len(pts) >= 3:
            valid_lines[key] = pts
            for p in pts:
                point_lines[p].append(key)
                
    # 3. Initial Configuration
    # We need 20 points.
    # Start with a random selection, ensuring no 4 collinear.
    
    def is_valid_selection(points):
        # Check if any line has >= 4 points from 'points'
        # We can count points on each line
        counts = defaultdict(int)
        for p in points:
            for line_key in point_lines[p]:
                counts[line_key] += 1
        for count in counts.values():
            if count >= 4:
                return False
        return True

    def count_triples(points):
        # Count number of collinear triples
        counts = defaultdict(int)
        for p in points:
            for line_key in point_lines[p]:
                counts[line_key] += 1
        total = 0
        for count in counts.values():
            if count >= 3:
                total += count * (count - 1) * (count - 2) // 6
        return total

    # Find a valid initial set
    # Random sampling might be slow if collisions are frequent, but grid is dense.
    # Better: Start with a known valid set or just pick random points and repair.
    
    best_score = 0
    best_points = None
    best_triples = -1
    
    # Try several random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomly pick 20 points
        current_points = set(random.sample(grid_points, 20))
        
        # Repair: If invalid, swap points until valid
        # This might take long. Let's try a different initialization:
        # Pick points one by one, ensuring validity.
        
        current_points = set()
        available = list(grid_points)
        random.shuffle(available)
        
        for p in available:
            if len(current_points) == 20:
                break
            # Check if adding p keeps it valid
            temp_points = list(current_points) + [p]
            if is_valid_selection(temp_points):
                current_points.add(p)
        
        if len(current_points) < 20:
            # Fallback: Just take first 20 that are valid? 
            # If we can't find 20, the grid might be too constrained? 
            # With 3721 points and "no 4 collinear", it should be easy to find 20.
            # If stuck, try a different random order
            continue
            
        current_triples = count_triples(current_points)
        
        if current_triples > best_triples:
            best_triples = current_triples
            best_points = set(current_points)
            
        # Local Search
        # Swap one point in current_points with one point outside
        # to increase triples count while maintaining validity
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all pairs (in, out)
            # To save time, we can just try a random subset of swaps or all if fast enough
            # 20 * (3721-20) ~ 74000 swaps. Each swap check is O(1) if we update counts incrementally?
            # Or O(N) if we recount. Let's use incremental update for speed.
            
            # Precompute line counts for current set
            line_counts = defaultdict(int)
            for p in current_points:
                for lk in point_lines[p]:
                    line_counts[lk] += 1
            
            # Helper to get triple count from line counts
            def get_triple_count(lc):
                t = 0
                for c in lc.values():
                    if c >= 3:
                        t += c * (c - 1) * (c - 2) // 6
                return t
            
            cur_triples = get_triple_count(line_counts)
            
            # Try swaps
            # We can't iterate all 74k swaps with full recount if slow.
            # But get_triple_count is O(L) where L is number of lines touched.
            # Let's just do a limited number of random swaps or a greedy pass.
            
            # Greedy Pass: Find the best swap
            best_swap = None
            best_swap_triples = cur_triples
            
            # Sample a subset of outside points to speed up search
            outside_points = [p for p in grid_points if p not in current_points]
            random.shuffle(outside_points)
            outside_sample = outside_points[:100] # Check 100 candidates
            
            for p_in in list(current_points):
                for p_out in outside_sample:
                    # Check validity of swap
                    # Remove p_in, Add p_out
                    
                    # Check if p_out creates 4-collinear with existing points (minus p_in)
                    # And if p_in's removal doesn't leave a 4-collinear (it won't, since it was valid)
                    
                    # Simulate counts
                    # We need to check lines affected by p_in and p_out
                    
                    # This incremental check is complex. 
                    # Given 150s limit, we can afford O(N) checks for a few hundred swaps.
                    
                    # Let's just do a simple check:
                    # If we swap, is the new set valid?
                    new_points_set = (current_points - {p_in}) | {p_out}
                    if is_valid_selection(new_points_set):
                        new_triples = count_triples(new_points_set)
                        if new_triples > best_swap_triples:
                            best_swap_triples = new_triples
                            best_swap = (p_in, p_out)
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            
            if best_swap:
                p_in, p_out = best_swap
                current_points.remove(p_in)
                current_points.add(p_out)
                current_triples = best_swap_triples
                improved = True
                
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = set(current_points)
                    
    # If we didn't find a valid set (unlikely), return a known valid one
    if best_points is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    # Convert to list of lists
    result = [list(p) for p in best_points]
    
    # Final verification
    if not is_valid_selection(result):
        return [[x, x * x] for x in range(-10, 10)]
        
    return result

# EVOLVE-BLOCK-END
