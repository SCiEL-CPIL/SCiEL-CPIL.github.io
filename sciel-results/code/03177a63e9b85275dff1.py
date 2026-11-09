"""Construct 32 lattice points on triangular lattice maximizing nearest-neighbor pairs.

Uses a structured hexagonal arrangement (hex(2) + best arc from ring 3) as seed,
then applies efficient local search to reach the Harborth optimum of 76 edges.
"""

import time
import random

def construct():
    """Return exactly 32 distinct integer pairs [a, b] with |a|,|b| <= 64
    on the triangular lattice (point = a + b*omega, omega = exp(i*pi/3))
    maximizing the number of pairs at distance exactly 1.
    
    Optimal: 76 edges (Harborth's theorem for n=32).
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin from 150s limit

    # The 6 unit neighbor directions on triangular lattice
    # (a,b) -> (a+1,b), (a,b+1), (a-1,b+1) and negatives
    NEIGHBORS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def count_edges(point_set):
        """Count number of pairs at distance exactly 1."""
        ps = set(point_set)
        count = 0
        for (a, b) in ps:
            for (da, db) in NEIGHBORS:
                if (a + da, b + db) in ps:
                    count += 1
        return count // 2

    def generate_hex_layers(max_ring=3):
        """Generate points by hexagonal ring using BFS."""
        visited = set()
        layers = {0: [(0, 0)]}
        visited.add((0, 0))
        current_layer = [(0, 0)]
        
        for r in range(1, max_ring + 1):
            next_layer = []
            for (a, b) in current_layer:
                for (da, db) in NEIGHBORS:
                    np = (a + da, b + db)
                    if np not in visited:
                        visited.add(np)
                        next_layer.append(np)
            layers[r] = next_layer
            current_layer = next_layer
        
        return layers

    def get_best_arc(layers, base_ring=2, arc_len=13):
        """Select the best contiguous arc of points from ring 3."""
        base_points = []
        for r in range(0, base_ring + 1):
            base_points.extend(layers[r])
        base_set = set(base_points)
        
        ring3 = layers[base_ring + 1]
        n_ring = len(ring3)
        
        # Sort ring 3 by angle to get cyclic order
        import math
        def angle(p):
            a, b = p
            x = a + 0.5 * b
            y = math.sqrt(3) / 2 * b
            return math.atan2(y, x)
        
        ring3_sorted = sorted(ring3, key=angle)
        
        # For each point, count neighbors in base
        neighbor_counts = {}
        for p in ring3_sorted:
            a, b = p
            count = 0
            for (da, db) in NEIGHBORS:
                if (a + da, b + db) in base_set:
                    count += 1
            neighbor_counts[p] = count
        
        # Find best contiguous block of arc_len
        best_score = -1
        best_start = 0
        for i in range(n_ring):
            s = 0
            for j in range(arc_len):
                idx = (i + j) % n_ring
                s += neighbor_counts[ring3_sorted[idx]]
            if s > best_score:
                best_score = s
                best_start = i
        
        selected = []
        for j in range(arc_len):
            idx = (best_start + j) % n_ring
            selected.append(ring3_sorted[idx])
        
        return base_points + selected

    # Generate candidate pool for local search
    # Use a larger bounding box to allow movement
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
    candidate_set = set(candidates)

    # Step 1: Structured seed
    layers = generate_hex_layers(max_ring=3)
    points = get_best_arc(layers, base_ring=2, arc_len=13)
    
    # Ensure exactly 32 distinct points
    point_set = set(points)
    if len(point_set) != 32:
        # Fallback: add/remove to get 32
        points = list(point_set)
        while len(points) < 32:
            for c in candidates:
                if c not in point_set:
                    points.append(c)
                    point_set.add(c)
                    break
        while len(points) > 32:
            points.pop()
            point_set.discard(points[-1] if points else None)
            # Rebuild set
            point_set = set(points)

    points = list(point_set)
    best_edges = count_edges(set(points))
    
    # Step 2: Efficient local search with incremental edge counting
    def local_search_efficient(points, max_iter=5000):
        """Local search with incremental edge count updates."""
        pts = list(points)
        pts_set = set(pts)
        current_edges = count_edges(pts_set)
        
        # Precompute: for each candidate, which neighbors are in candidate pool
        # and precompute neighbor lists
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            if time.time() - start_time > DEADLINE:
                break
            
            # Try each position: remove pts[i], try adding each candidate
            for i in range(len(pts)):
                old_pt = pts[i]
                old_a, old_b = old_pt
                
                # Remove old point
                pts_set.remove(old_pt)
                edges_after_remove = current_edges
                # Subtract edges that old_pt had with remaining points
                for (da, db) in NEIGHBORS:
                    if (old_a + da, old_b + db) in pts_set:
                        edges_after_remove -= 1
                
                best_new_pt = None
                best_new_edges = edges_after_remove
                best_gain = -1
                
                for (a, b) in candidates:
                    if (a, b) in pts_set:
                        continue
                    # Check distance >= 1 from all other points
                    valid = True
                    for (pa, pb) in pts_set:
                        da = a - pa
                        db = b - pb
                        d2 = da * da + da * db + db * db
                        if d2 < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Count new edges
                    new_edges = 0
                    for (da, db) in NEIGHBORS:
                        if (a + da, b + db) in pts_set:
                            new_edges += 1
                    
                    total = edges_after_remove + new_edges
                    if total > best_new_edges:
                        best_new_edges = total
                        best_gain = new_edges
                        best_new_pt = (a, b)
                
                if best_new_pt is not None and best_new_edges > current_edges:
                    pts[i] = best_new_pt
                    pts_set.add(best_new_pt)
                    current_edges = best_new_edges
                    improved = True
                    break
                else:
                    # Restore old point
                    pts_set.add(old_pt)
        
        return pts, current_edges

    points, best_edges = local_search_efficient(points, max_iter=3000)

    # Step 3: Multiple restarts with perturbation
    if time.time() - start_time < DEADLINE:
        random.seed(42)
        for restart in range(10):
            if time.time() - start_time > DEADLINE:
                break
            
            # Perturb current best
            perturbed = list(points)
            n_perturb = 2 + restart
            for _ in range(n_perturb):
                idx = random.randint(0, len(perturbed) - 1)
                new_pt = random.choice(candidates)
                if new_pt not in perturbed:
                    perturbed[idx] = new_pt
            
            # Ensure distinctness
            perturbed_set = set(perturbed)
            if len(perturbed_set) != 32:
                # Fix duplicates
                seen = set()
                fixed = []
                for p in perturbed:
                    if p not in seen:
                        fixed.append(p)
                        seen.add(p)
                while len(fixed) < 32:
                    for c in candidates:
                        if c not in seen:
                            fixed.append(c)
                            seen.add(c)
                            break
                perturbed = fixed
            
            pts, edges = local_search_efficient(perturbed, max_iter=2000)
            if edges > best_edges:
                best_edges = edges
                points = pts

    # Final verification
    final_set = set(points)
    final_edges = count_edges(final_set)
    
    assert len(final_set) == 32, f"Expected 32 points, got {len(final_set)}"
    for (a, b) in final_set:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
        # Verify mutual distance >= 1
    # Check all pairs have distance >= 1
    pts_list = list(final_set)
    for i in range(len(pts_list)):
        for j in range(i + 1, len(pts_list)):
            da = pts_list[i][0] - pts_list[j][0]
            db = pts_list[i][1] - pts_list[j][1]
            d2 = da * da + da * db + db * db
            assert d2 >= 1, f"Points {pts_list[i]} and {pts_list[j]} too close"

    result = [[a, b] for (a, b) in points]
    return result
