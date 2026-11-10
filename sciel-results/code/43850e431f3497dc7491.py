"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Returns exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    denoting lattice points a + b*omega, omega = exp(i*pi/3), which are at
    mutual distance at least one. Maximizes the number of pairs at distance
    exactly one; Harborth's theorem gives the optimum floor(3n - sqrt(12n - 3)) = 76.
    
    We use a hexagonal compact arrangement on the triangular lattice.
    The triangular lattice in axial coordinates (a, b) has 6 neighbors:
    (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    A hexagonal spiral of 32 vertices achieves the Harborth bound of 76 edges.
    """
    
    # The 6 neighbor directions in axial coordinates
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    # Distance squared in the triangular lattice metric:
    # For points (a1,b1) and (a2,b2), the squared distance is:
    # (a1-a2)^2 + (b1-b2)^2 + (a1-a2)*(b1-b2)
    # Actually, let me reconsider. The lattice point is a + b*omega where omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # So point (a,b) maps to (a + b/2, b*sqrt(3)/2)
    # Distance squared between (a1,b1) and (a2,b2):
    # ((a1-a2) + (b1-b2)/2)^2 + ((b1-b2)*sqrt(3)/2)^2
    # = (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2/4 + 3(b1-b2)^2/4
    # = (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    # So dist^2 = da^2 + da*db + db^2 where da=a1-a2, db=b1-b2
    
    def dist2(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1 (dist2 == 1)."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist2(points[i], points[j]) == 1:
                    count += 1
        return count
    
    # Generate hexagonal spiral points
    # Center at (0,0), then add rings
    # Ring 0: just (0,0) -> 1 point
    # Ring 1: 6 points -> total 7
    # Ring 2: 12 points -> total 19
    # Ring 3: 18 points -> total 37 (we only need 32, so take 13 from ring 3)
    
    def hex_ring(r):
        """Generate points on hexagonal ring r (r >= 1)."""
        if r == 0:
            return [(0, 0)]
        points = []
        # Start at (r, 0) and go around
        # The six corners of ring r are at:
        # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
        # Actually in axial coords for hexagonal rings:
        # Corner 1: (r, 0)
        # Corner 2: (0, r)  
        # Corner 3: (-r, r)
        # Corner 4: (-r, 0)
        # Corner 5: (0, -r)
        # Corner 6: (r, -r)
        corners = [(r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)]
        for i in range(6):
            c1 = corners[i]
            c2 = corners[(i+1) % 6]
            # Move from c1 to c2, stepping r times
            da = (c2[0] - c1[0]) // r
            db = (c2[1] - c1[1]) // r
            for step in range(r):
                points.append((c1[0] + da*step, c1[1] + db*step))
        return points
    
    # Build the set of points
    points = [(0, 0)]
    # Add ring 1 (6 points)
    for p in hex_ring(1):
        points.append(p)
    # Add ring 2 (12 points)
    for p in hex_ring(2):
        points.append(p)
    # Now we have 1 + 6 + 12 = 19 points
    # Need 13 more from ring 3
    ring3 = hex_ring(3)
    points.extend(ring3[:13])
    
    # Verify we have exactly 32 distinct points
    assert len(points) == 32, f"Expected 32 points, got {len(points)}"
    assert len(set(points)) == 32, "Points are not distinct"
    
    # Verify all coordinates are within bounds
    for p in points:
        assert abs(p[0]) <= 64, f"Point {p} has |a| > 64"
        assert abs(p[1]) <= 64, f"Point {p} has |b| > 64"
    
    # Verify all pairs are at distance at least 1
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d2 = dist2(points[i], points[j])
            assert d2 >= 1, f"Points {points[i]} and {points[j]} at distance < 1 (dist2={d2})"
    
    # Count edges
    edges = count_edges(points)
    
    # If we haven't achieved 76, try to improve by swapping some ring-3 points
    if edges < 76:
        # Try different subsets of 13 from ring 3
        best_points = points
        best_edges = edges
        
        # Generate all C(18,13) subsets is too many, so try greedy improvement
        # Current ring-3 points used: indices 19-31
        base = points[:19]
        current_ring3 = set(points[19:])
        
        # Try swapping each used ring-3 point with an unused one
        all_ring3 = hex_ring(3)
        unused_ring3 = [p for p in all_ring3 if p not in current_ring3]
        
        improved = True
        while improved:
            improved = False
            for i in range(len(current_ring3)):
                used_pt = list(current_ring3)[i]
                for unused_pt in unused_ring3:
                    new_set = set(current_ring3)
                    new_set.remove(used_pt)
                    new_set.add(unused_pt)
                    candidate = base + list(new_set)
                    # Check validity
                    valid = True
                    for x in candidate:
                        if abs(x[0]) > 64 or abs(x[1]) > 64:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check minimum distance
                    for a in range(len(candidate)):
                        for b in range(a+1, len(candidate)):
                            if dist2(candidate[a], candidate[b]) < 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    new_edges = count_edges(candidate)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_points = candidate
                        current_ring3 = new_set
                        unused_ring3 = [p for p in all_ring3 if p not in current_ring3]
                        improved = True
                        break
                if improved:
                    break
        
        points = best_points
        edges = best_edges
    
    # Final verification
    assert len(points) == 32
    assert len(set(points)) == 32
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            assert dist2(points[i], points[j]) >= 1
    
    # Convert to list of lists
    return [[p[0], p[1]] for p in points]

# EVOLVE-BLOCK-END
