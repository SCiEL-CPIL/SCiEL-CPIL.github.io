"""Improved program: Construct 32 points on triangular lattice maximizing unit-distance edges."""
import time
import random
import math

def _neighbors(p):
    """Return the 6 neighbors of point (a, b) on triangular lattice."""
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(point_set):
    """Count number of unit-distance pairs in point_set (a set of tuples)."""
    cnt = 0
    for p in point_set:
        for q in _neighbors(p):
            if q in point_set and p < q:
                cnt += 1
    return cnt

def _valid(point_set):
    """Check if point_set has 32 distinct points with |a|,|b| <= 64."""
    if len(point_set) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in point_set)

def _hexagonal_initial():
    """
    Generate a near-hexagonal configuration of 32 points.
    Use concentric hexagonal rings:
    - Ring 0: center (1 point)
    - Ring 1: 6 points
    - Ring 2: 12 points
    - Ring 3: 18 points
    Total so far: 37, we need 32.
    
    Let's use:
    - Ring 0: 1 point
    - Ring 1: 6 points
    - Ring 2: 12 points
    - Ring 3: 13 points (partial)
    Total: 32
    """
    points = set()
    
    # Ring 0: center
    points.add((0, 0))
    
    # Ring 1: 6 points at hexagonal distance 1
    # In axial coordinates, ring r has points where max(|a|,|b|,|a+b|) = r... 
    # Actually for triangular lattice with basis (1,0) and (1/2, sqrt(3)/2),
    # the 6 neighbors are as defined. Let me use a different approach.
    
    # Let me use a simpler approach: generate points in a hexagonal pattern
    # using the fact that for the triangular lattice with basis vectors e1=(1,0) and e2=(0,1)
    # in the (a,b) coordinate system, the 6 neighbors are the ones listed.
    
    # Hexagonal ring r consists of 6*r points for r >= 1.
    # Let me generate rings by iterating over all points with hexagonal distance exactly r.
    # Hexagonal distance in this coordinate system:
    # For point (a,b), the hexagonal distance from origin is max(|a|, |b|, |a+b|) ... 
    # Wait, that's for a different coordinate system. Let me think more carefully.
    
    # Actually, let me just use a simple spiral or filled hexagon approach.
    # A filled hexagon of "radius" r (in the sense of hexagonal distance) has:
    # 1 + 3*r*(r+1) points.
    # r=0: 1
    # r=1: 7
    # r=2: 19
    # r=3: 37
    # So we want 32 points, which is between r=2 (19) and r=3 (37).
    
    # Let me generate all points in the hexagon of radius 2 (19 points),
    # then add 13 more from the ring of radius 3.
    
    # For the triangular lattice with the given neighbor structure,
    # the hexagonal distance from (0,0) to (a,b) is:
    # d = (|a| + |b| + |a+b|) / 2  ... let me verify.
    # (1,0): (1+0+1)/2 = 1 ✓
    # (0,1): (0+1+1)/2 = 1 ✓
    # (1,-1): (1+1+0)/2 = 1 ✓
    # (-1,0): (1+0+1)/2 = 1 ✓
    # (0,-1): (0+1+1)/2 = 1 ✓
    # (-1,1): (1+1+0)/2 = 1 ✓
    # (2,0): (2+0+2)/2 = 2 ✓
    # (1,1): (1+1+2)/2 = 2 ✓
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    # Generate all points with hex_dist <= 2
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) <= 2:
                points.add((a, b))
    
    # We have 19 points. Need 13 more from hex_dist == 3.
    # Generate all points with hex_dist == 3
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    # Sort ring3 in some order and take 13
    # To maximize edges, we should pick points that are adjacent to existing points
    # and also adjacent to each other.
    # Let's sort by how many neighbors they have in the current set
    ring3_sorted = sorted(ring3, key=lambda p: -sum(1 for n in _neighbors(p) if n in points))
    for p in ring3_sorted[:13]:
        points.add(p)
    
    return points

def _local_search(points, max_time):
    """
    Perform local search to maximize edge count.
    Try swapping points or moving points to improve edge count.
    """
    start_time = time.time()
    best_points = set(points)
    best_edges = _edge_count(best_points)
    
    random.seed(42)
    
    while time.time() - start_time < max_time:
        improved = False
        # Try all possible moves: remove one point, add another
        point_list = list(best_points)
        random.shuffle(point_list)
        
        for p_remove in point_list:
            if time.time() - start_time >= max_time:
                break
            
            # Try adding each possible new point
            # Candidate new points: neighbors of existing points, or points in a bounding box
            candidates = set()
            for p in best_points:
                for n in _neighbors(p):
                    if n not in best_points and abs(n[0]) <= 64 and abs(n[1]) <= 64:
                        candidates.add(n)
            
            # Also try some random points
            for _ in range(20):
                a = random.randint(-10, 10)
                b = random.randint(-10, 10)
                if (a, b) not in best_points:
                    candidates.add((a, b))
            
            best_candidate = None
            best_candidate_edges = best_edges - 1  # at least lose the edges of p_remove
            
            # Quick prune: compute edges lost by removing p_remove
            edges_lost = sum(1 for n in _neighbors(p_remove) if n in best_points)
            base_edges = best_edges - edges_lost
            
            for p_add in candidates:
                if p_add == p_remove:
                    continue
                
                # Compute edges gained by adding p_add
                edges_gained = sum(1 for n in _neighbors(p_add) if n in best_points and n != p_remove)
                
                new_edges = base_edges + edges_gained
                
                if new_edges > best_candidate_edges:
                    best_candidate_edges = new_edges
                    best_candidate = p_add
            
            if best_candidate is not None and best_candidate_edges > best_edges:
                best_points.discard(p_remove)
                best_points.add(best_candidate)
                best_edges = best_candidate_edges
                improved = True
                break  # restart the outer loop
        
        if not improved:
            # Try random restarts or perturbations
            # Perturb: randomly move 1-2 points
            perturb_points = random.sample(list(best_points), k=random.randint(1, 2))
            new_points = set(best_points)
            for p in perturb_points:
                new_points.discard(p)
                # Add a random neighbor
                candidates = [n for n in _neighbors(p) if n not in new_points and abs(n[0]) <= 64 and abs(n[1]) <= 64]
                if candidates:
                    new_points.add(random.choice(candidates))
            
            new_edges = _edge_count(new_points)
            if new_edges >= best_edges:  # accept equal or better to escape plateaus
                best_points = new_points
                best_edges = new_edges
    
    return best_points, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Generate initial configuration
    points = _hexagonal_initial()
    
    # Verify validity
    if not _valid(points):
        # Fallback: simple strip
        return [[a, b] for a in range(16) for b in range(2)]
    
    # Local search
    points, edges = _local_search(points, max_time=135)
    
    # Final check
    if not _valid(points):
        return [[a, b] for a in range(16) for b in range(2)]
    
    # Return as list of lists
    return [list(p) for p in points]
