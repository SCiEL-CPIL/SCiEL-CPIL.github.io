import time
import random
import itertools
import numpy as np
from collections import Counter

_START = time.time()
_DEADLINE = 78.0  # safety margin before 90s kill

def _det(p1, p2, p3):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score_np(pts_array):
    """Vectorized score: largest multiplicity of a single positive doubled area."""
    n = pts_array.shape[0]
    if n < 3:
        return 0
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx = pts_array[j, 0] - pts_array[i, 0]
            dy = pts_array[j, 1] - pts_array[i, 1]
            dets = np.abs(dx * (pts_array[j+1:, 1] - pts_array[i, 1]) - 
                          dy * (pts_array[j+1:, 0] - pts_array[i, 0]))
            pos = dets[dets > 0]
            if len(pos) > 0:
                areas.append(pos)
    if not areas:
        return 0
    all_areas = np.concatenate(areas)
    if len(all_areas) == 0:
        return 0
    vals, counts = np.unique(all_areas, return_counts=True)
    return int(counts.max())

def _score(points):
    arr = np.array(points, dtype=np.int32)
    return _score_np(arr)

def _max_clique_branch_and_bound(adj, n, time_limit):
    """Bron-Kerbosch with pivot for maximum clique.
    adj: list of bitmasks (int), adj[i] has bit j set if edge i-j exists.
    Returns the maximum clique found before time_limit.
    """
    best_clique = []
    best_size = 0
    
    def expand(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > time_limit:
            return
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        if len(R) + len(P) <= best_size:
            return
        
        # Choose pivot u from P ∪ X with max degree in P
        PX = P | X
        if PX == 0:
            return
        # Pick pivot with most neighbors in P
        max_deg = -1
        pivot = -1
        # Sample a few candidates for speed
        candidates = []
        temp = PX
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            candidates.append(idx)
            temp ^= lb
            if len(candidates) >= 20:
                break
        for u in candidates:
            deg = bin(adj[u] & P).count('1')
            if deg > max_deg:
                max_deg = deg
                pivot = u
        if pivot == -1:
            return
        
        # Branch on vertices in P \ N(pivot)
        candidates = P & ~adj[pivot]
        while candidates:
            if time.time() > time_limit:
                return
            lb = candidates & (-candidates)
            v = lb.bit_length() - 1
            candidates ^= lb
            R.append(v)
            expand(R, P & adj[v], X & adj[v])
            R.pop()
            P ^= lb
            X |= lb
    
    expand([], (1 << n) - 1, 0)
    return best_clique

def _build_compatibility_graph(target_area, all_points, n_points=30):
    """Build graph where edge (i,j) exists if there's a point k such that
    det(i,j,k) == target_area. We want to find a clique of size n_points."""
    n = len(all_points)
    # For each pair (i,j), check if any k gives target_area
    # This is expensive for large n, so we limit
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            # Check if there exists k != i,j with det(i,j,k) == target_area
            found = False
            for k in range(n):
                if k == i or k == j:
                    continue
                if _det(all_points[i], all_points[j], all_points[k]) == target_area:
                    found = True
                    break
            if found:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _hexagonal_lattice_points(center_x, center_y, radius, n_points=30):
    """Generate points on a hexagonal/triangular lattice."""
    pts = set()
    # Triangular lattice basis: (1,0) and (1/2, sqrt(3)/2) ~ (1, 1.732)
    # For integer coords, use basis (2,0) and (1,1) or similar
    # Actually, let's use a proper triangular lattice approximation
    # Basis vectors: a = (2, 0), b = (1, 2) gives det=4
    # Or use (1,0), (0,1) for square, but hex is better
    # For hex: use points where x + y is even (checkerboard) on a grid
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            x = center_x + dx
            y = center_y + dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                # For triangular lattice feel, keep points where (x+y) % 2 == 0
                # or just use all points in a diamond shape
                if abs(dx) + abs(dy) <= radius:
                    pts.add((x, y))
    if len(pts) < n_points:
        # Fill with nearby points
        all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        random.shuffle(all_pts)
        for p in all_pts:
            if len(pts) >= n_points:
                break
            pts.add(p)
    return list(pts)[:n_points]

def construct():
    global _START
    start = time.time()
    deadline = start + 78.0
    
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Structured constructions (fast baseline)
    grid_constructions = []
    grid_constructions.append([(x, y) for x in range(-3, 3) for y in range(-2, 3)])
    grid_constructions.append([(x, y) for x in range(-2, 3) for y in range(-3, 3)])
    grid_constructions.append([(x, y) for x in range(-4, 6) for y in range(-1, 2)])
    grid_constructions.append([(x, y) for x in range(-1, 2) for y in range(-4, 6)])
    grid_constructions.append([(x, y) for x in range(-3, 4) for y in range(-2, 3)][:30])
    grid_constructions.append([(x, y) for x in range(-3, 5) for y in range(-2, 2)][:30])
    grid_constructions.append([(x, y) for x in range(-7, 8) for y in range(0, 2)][:30])
    grid_constructions.append([(x, y) for x in range(-9, 11) for y in range(0, 2)][:30])
    grid_constructions.append([(x, y) for x in range(-5, 7) for y in range(-1, 2)][:30])
    grid_constructions.append([(x, y) for x in range(-6, 7) for y in range(-1, 2)][:30])
    grid_constructions.append([(x, y) for x in range(-12, 13) for y in range(0, 2)][:30])
    grid_constructions.append([(x, y) for x in range(-14, 16) for y in range(0, 2)][:30])
    
    for cons in grid_constructions:
        if time.time() > deadline - 5:
            break
        if len(cons) >= 30:
            pts = cons[:30]
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                pts = list(set(pts))
                if len(pts) == 30:
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
    
    # Strategy 2: Maximum clique search on compatibility graph
    # For a given target area, find 30 points that are mutually "compatible"
    # (each pair has some third point completing a triangle of that area)
    
    # First, identify promising target areas by sampling
    area_counts = Counter()
    for _ in range(50000):
        i = random.randint(0, len(all_points)-1)
        j = random.randint(0, len(all_points)-1)
        k = random.randint(0, len(all_points)-1)
        if i == j or j == k or i == k:
            continue
        d = _det(all_points[i], all_points[j], all_points[k])
        if d > 0 and d <= 50:
            area_counts[d] += 1
    
    if area_counts:
        top_areas = [a for a, c in area_counts.most_common(15) if c > 0]
    else:
        top_areas = list(range(2, 30, 2))
    
    # For each target area, build a subgraph and find max clique
    for target_area in top_areas[:8]:
        if time.time() > deadline - 10:
            break
        
        # Build a smaller candidate set: points that participate in at least
        # one triangle with the target area
        participating = set()
        # Sample to find participating points
        for _ in range(20000):
            i = random.randint(0, len(all_points)-1)
            j = random.randint(0, len(all_points)-1)
            k = random.randint(0, len(all_points)-1)
            if i == j or j == k or i == k:
                continue
            if _det(all_points[i], all_points[j], all_points[k]) == target_area:
                participating.add(all_points[i])
                participating.add(all_points[j])
                participating.add(all_points[k])
        
        if len(participating) < 30:
            continue
        
        # Limit to a manageable subset
        candidate_pts = list(participating)
        random.shuffle(candidate_pts)
        candidate_pts = candidate_pts[:100]  # Limit for clique search
        
        if len(candidate_pts) < 30:
            continue
        
        # Build adjacency: edge (i,j) if exists k with det(i,j,k) == target_area
        n_cand = len(candidate_pts)
        adj = [0] * n_cand
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                found = False
                for k in range(n_cand):
                    if k == i or k == j:
                        continue
                    if _det(candidate_pts[i], candidate_pts[j], candidate_pts[k]) == target_area:
                        found = True
                        break
                if found:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Find maximum clique
        clique = _max_clique_branch_and_bound(adj, n_cand, time.time() + 5)
        if len(clique) >= 30:
            pts = [candidate_pts[i] for i in clique[:30]]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        
        # Also try greedy selection from participating points
        if time.time() < deadline - 8:
            selected = []
            remaining = list(participating)
            random.shuffle(remaining)
            # Greedy: pick point that maximizes triangles with target_area
            while len(selected) < 30 and remaining:
                best_pt = None
                best_gain = -1
                for pt in remaining[:50]:  # Limit for speed
                    gain = 0
                    for i in range(len(selected)):
                        for j in range(i+1, len(selected)):
                            if _det(pt, selected[i], selected[j]) == target_area:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = pt
                if best_pt is None:
                    best_pt = remaining[0]
                selected.append(best_pt)
                remaining.remove(best_pt)
            
            if len(selected) == 30:
                s = _score(selected)
                if s > best_score:
                    best_score = s
                    best_points = selected
    
    # Strategy 3: Hexagonal lattice constructions
    for cx in range(-5, 6, 3):
        for cy in range(-5, 6, 3):
            if time.time() > deadline - 5:
                break
            for radius in [4, 5, 6]:
                pts = _hexagonal_lattice_points(cx, cy, radius, 30)
                if len(pts) == 30:
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
    
    # Strategy 4: Local search from best
    if best_points and time.time() < deadline - 5:
        current = list(best_points)
        current_score = best_score
        selected_set = set(current)
        all_points_list = all_points
        
        for it in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(30):
                if time.time() > deadline - 3:
                    break
                candidates = [p for p in all_points_list if p not in selected_set]
                if not candidates:
                    break
                random.shuffle(candidates)
                max_cand = min(len(candidates), 100)
                for j in range(max_cand):
                    new_pt = candidates[j]
                    new_pts = current[:]
                    new_pts[i] = new_pt
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current = new_pts
                        current_score = new_score
                        selected_set = set(current)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_points = current
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    best_points = list(set(best_points))[:30]
    while len(best_points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in best_points:
            best_points.append((x, y))
    
    return best_points
