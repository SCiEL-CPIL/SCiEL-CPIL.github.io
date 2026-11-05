import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit

    n = 30

    def eval_points(pts):
        """Vectorized evaluation: returns (max_multiplicity, best_area)."""
        P = np.array(pts, dtype=np.int64)
        # Build pairwise difference matrix: P[i] - P[j] for all i,j
        # Use broadcasting: shape (n, n, 2)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]  # (n, n, 2)
        dx = diff[:, :, 0]  # (n, n)
        dy = diff[:, :, 1]  # (n, n)
        
        # For each triple (i, j, k) with i<j<k, det = (P[j]-P[i]) x (P[k]-P[i])
        # = (P[j]-P[i]).x * (P[k]-P[i]).y - (P[j]-P[i]).y * (P[k]-P[i]).x
        # Using diff: det(i,j,k) = dx[i,j]*dy[i,k] - dy[i,j]*dx[i,k]
        # But we need all triples i<j<k
        
        # More efficient: for each i, compute cross products of all pairs (j,k) with j,k > i
        # Actually, let's use the full matrix approach:
        # det[i,j,k] = (P[j]-P[i]) x (P[k]-P[i])
        # = dx[i,j]*dy[i,k] - dy[i,j]*dx[i,k]
        
        # Build (n,n,n) tensor would be 30^3=27000, manageable
        # det_tensor[i,j,k] = dx[i,j]*dy[i,k] - dy[i,j]*dx[i,k]
        # But we only need i<j<k
        
        # Alternative: for each i, compute cross products between all other pairs
        areas = []
        for i in range(n):
            # Vectors from i to all other points
            v = P - P[i]  # (n, 2)
            # Cross product v[j] x v[k] for all j,k
            cross = v[:, 0] * v[:, 1]  # not quite
            # cross(j,k) = v[j,0]*v[k,1] - v[j,1]*v[k,0]
            cross_mat = np.outer(v[:, 0], v[:, 1]) - np.outer(v[:, 1], v[:, 0])  # (n, n)
            # We want j > i, k > i, j < k
            # Get upper triangle excluding row/col i
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d = abs(cross_mat[j, k])
                    if d > 0:
                        areas.append(d)
        
        if not areas:
            return 0, 1
        
        counter = Counter(areas)
        max_mult = max(counter.values())
        # Find the area with max multiplicity
        best_area = max(counter, key=lambda a: (counter[a], -a))
        return max_mult, best_area

    def eval_fast(pts):
        """Faster vectorized eval using numpy operations."""
        P = np.array(pts, dtype=np.int64)
        areas_list = []
        for i in range(n):
            vi = P[i]
            others = np.delete(P, i, axis=0)
            # Compute all cross products among 'others' relative to vi
            # For points a, b in others: det = (a-vi) x (b-vi)
            # = (a[0]-vi[0])*(b[1]-vi[1]) - (a[1]-vi[1])*(b[0]-vi[0])
            ox = others[:, 0] - vi[0]
            oy = others[:, 1] - vi[1]
            # cross matrix
            cross = np.outer(ox, oy) - np.outer(oy, ox)
            # Upper triangle
            m = len(others)
            for j in range(m):
                for k in range(j + 1, m):
                    d = abs(cross[j, k])
                    if d > 0:
                        areas_list.append(d)
        
        if not areas_list:
            return 0, 1
        counter = Counter(areas_list)
        max_mult = max(counter.values())
        best_area = max(counter, key=lambda a: (counter[a], -a))
        return max_mult, best_area

    def eval_vectorized(pts):
        """Fully vectorized: compute all 4060 determinants at once."""
        P = np.array(pts, dtype=np.int64)
        # For each i, compute cross products of all pairs (j,k) with j,k != i
        all_areas = []
        for i in range(n):
            vi = P[i]
            others = np.delete(P, i, axis=0)
            ox = others[:, 0] - vi[0]
            oy = others[:, 1] - vi[1]
            # cross[j,k] = ox[j]*oy[k] - oy[j]*ox[k]
            cross = np.outer(ox, oy) - np.outer(oy, ox)
            # Get upper triangle indices
            m = len(others)
            rows, cols = np.triu_indices(m, k=1)
            vals = np.abs(cross[rows, cols])
            all_areas.append(vals)
        
        all_areas = np.concatenate(all_areas)
        all_areas = all_areas[all_areas > 0]
        
        if len(all_areas) == 0:
            return 0, 1
        
        # Count occurrences
        unique, counts = np.unique(all_areas, return_counts=True)
        max_idx = np.argmax(counts)
        return int(counts[max_idx]), int(unique[max_idx])

    # Generate initial candidates
    random.seed(42)
    candidates = []
    
    # Grid
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30:
            candidates.append(pts)
    
    # Sheared grids
    for w, h in [(6, 5), (5, 6)]:
        for k in [1, 2, 3]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid:
                    break
            if valid and len(pts) == 30:
                candidates.append(pts)
    
    # Random
    for _ in range(30):
        pts = set()
        while len(pts) < 30:
            pts.add((random.randint(-20, 20), random.randint(-20, 20)))
        candidates.append(list(pts))

    # Evaluate candidates
    best_score = 0
    best_points = None
    best_area = 1
    
    for pts in candidates:
        if time.time() > deadline - 30:
            break
        score, area = eval_vectorized(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
            best_area = area

    if best_points is None:
        best_points = [(0, 0)] * 30
        best_score = 0
        best_area = 1

    # Simulated annealing
    current = list(best_points)
    current_score = best_score
    current_area = best_area
    
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling = 0.9995
    
    no_improve = 0
    max_no_improve = 500
    
    step = 0
    while time.time() < deadline - 2:
        step += 1
        no_improve += 1
        
        idx = random.randint(0, n - 1)
        old_pos = current[idx]
        
        # Generate new position: small perturbation or larger jump
        if random.random() < 0.3:
            # Larger jump
            new_pos = (random.randint(-30, 30), random.randint(-30, 30))
        else:
            # Small perturbation
            radius = random.randint(1, 5)
            dx = random.randint(-radius, radius)
            dy = random.randint(-radius, radius)
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        
        if not (-30 <= new_pos[0] <= 30 and -30 <= new_pos[1] <= 30):
            continue
        if new_pos in current:
            continue
        
        # Try the move
        current[idx] = new_pos
        new_score, new_area = eval_vectorized(current)
        
        # Acceptance criterion
        if new_score > current_score:
            # Always accept improvement
            current_score = new_score
            current_area = new_area
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current)
                best_area = current_area
        else:
            # Metropolis acceptance
            delta = current_score - new_score  # positive means worse
            if delta == 0:
                # Same score, accept with some probability to explore
                if random.random() < 0.5:
                    pass  # accept
                else:
                    current[idx] = old_pos
            else:
                # delta > 0, probability = exp(-delta/T)
                prob = np.exp(-delta / T) if T > T_min else 0
                if random.random() < prob:
                    current_score = new_score
                    current_area = new_area
                else:
                    current[idx] = old_pos
        
        # Cool down
        T = max(T * cooling, T_min)
        
        if no_improve > max_no_improve:
            # Reset to best and try a different strategy
            current = list(best_points)
            current_score = best_score
            current_area = best_area
            no_improve = 0
            T = 5.0  # Reheat
    
    # Final evaluation
    final_score, final_area = eval_vectorized(best_points)
    if final_score > best_score:
        best_score = final_score
        best_area = final_area

    return best_points


def _calculate_max_multiplicity(points):
    """Calculate the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0, 0
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
    if not areas:
        return 0, 0
    max_mult = max(areas.values())
    max_area = [a for a, m in areas.items() if m == max_mult][0]
    return max_mult, max_area
