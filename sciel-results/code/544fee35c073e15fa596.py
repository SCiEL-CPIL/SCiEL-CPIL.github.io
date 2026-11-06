"""Combined: vectorized evaluation + lattice candidates + two-line insight + targeted local search."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    # Precompute triple indices for n=30
    triple_indices = np.array([(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)], dtype=np.int64)
    I, J, K = triple_indices[:, 0], triple_indices[:, 1], triple_indices[:, 2]
    N_TRIPLES = len(I)

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Compute doubled area (determinant) for all triples
        det = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
        det = np.abs(det)
        det = det[det > 0]
        
        if len(det) == 0:
            return 0
        
        vals, cnts = np.unique(det, return_counts=True)
        return int(cnts.max())

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    def make_grid(a, b, ox=0, oy=0, scale=1):
        """Create a x b grid of points scaled by 'scale' and offset by (ox, oy)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * scale
                y = oy + j * scale
                pts.append((x, y))
        return pts

    def make_lattice(a, b, u1, u2, v1, v2, ox=0, oy=0):
        """Create a x b lattice with basis vectors (u1,u2) and (v1,v2)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * u1 + j * v1
                y = oy + i * u2 + j * v2
                pts.append((x, y))
        return pts

    # Start with the two-parallel-lines configuration (15 points on y=0, 15 on y=1)
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate_fast(best_points)

    # Generate diverse candidates
    candidates = []

    # Two-line configurations with varying separation
    for d in range(1, 6):
        pts = [(x, 0) for x in range(15)] + [(x, d) for x in range(15)]
        if valid(pts):
            candidates.append(pts)

    # Standard axis-aligned grids
    for a, b in [(5,6), (6,5), (3,10), (10,3), (2,15), (15,2)]:
        for ox in range(-15, 16, 3):
            for oy in range(-15, 16, 3):
                pts = make_grid(a, b, ox=ox, oy=oy)
                if valid(pts):
                    candidates.append(pts)

    # Scaled grids
    for a, b in [(5,6), (6,5), (3,10), (10,3)]:
        for scale in [1, 2, 3]:
            for ox in range(-10, 11, 2):
                for oy in range(-10, 11, 2):
                    pts = make_grid(a, b, ox=ox, oy=oy, scale=scale)
                    if valid(pts):
                        candidates.append(pts)

    # Sheared lattices
    for a, b in [(5,6), (6,5), (3,10), (10,3)]:
        for s in range(-3, 4):
            for ox in range(-10, 11, 2):
                for oy in range(-10, 11, 2):
                    pts = make_lattice(a, b, 1, 0, s, 1, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)

    # Rotated lattices with unimodular basis
    basis_vectors = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2),
        (3, 1), (3, -1), (1, 3), (1, -3), (2, 3), (2, -3), (3, 2), (3, -2)
    ]
    for a, b in [(5,6), (6,5), (3,10), (10,3)]:
        for i, (u1, u2) in enumerate(basis_vectors):
            for j, (v1, v2) in enumerate(basis_vectors[i+1:], i+1):
                det_val = u1 * v2 - u2 * v1
                if abs(det_val) != 1:
                    continue
                for ox in range(-8, 9, 2):
                    for oy in range(-8, 9, 2):
                        pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                        if valid(pts):
                            candidates.append(pts)

    # Evaluate all candidates
    for pts in candidates:
        if time.time() > deadline - 30:
            break
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    # Local search: hill-climbing with larger neighborhood
    random.seed(42)
    
    improved = True
    max_iterations = 100
    iteration = 0
    while improved and time.time() < deadline - 10 and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(30):
            if time.time() > deadline - 10:
                break
            old_x, old_y = best_points[idx]
            # Try moves in a larger neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate_fast(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Random restarts with perturbation
    if time.time() < deadline - 5:
        random.seed(123)
        for restart in range(200):
            if time.time() > deadline - 2:
                break
            
            if random.random() < 0.6:
                # Perturbation of best
                new_points = list(best_points)
                for _ in range(random.randint(1, 5)):
                    idx = random.randint(0, 29)
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    new_x = new_points[idx][0] + dx
                    new_y = new_points[idx][1] + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        new_points[idx] = (new_x, new_y)
            else:
                # Grid-like structure with noise
                w, h = random.choice([(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)])
                sx = random.randint(1, max(1, 30 // w))
                sy = random.randint(1, max(1, 30 // h))
                cx = random.randint(-5, 5)
                cy = random.randint(-5, 5)
                new_points = [(cx + i*sx + random.randint(-1,1), cy + j*sy + random.randint(-1,1)) 
                              for i in range(w) for j in range(h)]
            
            if not valid(new_points):
                continue
            score = evaluate_fast(new_points)
            if score > best_score:
                best_score = score
                best_points = new_points

    return best_points
# EVOLVE-BLOCK-END
