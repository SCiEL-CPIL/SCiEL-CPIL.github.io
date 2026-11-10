import numpy as np
from math import gcd

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2:
        return None
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    return _normalize_line(A, B, C)

def _count_incidence(points, A, B, C):
    return np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)

def _pick_top_k_lines(points, k):
    N = len(points)
    candidate_set = {}
    for i in range(N):
        for j in range(i + 1, N):
            norm = _line_from_points(points[i], points[j])
            if norm is None:
                continue
            candidate_set[norm] = True
            
    incidences = {}
    for norm in candidate_set:
        A, B, C = norm
        incidences[norm] = _count_incidence(points, A, B, C)
        
    sorted_lines = sorted(candidate_set.keys(), key=lambda n: (-incidences[n], n))
    chosen = sorted_lines[:k]
    
    if len(chosen) < k:
        for i in range(N):
            for j in range(i + 1, N):
                if len(chosen) >= k:
                    break
                norm = _line_from_points(points[i], points[j])
                if norm is not None and norm not in chosen:
                    chosen.append(norm)
                    
    return np.array(chosen, dtype=np.int64)

def _count_total_incidence(points, lines):
    total = 0
    for A, B, C in lines:
        total += _count_incidence(points, A, B, C)
    return total

def _generate_grid_points(n, m):
    xs = np.arange(n, dtype=np.int64)
    ys = np.arange(m, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _generate_triangular_points(N):
    # Generate a triangular lattice of roughly N points
    # k(k+1)/2 <= N
    k = int((np.sqrt(1 + 8*N) - 1) / 2)
    points = []
    for i in range(k+1):
        for j in range(k+1-i):
            points.append((i, j))
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    return np.array(points[:N], dtype=np.int64)

def _generate_perturbed_grid(n, m, seed=42):
    np.random.seed(seed)
    points = _generate_grid_points(n, m)
    # Add small random perturbations? 
    # This might break integer constraints or reduce collinearity.
    # Better to try different grid spacings?
    # Let's just return the standard grid for now, this function is a placeholder for other constructions.
    return points

def szemeredi_trotter_incidences():
    candidates = [
        _generate_grid_points(6, 9),
        _generate_grid_points(9, 6),
        _generate_grid_points(3, 18),
        _generate_grid_points(18, 3),
        _generate_grid_points(2, 27),
        _generate_grid_points(27, 2),
        _generate_grid_points(1, 54),
        _generate_grid_points(54, 1),
        _generate_triangular_points(54),
    ]
    
    best_incidence = -1
    best_points = None
    best_lines = None
    
    for points in candidates:
        if len(points) != 54:
            continue
        # Ensure unique points
        points = np.unique(points, axis=0)
        if len(points) != 54:
            continue
            
        lines = _pick_top_k_lines(points, 27)
        inc = _count_total_incidence(points, lines)
        if inc > best_incidence:
            best_incidence = inc
            best_points = points
            best_lines = lines
            
    points = np.round(best_points).astype(np.int64)
    lines = np.round(best_lines).astype(np.int64)
    return points, lines
