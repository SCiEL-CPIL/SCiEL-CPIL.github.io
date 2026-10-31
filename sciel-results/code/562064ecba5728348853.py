"""Heilbronn convex regions - support-line polygon parameterization."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 13
_DEADLINE = 150.0
_START = time.time()

# Fixed normal angles for the 13 support lines (evenly spaced)
_ANGLES = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
_COS = np.cos(_ANGLES)
_SIN = np.sin(_ANGLES)
_COS2 = _COS ** 2
_SIN2 = _SIN ** 2
_COS_SIN = _COS * _SIN

# Precompute all triples for triangle area computation
_TRIPLES = np.array(list(combinations(range(N), 3)))
_T_I = _TRIPLES[:, 0]
_T_J = _TRIPLES[:, 1]
_T_K = _TRIPLES[:, 2]


def _polygon_from_offsets(offsets):
    """Compute polygon vertices from 13 support line offsets.
    
    Each support line: cos(a_i)*x + sin(a_i)*y = offsets[i]
    The polygon is the intersection of half-planes cos(a_i)*x + sin(a_i)*y <= offsets[i]
    Vertices are intersections of consecutive support lines.
    """
    n = len(offsets)
    vertices = np.zeros((n, 2))
    for i in range(n):
        a1 = i
        a2 = (i + 1) % n
        # Intersection of lines:
        # cos(a1)*x + sin(a1)*y = offsets[a1]
        # cos(a2)*x + sin(a2)*y = offsets[a2]
        c1, s1 = _COS[a1], _SIN[a1]
        c2, s2 = _COS[a2], _SIN[a2]
        det = c1 * s2 - c2 * s1
        if abs(det) < 1e-12:
            vertices[i] = [0.0, 0.0]
            continue
        x = (offsets[a1] * s2 - offsets[a2] * s1) / det
        y = (c1 * offsets[a2] - c2 * offsets[a1]) / det
        vertices[i] = [x, y]
    return vertices


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area over all triples."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _convex_hull_area_vec(points):
    """Compute convex hull area using the polygon (assumed convex and ordered)."""
    # For a convex polygon with vertices in order, area = 0.5 * |sum of cross products|
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    # Shoelace formula
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * abs(np.sum(x * y_next - x_next * y))
    return area


def _objective(offsets):
    """Negative of the score (min triangle area / hull area) for minimization."""
    elapsed = time.time() - _START
    if elapsed > _DEADLINE - 2.0:
        return 1e6
    
    points = _polygon_from_offsets(offsets)
    
    # Check if polygon is valid (positive area, no degeneracy)
    hull_area = _convex_hull_area_vec(points)
    if hull_area < 1e-10:
        return 1e6
    
    # Check that all points satisfy the half-plane constraints (convexity)
    # For each point, check it's inside all half-planes
    for i in range(N):
        val = _COS[i] * points[:, 0] + _SIN[i] * points[:, 1]
        if np.max(val) > offsets[i] + 1e-8:
            # Point outside its own half-plane - invalid
            return 1e6 + (np.max(val) - offsets[i])
    
    min_area = _min_triangle_area_vec(points)
    
    if min_area < 0:
        return 1e6
    
    return -min_area / hull_area


def _is_valid(points):
    """Check if points form a valid configuration."""
    if points.shape != (N, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    hull_area = _convex_hull_area_vec(points)
    if hull_area < 1e-10:
        return False
    min_area = _min_triangle_area_vec(points)
    return min_area > 0


def construct():
    global _START
    _START = time.time()
    deadline = _START + _DEADLINE - 5.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial guess: regular 13-gon (all offsets equal)
    initial_offsets = np.ones(N) * 0.5
    
    # Try multiple initializations with Nelder-Mead
    rng = np.random.default_rng(42)
    
    n_restarts = 25
    max_iters = 300
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            offsets0 = initial_offsets.copy()
        elif restart == 1:
            # Slightly perturbed regular polygon
            offsets0 = initial_offsets + rng.normal(0, 0.05, N)
        else:
            # More diverse initializations
            scale = 0.3 + 0.4 * rng.random()
            offsets0 = scale * (1.0 + 0.2 * rng.standard_normal(N))
            # Ensure positive offsets
            offsets0 = np.maximum(offsets0, 0.1)
        
        # Normalize offsets so the polygon has reasonable size
        offsets0 = offsets0 / np.mean(offsets0) * 0.5
        
        try:
            result = minimize(
                _objective,
                offsets0,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iters,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
        except Exception:
            continue
        
        if result.success or result.fun < 0:
            points = _polygon_from_offsets(result.x)
            if _is_valid(points):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
        
        # Also check the initial point
        points0 = _polygon_from_offsets(offsets0)
        if _is_valid(points0):
            hull_area = _convex_hull_area_vec(points0)
            min_area = _min_triangle_area_vec(points0)
            score0 = min_area / hull_area
            if score0 > best_score:
                best_score = score0
                best_points = points0.copy()
    
    # If no valid result, fall back to regular polygon
    if best_points is None:
        best_points = _polygon_from_offsets(np.ones(N) * 0.5)
    
    # Final validation
    if not _is_valid(best_points):
        # Ultimate fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
