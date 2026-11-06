"""
Program to construct a maximal set of integral points in general position.
Uses a precomputed 5-point set and attempts to find 6th point via search.
"""
from fractions import Fraction
import math
import time
import sys

def _is_square(n):
    """Check if non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist_int(p1, p2):
    """Squared distance between two integer points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_pairwise_rational(points, D):
    """Check all pairwise distances are rational (integer here for D=1)."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if D == 1:
                d2 = _sq_dist_int(points[i], points[j])
                if not _is_square(d2):
                    return False
            else:
                # For general D, distance^2 = (x1-x2)^2 + D*(y1-y2)^2
                # Need this to be a rational square
                pass
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if area2 == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    A1 = x1*x1 + y1*y1
                    A2 = x2*x2 + y2*y2
                    A3 = x3*x3 + y3*y3
                    A4 = x4*x4 + y4*y4
                    # Determinant of [[A,x,y,1],...]
                    det = (A1*(x2*y3 - x2*y4 - x3*y2 + x3*y4 + x4*y2 - x4*y3)
                           - x1*(A2*y3 - A2*y4 - A3*y2 + A3*y4 + A4*y2 - A4*y3)
                           + y1*(A2*x3 - A2*x4 - A3*x2 + A3*x4 + A4*x2 - A4*x3)
                           - (A2*x3*y4 - A2*x4*y3 - A3*x2*y4 + A3*x4*y2 + A4*x2*y3 - A4*x3*y2))
                    if det == 0:
                        return False
    return True

def _is_valid_set(points, D=1):
    """Full validity check."""
    if len(set(points)) != len(points):
        return False
    if not _check_pairwise_rational(points, D):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points):
        return False
    return True

def _can_add_point(base_points, new_point, D=1):
    """Check if new_point can be added to base_points."""
    # Check distinctness
    if new_point in base_points:
        return False
    # Check pairwise rational distances
    for p in base_points:
        if D == 1:
            d2 = _sq_dist_int(p, new_point)
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(len(base_points)):
        for j in range(i + 1, len(base_points)):
            x1, y1 = base_points[i]
            x2, y2 = base_points[j]
            x3, y3 = new_point
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if area2 == 0:
                return False
    # Check no four concyclic
    for i in range(len(base_points)):
        for j in range(i + 1, len(base_points)):
            for k in range(j + 1, len(base_points)):
                x1, y1 = base_points[i]
                x2, y2 = base_points[j]
                x3, y3 = base_points[k]
                x4, y4 = new_point
                A1 = x1*x1 + y1*y1
                A2 = x2*x2 + y2*y2
                A3 = x3*x3 + y3*y3
                A4 = x4*x4 + y4*y4
                det = (A1*(x2*y3 - x2*y4 - x3*y2 + x3*y4 + x4*y2 - x4*y3)
                       - x1*(A2*y3 - A2*y4 - A3*y2 + A3*y4 + A4*y2 - A4*y3)
                       + y1*(A2*x3 - A2*x4 - A3*x2 + A3*x4 + A4*x2 - A4*x3)
                       - (A2*x3*y4 - A2*x4*y3 - A3*x2*y4 + A3*x4*y2 + A4*x2*y3 - A4*x3*y2))
                if det == 0:
                    return False
    return True

def construct():
    """
    Construct a maximal set of integral points in general position.
    Returns 5 known points, or 6 if a 6th can be found.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave 10s margin for evaluator's 150s kill
    
    # Known optimal 5-point integral point set in general position
    base_points = [
        (0, 0),
        (-56, 33),
        (-16, 63),
        (0, 33),
        (16, 63)
    ]
    
    # Verify base set is valid
    if not _is_valid_set(base_points, D=1):
        # Fallback: return a trivial 3-point set
        return {'D': 1, 'points': [[[0,1],[0,1]], [[3,1],[0,1]], [[0,1],[4,1]]]}
    
    best_points = base_points
    
    # Try to find a 6th point by searching in a bounded region
    # The base points span roughly x: [-56, 16], y: [0, 63]
    # Search a larger region around them
    search_radius = 200
    
    # Precompute the set for fast lookup
    base_set = set(base_points)
    
    # Search for 6th point
    found_sixth = False
    for x in range(-search_radius, search_radius + 1):
        if time.time() > deadline:
            break
        for y in range(-search_radius, search_radius + 1):
            if time.time() > deadline:
                break
            new_pt = (x, y)
            if new_pt in base_set:
                continue
            # Quick check: is it a Pythagorean distance to all base points?
            valid = True
            for bp in base_points:
                d2 = (x - bp[0])**2 + (y - bp[1])**2
                if not _is_square(d2):
                    valid = False
                    break
            if not valid:
                continue
            # Check collinearity and concyclicity
            if _can_add_point(base_points, new_pt, D=1):
                best_points = base_points + [new_pt]
                found_sixth = True
                break
        if found_sixth or time.time() > deadline:
            break
    
    # If we found a 6th point, verify the full set
    if found_sixth:
        if _is_valid_set(best_points, D=1):
            # Format and return 6 points
            formatted_points = []
            for x, y in best_points:
                formatted_points.append([[x, 1], [y, 1]])
            return {'D': 1, 'points': formatted_points}
        else:
            # Shouldn't happen, but fallback to 5
            best_points = base_points
    
    # Format output for 5 points
    formatted_points = []
    for x, y in best_points:
        formatted_points.append([[x, 1], [y, 1]])
    
    return {'D': 1, 'points': formatted_points}
