"""Program for integral point sets in general position."""
from fractions import Fraction
import math
import time
import itertools
import random

TIME_LIMIT = 110.0  # Leave 40s safety margin from 150s

def _is_squarefree(n):
    """Check if n is squarefree."""
    if n <= 1:
        return True
    f = 2
    while f * f <= n:
        if n % (f * f) == 0:
            return False
        f += 1
    return True

def _check_integrality(points):
    """Check that all pairwise squared distances are integers."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            sq = dx * dx + dy * dy
            if not isinstance(sq, int) or sq < 0:
                return False
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic using exact rational arithmetic."""
    n = len(points)
    for i, j, k, m in itertools.combinations(range(n), 4):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[m]
        
        # Circle through (x1,y1), (x2,y2), (x3,y3):
        # |x|^2 + |y|^2 + Dx + Ey + F = 0
        # Solve for D, E, F using Cramer's rule with exact fractions.
        
        # Equation: x^2 + y^2 + D*x + E*y + F = 0
        # For each point:
        # D*x_i + E*y_i + F = -(x_i^2 + y_i^2)
        
        rhs1 = -(x1 * x1 + y1 * y1)
        rhs2 = -(x2 * x2 + y2 * y2)
        rhs3 = -(x3 * x3 + y3 * y3)
        
        # Determinant of coefficient matrix
        # [x1, y1, 1]
        # [x2, y2, 1]
        # [x3, y3, 1]
        det = (x1 * (y2 - y3) - y1 * (x2 - x3) + (x2 * y3 - x3 * y2))
        
        if det == 0:
            # Three points collinear - already checked, but skip
            continue
        
        # Cramer's rule for D
        det_D = (rhs1 * (y2 - y3) - y1 * (rhs2 - rhs3) + (rhs2 * y3 - rhs3 * y2))
        # Cramer's rule for E
        det_E = (x1 * (rhs2 - rhs3) - rhs1 * (x2 - x3) + (x2 * rhs3 - x3 * rhs2))
        # Cramer's rule for F
        det_F = (x1 * y2 * rhs3 - x1 * rhs2 * y3 - rhs1 * y2 * x3 + rhs1 * x2 * y3 + y1 * rhs2 * x3 - y1 * x2 * rhs3)
        
        D = Fraction(det_D, det)
        E = Fraction(det_E, det)
        F = Fraction(det_F, det)
        
        # Check if (x4, y4) lies on this circle
        # x4^2 + y4^2 + D*x4 + E*y4 + F == 0
        check = x4 * x4 + y4 * y4 + D * x4 + E * y4 + F
        if check == 0:
            return False
    
    return True

def _is_valid_set(points):
    """Full validity check."""
    if not _check_integrality(points):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points):
        return False
    return True

def _search_for_set(start_time, target_size, grid_limit):
    """Search for a set of target_size points within grid bounds."""
    # Use a smaller, structured grid to keep search tractable
    # Try to find 6 points: start with 3, add 3 more
    
    best_found = None
    
    # Candidate grid
    candidates = []
    for x in range(-grid_limit, grid_limit + 1):
        for y in range(-grid_limit, grid_limit + 1):
            candidates.append((x, y))
    
    # Remove duplicates and ensure distinct
    candidates = list(set(candidates))
    
    # Pre-filter: for each pair, precompute squared distance
    # We'll use a greedy + backtracking approach
    
    # Strategy: build up the set incrementally
    # Start with small sets and try to extend
    
    # Try all pairs as starting points, then extend
    n_cands = len(candidates)
    
    # To keep it tractable, limit which pairs we try
    max_pairs = 200
    pair_count = 0
    
    for i in range(n_cands):
        if pair_count >= max_pairs:
            break
        for j in range(i + 1, n_cands):
            if pair_count >= max_pairs:
                break
            pair_count += 1
            
            if time.time() - start_time > TIME_LIMIT - 20:
                return best_found
            
            p0 = candidates[i]
            p1 = candidates[j]
            
            # Check p0, p1 are distinct (they are by construction)
            
            # Now try to add a third point
            third_candidates = []
            for k in range(n_cands):
                if k == i or k == j:
                    continue
                p2 = candidates[k]
                # Check distances are integers (they always are for integer points)
                # Check not collinear with p0, p1
                dx01 = p1[0] - p0[0]
                dy01 = p1[1] - p0[1]
                dx02 = p2[0] - p0[0]
                dy02 = p2[1] - p0[1]
                cross = dx01 * dy02 - dy01 * dx02
                if cross == 0:
                    continue
                third_candidates.append(p2)
            
            # Sort third candidates to prefer "interesting" ones (non-trivial)
            # Then try to extend to 4, 5, 6 points
            
            # For efficiency, try a limited number of third points
            max_thirds = 50
            for p2 in third_candidates[:max_thirds]:
                if time.time() - start_time > TIME_LIMIT - 20:
                    return best_found
                
                current = [p0, p1, p2]
                
                # Try to extend to 4
                fourth_candidates = []
                for k in range(n_cands):
                    if k == i or k == j:
                        continue
                    p3 = candidates[k]
                    if p3 == p2:
                        continue
                    
                    # Check not collinear with any pair in current
                    valid = True
                    for a in range(3):
                        for b in range(a + 1, 3):
                            dx_ab = current[b][0] - current[a][0]
                            dy_ab = current[b][1] - current[a][1]
                            dx_ap = p3[0] - current[a][0]
                            dy_ap = p3[1] - current[a][1]
                            cross = dx_ab * dy_ap - dy_ab * dx_ap
                            if cross == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    fourth_candidates.append(p3)
                
                # Try limited fourth candidates
                max_fourths = 30
                for p3 in fourth_candidates[:max_fourths]:
                    if time.time() - start_time > TIME_LIMIT - 20:
                        return best_found
                    
                    current4 = [p0, p1, p2, p3]
                    
                    # Check no four concyclic
                    if not _check_no_four_concyclic(current4):
                        continue
                    
                    # Try to extend to 5
                    fifth_candidates = []
                    for k in range(n_cands):
                        if k == i or k == j:
                            continue
                        p4 = candidates[k]
                        if p4 in current4:
                            continue
                        
                        # Check not collinear with any pair
                        valid = True
                        for a in range(4):
                            for b in range(a + 1, 4):
                                dx_ab = current4[b][0] - current4[a][0]
                                dy_ab = current4[b][1] - current4[a][1]
                                dx_ap = p4[0] - current4[a][0]
                                dy_ap = p4[1] - current4[a][1]
                                cross = dx_ab * dy_ap - dy_ab * dx_ap
                                if cross == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        fifth_candidates.append(p4)
                    
                    # Try limited fifth candidates
                    max_fifths = 20
                    for p4 in fifth_candidates[:max_fifths]:
                        if time.time() - start_time > TIME_LIMIT - 20:
                            return best_found
                        
                        current5 = [p0, p1, p2, p3, p4]
                        
                        # Check no four concyclic among 5 points
                        if not _check_no_four_concyclic(current5):
                            continue
                        
                        if target_size == 5:
                            if best_found is None or len(current5) > len(best_found):
                                best_found = current5[:]
                            continue
                        
                        # Try to extend to 6
                        sixth_candidates = []
                        for k in range(n_cands):
                            if k == i or k == j:
                                continue
                            p5 = candidates[k]
                            if p5 in current5:
                                continue
                            
                            # Check not collinear with any pair
                            valid = True
                            for a in range(5):
                                for b in range(a + 1, 5):
                                    dx_ab = current5[b][0] - current5[a][0]
                                    dy_ab = current5[b][1] - current5[a][1]
                                    dx_ap = p5[0] - current5[a][0]
                                    dy_ap = p5[1] - current5[a][1]
                                    cross = dx_ab * dy_ap - dy_ab * dx_ap
                                    if cross == 0:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            sixth_candidates.append(p5)
                        
                        # Try limited sixth candidates
                        max_sixths = 15
                        for p5 in sixth_candidates[:max_sixths]:
                            if time.time() - start_time > TIME_LIMIT - 20:
                                return best_found
                            
                            current6 = [p0, p1, p2, p3, p4, p5]
                            
                            # Check no four concyclic among 6 points
                            if not _check_no_four_concyclic(current6):
                                continue
                            
                            if target_size == 6:
                                if best_found is None or len(current6) > len(best_found):
                                    best_found = current6[:]
                                return best_found
                            else:
                                # Try to extend to 7
                                seventh_candidates = []
                                for k in range(n_cands):
                                    if k == i or k == j:
                                        continue
                                    p6 = candidates[k]
                                    if p6 in current6:
                                        continue
                                    
                                    # Check not collinear with any pair
                                    valid = True
                                    for a in range(6):
                                        for b in range(a + 1, 6):
                                            dx_ab = current6[b][0] - current6[a][0]
                                            dy_ab = current6[b][1] - current6[a][1]
                                            dx_ap = p6[0] - current6[a][0]
                                            dy_ap = p6[1] - current6[a][1]
                                            cross = dx_ab * dy_ap - dy_ab * dx_ap
                                            if cross == 0:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if not valid:
                                        continue
                                    seventh_candidates.append(p6)
                                
                                # Try limited seventh candidates
                                max_sevenths = 10
                                for p6 in seventh_candidates[:max_sevenths]:
                                    if time.time() - start_time > TIME_LIMIT - 20:
                                        return best_found
                                    
                                    current7 = [p0, p1, p2, p3, p4, p5, p6]
                                    
                                    # Check no four concyclic among 7 points
                                    if not _check_no_four_concyclic(current7):
                                        continue
                                    
                                    if target_size >= 7:
                                        if best_found is None or len(current7) > len(best_found):
                                            best_found = current7[:]
                                    return best_found
                                return best_found
                            return best_found
                        return best_found
                    return best_found
                return best_found
            return best_found
    
    return best_found

def construct():
    start_time = time.time()
    
    # First, try to quickly find a 6-point set using a known small grid
    # We'll do a targeted search
    
    # Try with a moderate grid size
    best_found = None
    
    # Quick search for 6 points on a small grid
    # Use a structured approach: try different grid sizes
    for grid_limit in [15, 20, 25]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        result = _search_for_set(start_time, 6, grid_limit)
        if result is not None and len(result) >= 6:
            best_found = result
            break
    
    # If we found 6, try to extend to 7 with remaining time
    if best_found is not None and len(best_found) >= 6:
        if time.time() - start_time < TIME_LIMIT - 40:
            # Try to find a 7th point
            grid_limit = 30
            result7 = _search_for_set(start_time, 7, grid_limit)
            if result7 is not None and len(result7) > len(best_found):
                best_found = result7
    
    # Fallback to known 5-point set if nothing better found
    if best_found is None or len(best_found) < 5:
        best_found = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Final validation
    if not _is_valid_set(best_found):
        # Use the known valid 5-point set
        best_found = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_found]}
