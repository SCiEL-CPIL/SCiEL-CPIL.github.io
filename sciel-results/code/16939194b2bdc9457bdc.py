import itertools
import math

def construct():
    # WLOG, place one point at (0,0).
    # All other points must be at integer distance from (0,0) 
    # if we restrict to integer coordinates.
    # This generates a candidate pool of "Pythagorean points".
    
    N = 100
    candidates = [(0, 0)]
    
    # Generate Pythagorean points
    for x in range(-N, N + 1):
        for y in range(-N, N + 1):
            if x == 0 and y == 0:
                continue
            d2 = x * x + y * y
            d = int(math.isqrt(d2))
            if d * d == d2:
                candidates.append((x, y))
                
    # Sort candidates to help backtracking? 
    # Maybe sort by distance to origin or degree?
    # For now, just use the list.
    
    # Precompute pairwise distances squared for candidates
    # We only care about integer squared distances.
    n = len(candidates)
    dist2 = [[0] * n for _ in range(n)]
    valid_pair = [[False] * n for _ in range(n)]
    
    for i in range(n):
        x1, y1 = candidates[i]
        for j in range(i + 1, n):
            x2, y2 = candidates[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx * dx + dy * dy
            d = int(math.isqrt(d2))
            if d * d == d2:
                valid_pair[i][j] = True
                valid_pair[j][i] = True
            dist2[i][j] = d2
            dist2[j][i] = d2

    # Backtracking search
    # We must include (0,0) which is index 0.
    best_set = [0]
    
    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        # Determinant check
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 | = 0
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 |
        d1 = x1*x1 + y1*y1
        d2 = x2*x2 + y2*y2
        d3 = x3*x3 + y3*y3
        d4 = x4*x4 + y4*y4
        
        # Expansion
        # Using fractions to be safe? No, integers are fine for determinant.
        # det = x1(y2(d3-d4) - d2(y3-y4) + (y3*d4 - d3*y4))
        #       - y1(x2(d3-d4) - d2(x3-x4) + (x3*d4 - d3*x4))
        #       + d1(x2(y3-y4) - y2(x3-x4) + (x3-y3)) -- Wait, last term is (x3*1 - 1*x4) = x3-x4? No.
        # Let's use a generic det4 function to avoid errors.
        
        M = [[x1, y1, d1, 1],
             [x2, y2, d2, 1],
             [x3, y3, d3, 1],
             [x4, y4, d4, 1]]
             
        def det4(M):
            return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]))
                - M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]))
                + M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
                - M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))
                
        return det4(M) == 0

    def backtrack(current_set, start_idx):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Heuristic: try to add points that connect to many existing points
        # For simplicity, iterate through remaining candidates
        for i in range(start_idx, n):
            if i in current_set:
                continue
                
            # Check distance to all in current_set
            ok = True
            for j in current_set:
                if not valid_pair[i][j]:
                    ok = False
                    break
            if not ok:
                continue
                
            # Check collinearity
            p_new = candidates[i]
            collinear = False
            for a in range(len(current_set) - 1):
                for b in range(a + 1, len(current_set)):
                    if is_collinear(candidates[current_set[a]], candidates[current_set[b]], p_new):
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
                
            # Check concyclicity
            concyclic = False
            if len(current_set) >= 3:
                # Check if p_new is on circle through any 3 points in current_set
                indices = current_set
                for a in range(len(indices)):
                    for b in range(a + 1, len(indices)):
                        for c in range(b + 1, len(indices)):
                            if is_concyclic(candidates[indices[a]], candidates[indices[b]], candidates[indices[c]], p_new):
                                concyclic = True
                                break
                        if concyclic:
                            break
                    if concyclic:
                        break
            if concyclic:
                continue
                
            # Add and recurse
            current_set.append(i)
            backtrack(current_set, i + 1)
            current_set.pop()
            
    # Start with (0,0)
    backtrack([0], 1)
    
    # Convert best_set indices to points
    # best_set contains indices into 'candidates'
    # We need to return D=1 and points
    # If best_set has < 3 points, we might need to handle it, but 3 is the min.
    # The problem says 3 to 12.
    
    # Ensure we have at least 3 points. If not, add some arbitrary valid ones?
    # The backtrack should find at least 3 easily.
    
    points = []
    for idx in best_set:
        x, y = candidates[idx]
        points.append([[[x, 1], [y, 1]]])
        
    # Wait, the return format is:
    # {'D': D, 'points': [[[x_num, x_den], [y_num, y_den]], ...]}
    # So points is a list of [ [x_num, x_den], [y_num, y_den] ]
    
    formatted_points = []
    for idx in best_set:
        x, y = candidates[idx]
        formatted_points.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': formatted_points}
