import numpy as np
from scipy.optimize import minimize
import time

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Regular N-gon
    # Gaps are the arc lengths between consecutive vertices on the unit circle
    # Sum of gaps = 2 * pi
    initial_gaps = np.full(N, 2 * np.pi / N)
    
    # We optimize the first N-1 gaps. The last gap is 2*pi - sum(first N-1)
    # To ensure positivity, we can optimize in log-space or just use bounds?
    # Nelder-Mead doesn't support bounds well.
    # Given the starting point is symmetric and likely optimal, small perturbations 
    # should keep gaps positive.
    
    def calculate_score_from_gaps(gaps_opt):
        # gaps_opt has length N-1
        gaps = np.empty(N)
        gaps[:N-1] = gaps_opt
        gaps[N-1] = 2 * np.pi - np.sum(gaps_opt)
        
        # Check for invalid gaps (negative or zero)
        if np.any(gaps <= 1e-12):
            return -1e10 # Penalize heavily
            
        # Calculate Hull Area (proportional)
        # Area = 0.5 * R^2 * sum(sin(gap_i))
        hull_area_prop = np.sum(np.sin(gaps))
        
        # Calculate Minimum Triangle Area (proportional)
        # Area = 2 * R^2 * sin(a/2) * sin(b/2) * sin(c/2)
        # where a, b, c are the three arcs summing to 2*pi.
        # Let a = sum(gaps[i:j]), etc.
        
        min_tri_area_prop = float('inf')
        
        # Precompute prefix sums for faster arc calculation?
        # N is small (14), so O(N^3) is fine (2744 ops)
        # But we can be smarter.
        # For each pair (i, j), the arc length is known.
        # We need to iterate over all triples (i, j, k).
        # There are 14 * 13 * 12 / 6 = 364 triangles.
        
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for k in range(N):
                    if i == k or j == k: continue
                    
                    # Determine the three arcs
                    # Sort indices to find the arcs between them in circular order
                    # This is a bit slow. 
                    # Alternative: 
                    # The arcs are determined by the positions.
                    # Let's just compute the arc lengths directly.
                    
                    # Arc from i to j (forward)
                    # Arc from j to k (forward)
                    # Arc from k to i (forward)
                    
                    # To get arc from i to j forward:
                    # sum of gaps from i to j-1
                    # This logic is tricky with wrap-around.
                    
                    # Let's use the property that for points on a circle,
                    # the area is determined by the three arc lengths.
                    # We can just iterate over all combinations of 3 distinct indices.
                    
        # Let's rewrite the triangle loop more efficiently.
        # Precompute all pairwise arc lengths?
        # arc[i][j] = sum of gaps from i to j (exclusive j, inclusive i) in forward direction.
        arcs = np.zeros((N, N))
        for i in range(N):
            s = 0.0
            for j in range(N):
                arcs[i][j] = s
                s += gaps[j]
            # Note: arcs[i][i] = 0. arcs[i][(i+1)%N] = gaps[i].
            # arcs[i][j] is the arc length from i to j moving forward.
            
        min_tri_area_prop = float('inf')
        for i in range(N):
            for j in range(N):
                if i == j: continue
                for k in range(N):
                    if i == k or j == k: continue
                    
                    # Get the three arcs
                    # The vertices are i, j, k.
                    # The arcs are:
                    # a = arc from i to j
                    # b = arc from j to k
                    # c = arc from k to i
                    # But we must ensure they sum to 2*pi and are the "inner" arcs.
                    # Actually, the triangle is formed by the chords.
                    # The area formula 2R^2 sin(a/2)sin(b/2)sin(c/2) works for the arcs 
                    # subtended by the sides at the center, which must sum to 2*pi.
                    # The arcs corresponding to the sides of the triangle are:
                    # Side ij subtends arc(i,j) or 2*pi - arc(i,j)?
                    # The inscribed angle theorem: Angle at k is half the arc ij not containing k.
                    
                    # Let's just calculate the three arc lengths between the points in circular order.
                    # Sort i, j, k in circular order starting from i?
                    # Easier: 
                    # a_ij = arcs[i][j]
                    # a_jk = arcs[j][k]
                    # a_ki = arcs[k][i]
                    # If we traverse i->j->k->i, the sum of arcs is 2*pi.
                    # Is this always the case?
                    # Yes, if we define arcs[x][y] as the forward distance.
                    # arcs[i][j] + arcs[j][k] + arcs[k][i] = 2*pi.
                    
                    a = arcs[i][j]
                    b = arcs[j][k]
                    c = arcs[k][i]
                    
                    # The area is 2 * sin(a/2) * sin(b/2) * sin(c/2)
                    # (R=1, factor 2 from formula 2R^2... wait.
                    # Area = 0.5 * R^2 * (sin(a) + sin(b) + sin(c)) ? No.
                    # Area of triangle with circumradius R and central angles a, b, c:
                    # Area = 2 R^2 sin(a/2) sin(b/2) sin(c/2) is WRONG.
                    # Correct formula: Area = 2 R^2 sin(A) sin(B) sin(C) where A,B,C are angles of triangle.
                    # Angle A (at vertex k) = a/2? No.
                    # The angle at vertex k subtends arc ij. The arc ij (not containing k) is a.
                    # So angle K = a/2.
                    # Similarly angle I = c/2, angle J = b/2.
                    # So Area = 2 R^2 sin(a/2) sin(b/2) sin(c/2).
                    # Let's verify for equilateral triangle. a=b=c=2*pi/3.
                    # Area = 2 * 1 * sin(pi/3)^3 = 2 * (sqrt(3)/2)^3 = 2 * 3sqrt(3)/8 = 3sqrt(3)/4.
                    # Regular triangle area in unit circle: 3 * 0.5 * 1 * 1 * sin(120) = 1.5 * sqrt(3)/2 = 3sqrt(3)/4. Correct.
                    
                    tri_area = 2.0 * np.sin(a/2.0) * np.sin(b/2.0) * np.sin(c/2.0)
                    
                    if tri_area < min_tri_area_prop:
                        min_tri_area_prop = tri_area
                        
        if hull_area_prop == 0:
            return -1e10
            
        score = min_tri_area_prop / hull_area_prop
        return -score # We want to maximize, so minimize negative

    # Optimization
    # We optimize the first N-1 gaps.
    x0 = initial_gaps[:N-1]
    
    # Check if we have time
    if time.time() > deadline:
         # Return regular polygon
         angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
         return np.column_stack((np.cos(angles), np.sin(angles)))

    try:
        # Nelder-Mead is robust for small problems
        res = minimize(calculate_score_from_gaps, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if time.time() > deadline:
             # Return regular polygon if optimization takes too long
             angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
             return np.column_stack((np.cos(angles), np.sin(angles)))

        # Check if result is better than initial
        if res.success or res.fun < calculate_score_from_gaps(x0):
            best_gaps_opt = res.x
        else:
            best_gaps_opt = x0
            
    except Exception:
        best_gaps_opt = x0

    # Reconstruct points from best gaps
    best_gaps = np.empty(N)
    best_gaps[:N-1] = best_gaps_opt
    best_gaps[N-1] = 2 * np.pi - np.sum(best_gaps_opt)
    
    # Ensure gaps are positive (clamp if necessary, though minimizer should keep them)
    if np.any(best_gaps <= 0):
        # Fallback to regular
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

    # Calculate angles
    angles = np.zeros(N)
    angles[0] = 0
    for i in range(1, N):
        angles[i] = angles[i-1] + best_gaps[i-1]
        
    # Normalize angles to [0, 2*pi)
    angles = angles % (2 * np.pi)
    
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return points
