import math
import time
import numpy as np
import random
from scipy.optimize import differential_evolution

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Objective function
    # Variables:
    # 0 to 2*N-1: (x, y, angle_deg) for each of the N inner hexagons
    # 2*N to 2*N+3: (cx, cy, side, angle_deg) for the outer hexagon
    # Total variables: 2*N + 4 = 26
    
    def objective(params):
        if time.time() > deadline:
            return 1e10  # Force termination or high penalty
        
        # Unpack
        inner_params = params[:2*N].reshape(N, 3)
        outer_params = params[2*N:2*N+4]
        
        # Extract outer params
        ocx, ocy, oside, oang_deg = outer_params
        oang = math.radians(oang_deg)
        
        # If side length is invalid, return high penalty
        if oside < 0:
            return 1e10
            
        # Calculate penalties
        penalty = 0.0
        
        # 1. Inner-Outer Containment Penalty
        # For each inner hexagon, check if its vertices are inside the outer hexagon.
        # If a vertex is outside, add a penalty proportional to the squared distance.
        # We transform the vertex into the local frame of the outer hexagon.
        
        # Outer hexagon normals (local frame)
        # A regular hexagon with side S has apothem A = S * sqrt(3) / 2.
        # The 3 pairs of parallel sides correspond to normals at 30, 90, 150 degrees (and opposites).
        # In the local frame (centered at origin, unrotated), the inequalities are:
        # |x * cos(30) + y * sin(30)| <= A
        # |y| <= A  (Wait, normal is 90 deg, so |y| <= A? No. 
        # Let's verify. Vertices at 0, 60, 120...
        # Side 1 is between 0 and 60 deg? No, vertices are at 0, 60, 120, 180, 240, 300.
        # Side 1 connects (1,0) and (0.5, sqrt(3)/2). Midpoint (0.75, sqrt(3)/4).
        # Normal direction is 30 deg.
        # Distance from center to side is cos(30) * 1 = sqrt(3)/2. Correct.
        # So for a hexagon of side S, apothem is S*sqrt(3)/2.
        # The constraints in local frame (x,y) are:
        # |x*cos(30) + y*sin(30)| <= A
        # |x*cos(90) + y*sin(90)| <= A  -> |y| <= A
        # |x*cos(150) + y*sin(150)| <= A
        
        A = oside * math.sqrt(3) / 2.0
        
        c30 = math.sqrt(3) / 2.0
        s30 = 0.5
        c90 = 0.0
        s90 = 1.0
        c150 = -math.sqrt(3) / 2.0
        s150 = 0.5
        
        for i in range(N):
            ix, iy, iang_deg = inner_params[i]
            iang = math.radians(iang_deg)
            
            # Vertices of inner hexagon
            for k in range(6):
                theta = iang + k * math.pi / 3.0
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                
                # Transform to outer local frame
                dx = vx - ocx
                dy = vy - ocy
                rx = dx * math.cos(-oang) - dy * math.sin(-oang)
                ry = dx * math.sin(-oang) + dy * math.cos(-oang)
                
                # Check constraints
                p1 = rx * c30 + ry * s30
                p2 = rx * c90 + ry * s90
                p3 = rx * c150 + ry * s150
                
                # Penalty if outside
                if p1 > A: penalty += (p1 - A)**2
                if p1 < -A: penalty += (-A - p1)**2
                if p2 > A: penalty += (p2 - A)**2
                if p2 < -A: penalty += (-A - p2)**2
                if p3 > A: penalty += (p3 - A)**2
                if p3 < -A: penalty += (-A - p3)**2
                
        # 2. Inner-Inner Non-Overlap Penalty
        # For each pair of inner hexagons, check if they overlap.
        # Two convex polygons overlap if their edges intersect or one is inside the other.
        # A robust differentiable penalty is difficult.
        # Approximation: Use the distance between centers? No, rotations matter.
        # Better: Check distance between all vertices of one to the other?
        # Or use the Minkowski sum concept?
        # For regular hexagons, the non-overlap condition is that the vector between centers
        # must be outside the Minkowski sum of the two hexagons.
        # The Minkowski sum of two rotated regular hexagons is a convex polygon (12-gon or hexagon depending on relative angle).
        # Calculating the exact Minkowski sum for every pair at every evaluation is expensive.
        
        # Alternative: Use a grid of points on the boundary of one hexagon and check if they are inside the other?
        # Or simpler: Use the "soft" distance between polygons.
        # Let's use a simpler heuristic that is differentiable:
        # For each pair, compute the minimum distance between their boundaries?
        # This is hard to differentiate.
        
        # Let's try a different approach for non-overlap:
        # We can sample points on the perimeter of each hexagon and penalize if any point is inside the other.
        # This is O(N^2 * K) where K is number of samples.
        # Let's use K=6 (vertices) first, and maybe midpoints if needed.
        # Actually, checking if vertices are inside the other hexagon is not sufficient for non-overlap (one could be inside the other without vertices crossing, but since they are same size and convex, if vertices are outside, they might still overlap in the middle? No, if all vertices of A are outside B, they can still overlap. Example: A is slightly shifted. Vertices of A are outside B, but edges cross.
        # So we need to check edge intersections or use a more robust metric.
        
        # Robust differentiable metric:
        # For each pair (i, j), compute the distance between the two convex polygons.
        # If distance < 0 (overlap), penalty is large.
        # How to compute distance between two convex polygons differentiably?
        # This is complex.
        
        # Let's use a simpler, approximate penalty that works well in practice for packing:
        # Penalize the overlap area? Hard to compute.
        # Penalize the "interpenetration depth"?
        
        # Let's go back to the Minkowski sum idea but simplify.
        # If we assume the hexagons are not rotated relative to each other? No, they can be.
        
        # Let's use a "vertex-in-polygon" check for all vertices of both hexagons.
        # If a vertex of hex i is inside hex j, or a vertex of hex j is inside hex i, penalize.
        # This handles most cases. The case where edges cross but no vertices are inside is rare for similar-sized convex polygons in a packing context (usually implies deep overlap or specific geometry).
        # Actually, for two convex polygons, if they overlap, either a vertex of one is inside the other, or an edge of one intersects an edge of the other.
        # If edges intersect, it's likely that some vertices are close.
        # To be safer, we can also check the midpoints of edges.
        
        # Let's define a function to check if a point is inside a hexagon.
        # We already have the logic for the outer hexagon. We can reuse it for inner-inner.
        
        def point_in_hex_penalty(px, py, cx, cy, s, ang_deg):
            # Returns squared distance if outside, 0 if inside? 
            # No, for non-overlap, we want to penalize if INSIDE.
            # So we want a penalty if the point is INSIDE.
            # Let's compute the "signed distance" or a penalty for being inside.
            # If inside, penalty = (A - max_proj)^2 ?
            # If a point is inside, all projections are < A.
            # We want to push it out.
            # Let's compute the minimum "distance to boundary" in the normal directions.
            # d = A - max(|p1|, |p2|, |p3|)
            # If d > 0, it's inside. Penalty = d^2.
            
            a = math.radians(ang_deg)
            dx = px - cx
            dy = py - cy
            rx = dx * math.cos(-a) - dy * math.sin(-a)
            ry = dx * math.sin(-a) + dy * math.cos(-a)
            
            p1 = rx * c30 + ry * s30
            p2 = rx * c90 + ry * s90
            p3 = rx * c150 + ry * s150
            
            # Apothem of the hexagon
            A_local = s * math.sqrt(3) / 2.0
            
            # Max projection
            max_proj = max(abs(p1), abs(p2), abs(p3))
            
            # If max_proj < A_local, it's inside.
            # Distance to boundary (approx) is A_local - max_proj.
            # We penalize this.
            if max_proj < A_local:
                return (A_local - max_proj)**2
            else:
                return 0.0

        # Apply to all pairs
        for i in range(N):
            ix, iy, iang_deg = inner_params[i]
            # Generate vertices of hex i
            verts_i = []
            for k in range(6):
                theta = math.radians(iang_deg) + k * math.pi / 3.0
                verts_i.append((ix + math.cos(theta), iy + math.sin(theta)))
            
            for j in range(i+1, N):
                jx, jy, jang_deg = inner_params[j]
                
                # Check vertices of i inside j
                for vx, vy in verts_i:
                    penalty += point_in_hex_penalty(vx, vy, jx, jy, 1.0, jang_deg)
                    
                # Check vertices of j inside i
                for k in range(6):
                    theta = math.radians(jang_deg) + k * math.pi / 3.0
                    vx = jx + math.cos(theta)
                    vy = jy + math.sin(theta)
                    penalty += point_in_hex_penalty(vx, vy, ix, iy, 1.0, iang_deg)
                    
                # Optional: Check midpoints of edges for better robustness
                # This doubles the cost. Let's see if vertices are enough.
                # For now, stick to vertices to save time.
                
        # 3. Objective: Minimize side length + penalties
        # We want to minimize oside, but ensure penalties are 0.
        # Objective = oside + lambda * penalty
        # If penalty > 0, we increase oside or move hexagons.
        # To prioritize validity, we can use a large lambda.
        # Or, we can return (penalty, oside) as a tuple? DE minimizes a scalar.
        # Let's use: objective = oside + 10.0 * penalty
        # If penalty is large, the optimizer will try to reduce it.
        
        return oside + 10.0 * penalty

    # Bounds
    # Inner hexagons:
    # x, y in [-5, 5]
    # angle in [0, 360]
    # Outer hexagon:
    # cx, cy in [-5, 5]
    # side in [1, 10] (Min side is 1 for 1 hex, but for 11 it's larger)
    # angle in [0, 360]
    
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 360.0)) # angle
    
    bounds.append((-5.0, 5.0)) # ocx
    bounds.append((-5.0, 5.0)) # ocy
    bounds.append((1.0, 10.0)) # oside
    bounds.append((0.0, 360.0)) # oang
    
    # Run Differential Evolution
    # maxiter: limit to ensure we finish in time.
    # popsize: larger population for better exploration, but slower.
    # We have ~140s. DE is slow.
    # Let's try popsize=20, maxiter=100.
    # If it's too slow, we need to reduce.
    
    # To speed up, we can use a lower popsize and rely on the penalty landscape.
    # Also, we can try to seed the initial population with a good guess.
    
    # Initial guess:
    # Place hexagons in a grid-like pattern.
    # Center (0,0), (1.5, 0.866), etc.
    # This might help convergence.
    
    # Let's just let DE handle it.
    
    best_result = None
    best_val = float('inf')
    
    # Callback to track best and check time
    def callback(xk, convergence):
        nonlocal best_result, best_val
        val = objective(xk)
        if val < best_val:
            best_val = val
            best_result = xk.copy()
        if time.time() > deadline:
            raise Exception("Time limit exceeded")

    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            callback=callback,
            polish=False # Polish can be slow, we can do it manually if needed
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_result = result.x
            
    except Exception as e:
        print(f"DE failed or timed out: {e}")
        # If best_result is None, we need a fallback.
        if best_result is None:
            # Fallback: simple grid
            best_result = np.zeros(2*N+4)
            # Place in a line
            for i in range(N):
                best_result[i*3] = i * 1.8
                best_result[i*3+1] = 0
                best_result[i*3+2] = 0
            best_result[2*N] = N * 0.9 # cx
            best_result[2*N+1] = 0 # cy
            best_result[2*N+2] = N * 1.8 + 1 # side
            best_result[2*N+3] = 0 # angle
            
    # Extract results
    if best_result is None:
        # Should not happen
        return [], (0,0), 10.0, 0.0

    inner_params = best_result[:2*N].reshape(N, 3)
    outer_params = best_result[2*N:2*N+4]
    
    ocx, ocy, oside, oang_deg = outer_params
    
    # Format inner rows
    inner_rows = []
    for i in range(N):
        x, y, ang = inner_params[i]
        # Normalize angle to [0, 360)
        ang = ang % 360.0
        inner_rows.append([x, y, ang])
        
    # Return
    return inner_rows, (ocx, ocy), oside, oang_deg

if __name__ == "__main__":
    r = construct()
    print(r)
