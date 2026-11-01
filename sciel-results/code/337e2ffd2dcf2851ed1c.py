import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_vertices(cx, cy, s=1.0, angle_deg=0.0):
        """Return 6 vertices of a regular hexagon."""
        a = np.radians(angle_deg)
        angles = a + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ])

    def required_side(positions, outer_angle=0.0):
        """
        Calculate the minimum side length of the outer hexagon (centered at origin, rotated by outer_angle)
        to contain all inner hexagons defined by positions (x, y, angle).
        """
        # Outer hexagon support function for a given direction theta
        # The outer hexagon has side S. The distance from center to edge in direction theta is:
        # d(theta) = S * sqrt(3) / 2 * 1 / cos((theta - 30 - k*60) mod 60 - 30) ? 
        # Actually, for a regular hexagon with side S, the apothem is S*sqrt(3)/2.
        # The support function h(θ) = apothem / cos(Δ) where Δ is the angle to the nearest normal.
        
        # Let's compute the max projection of all inner vertices onto the 6 normal directions of the outer hexagon.
        # Outer hexagon normals are at angles: outer_angle + 90 + k*60.
        # Wait, the "side length" S is related to the apothem a by S = 2a/sqrt(3).
        # If the max projection of a point p onto a normal n is P, then we need apothem >= P.
        # So S >= 2 * P / sqrt(3).
        
        # We need to maximize over the 6 normals of the outer hexagon.
        # Let normals be u_k = [cos(θ_k), sin(θ_k)] where θ_k = outer_angle + 90 + k*60.
        # For each inner hexagon, we need its max projection onto u_k.
        # Max projection of a convex polygon onto u is max(v · u) for vertices v.
        
        # Total objective is max over k of (max over all inner hexagons of (max vertex dot u_k)) * 2/sqrt(3).
        
        all_v = np.vstack([hex_vertices(*p) for p in positions])
        
        # Outer hexagon orientation
        theta0 = np.radians(outer_angle)
        # Normals of the outer hexagon (pointing outwards)
        # For a regular hexagon centered at origin, vertices are at angles 0, 60, ...
        # Normals are at 30, 90, ... if vertices are at 0, 60...
        # Let's stick to standard: Vertices at 0, 60, 120, 180, 240, 300.
        # Normals at 30, 90, 150, 210, 270, 330.
        normal_angles = theta0 + np.radians(30 + 60 * np.arange(6))
        
        max_proj = 0.0
        for na in normal_angles:
            u = np.array([np.cos(na), np.sin(na)])
            projs = all_v @ u
            max_proj = max(max_proj, np.max(projs))
            
        # The distance from center to the side is the apothem.
        # Apothem = max_proj.
        # Side length S = 2 * Apothem / sqrt(3).
        return 2.0 * max_proj / np.sqrt(3.0)

    def smooth_overlap_penalty(positions, outer_angle=0.0, temperature=0.1):
        """
        Smooth penalty for overlaps between inner hexagons.
        Uses the Minkowski difference support function.
        Two convex polygons A and B are disjoint iff for all directions u, h_A(u) + h_B(-u) <= ||c_A - c_B||_u?
        Actually, A and B are disjoint iff the distance between them is > 0.
        A robust smooth approximation of the distance between two convex polygons is:
        min_{u} (h_A(u) + h_B(-u) - (c_A - c_B)·u)
        If this min is negative, they overlap.
        We want to penalize this.
        Let D(u) = h_A(u) + h_B(-u) - (c_A - c_B)·u.
        The minimum over u of D(u) is the signed distance (negative if overlap).
        We can approximate the min with a soft-min (log-sum-exp) or just take the min over a set of directions.
        Since the polygons are hexagons, the critical directions are the normals of the edges.
        The Minkowski sum A + (-B) is a convex polygon. The distance from origin to this polygon is the separation.
        If we define g(u) = h_A(u) + h_B(-u) - (c_A - c_B)·u, then min_u g(u) < 0 means overlap.
        We can use a penalty like max(0, -min_u g(u)) or a smooth version.
        
        For efficiency, we check the 12 normals (6 from A, 6 from B) and maybe bisectors.
        """
        penalty = 0.0
        # Precompute vertices for all hexagons
        verts = [hex_vertices(*p) for p in positions]
        
        for i in range(N):
            for j in range(i + 1, N):
                vi = verts[i]
                vj = verts[j]
                ci = np.array(positions[i][:2])
                cj = np.array(positions[j][:2])
                
                # Relative position
                dc = ci - cj
                
                # Normals for vi and vj
                # Edges of vi: v_{k+1} - v_k. Normal: rotate 90 deg.
                # We need h_vi(u) and h_vj(-u).
                
                min_sep = float('inf')
                
                # Generate candidate directions: normals of vi, normals of vj
                dirs = []
                for k in range(6):
                    e = vi[(k+1)%6] - vi[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    dirs.append(n)
                    dirs.append(-n)
                    
                for k in range(6):
                    e = vj[(k+1)%6] - vj[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    dirs.append(n)
                    dirs.append(-n)
                
                # Also add directions of dc if non-zero
                if np.linalg.norm(dc) > 1e-6:
                    dirs.append(dc / np.linalg.norm(dc))
                    dirs.append(-dc / np.linalg.norm(dc))
                
                for u in dirs:
                    # h_vi(u) = max(vi @ u)
                    h_i = np.max(vi @ u)
                    # h_vj(-u) = max(vj @ (-u)) = - min(vj @ u)
                    h_j = -np.min(vj @ u)
                    
                    # Separation in direction u
                    # The projection of the gap is (ci - cj)·u - h_i - h_j?
                    # Wait. 
                    # Distance from ci to vi edge in direction u is h_i.
                    # Distance from cj to vj edge in direction -u is h_j.
                    # If (ci - cj)·u < h_i + h_j, they overlap in this direction?
                    # Actually, if (ci - cj)·u <= h_i + h_j, the projections overlap.
                    # Separation = (ci - cj)·u - (h_i + h_j).
                    # If this is positive, they are separated in this direction.
                    # We want the MINIMUM separation over all directions.
                    sep = dc @ u - (h_i + h_j)
                    if sep < min_sep:
                        min_sep = sep
                
                # min_sep is the approximate minimum separation.
                # If min_sep < 0, they overlap.
                # We want a smooth penalty.
                # Use softplus or exponential.
                # penalty += max(0, -min_sep)^2 is non-diff.
                # Use: penalty += -temperature * log(1 + exp(-min_sep/temperature)) ?
                # Or simpler: if min_sep < 0, penalty += (-min_sep)**2.
                # To make it differentiable, use a smooth hinge:
                # p = 0.5 * (min_sep - sqrt(min_sep^2 + eps^2)) * -1 ?
                # Let's use: penalty += max(0, -min_sep) ** 2. This is not differentiable at 0.
                # Use smooth max: smax(x, 0) = 0.5 * (x + sqrt(x^2 + eps^2)).
                # We want to penalize negative min_sep.
                # Let y = -min_sep. We want smax(y, 0)^2.
                y = -min_sep
                if y > 0:
                    # Smooth square of positive part
                    # Approximation: y^2 for y>0, 0 for y<0.
                    # Use: 0.5 * (y + sqrt(y^2 + 1e-4)) ** 2 ? No.
                    # Just use y^2 if y > 1e-6, else 0.5 * (y + sqrt(y^2+1e-6))**2 * 0.5?
                    # Let's stick to a simple differentiable approximation:
                    # p = 0.5 * (y + sqrt(y^2 + 1e-4))
                    # penalty += p * p
                    p = 0.5 * (y + np.sqrt(y*y + 1e-4))
                    penalty += p * p
                else:
                    penalty += 0.0
        return penalty

    def objective(params):
        """
        params: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
        """
        positions = params[:3*N].reshape(N, 3)
        outer_angle = params[3*N]
        
        # Objective: Minimize outer side length + Penalty
        side = required_side(positions, outer_angle)
        
        # Penalty for overlaps
        # Scale penalty to be comparable to side length
        pen = smooth_overlap_penalty(positions, outer_angle)
        
        # Weight for penalty
        w = 10.0
        return side + w * pen

    def objective_grad(params):
        val = objective(params)
        eps = 1e-5
        grad = np.zeros_like(params)
        for k in range(len(params)):
            p2 = params.copy()
            p2[k] += eps
            grad[k] = (objective(p2) - val) / eps
        return val, grad

    # Initial guess
    # Place hexagons in a loose hexagonal lattice
    s3 = np.sqrt(3)
    init_params = np.zeros(3*N + 1)
    for i in range(N):
        # Simple grid or circle
        # Circle of radius 3
        a = 2 * np.pi * i / N
        init_params[3*i] = 3.0 * np.cos(a)
        init_params[3*i+1] = 3.0 * np.sin(a)
        init_params[3*i+2] = 0.0
    init_params[3*N] = 0.0

    # Optimization
    best_params = init_params
    best_val = objective(init_params)
    
    # Use L-BFGS-B
    try:
        res = minimize(
            objective_grad, 
            init_params, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if res.success or res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception as e:
        print("Optimization error:", e)
        pass

    # Refinement: Try a few random restarts if time permits
    if time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            # Perturb initial guess
            p0 = best_params.copy()
            p0[:3*N] += np.random.normal(0, 0.5, 3*N)
            # Keep angles somewhat stable
            p0[3*N] = np.random.uniform(0, 60)
            
            try:
                res = minimize(
                    objective_grad, 
                    p0, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_params = res.x
                    best_val = res.fun
            except:
                pass

    # Extract results
    final_positions = best_params[:3*N].reshape(N, 3)
    final_outer_angle = best_params[3*N]
    
    # Calculate final side
    final_side = required_side(final_positions, final_outer_angle)
    
    # Construct return values
    inner = []
    for i in range(N):
        x, y, a = final_positions[i]
        inner.append([float(x), float(y), float(a)])
        
    return inner, (0.0, 0.0), float(final_side), float(final_outer_angle)

# EVOLVE-BLOCK-END
