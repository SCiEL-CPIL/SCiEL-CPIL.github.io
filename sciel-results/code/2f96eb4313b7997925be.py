import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    # bary[i, 0] * V0 + bary[i, 1] * V1 + bary[i, 2] * V2
    # V0 is (0,0) so term 0 vanishes
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack((x, y))

def cart_to_bary(cart):
    """Convert Cartesian (N, 2) to Barycentric (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    
    # Solve for barycentric coordinates (u, v, w) such that P = u*V0 + v*V1 + w*V2
    # u + v + w = 1
    # x = v*1 + w*0.5
    # y = v*0 + w*(sqrt(3)/2)
    
    # From y: w = y / (sqrt(3)/2) = 2*y/sqrt(3)
    w = y / S3H
    # From x: v = x - 0.5*w = x - 0.5*(y/S3H)
    v = x - 0.5 * w
    # u = 1 - v - w
    u = 1.0 - v - w
    
    return np.column_stack((u, v, w))

def compute_min_area_cartesian(points):
    """Compute the minimum area of any triangle formed by three points.
    Normalized by the area of the equilateral triangle."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = (cross / 2) / TRI_AREA
                if area < min_area:
                    min_area = area
    return min_area

def objective_and_grad(bary, beta=50.0):
    """
    Objective: Minimize -log(sum(exp(-beta * area_i))) 
    This approximates -beta * min(area_i) for large beta.
    Returns (value, gradient).
    """
    # Ensure barycentric coordinates are valid (sum to 1, non-negative)
    # SLSQP handles equality constraint sum=1. We add penalty for negativity in objective.
    
    cart = bary_to_cart(bary)
    
    # Compute all triangle areas
    n = N
    areas = []
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = cart[i], cart[j], cart[k]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                cross = dx1 * dy2 - dy1 * dx2
                # Use signed area to maintain differentiability
                # Area = 0.5 * |cross| / TRI_AREA
                # To make it smooth, we can use cross^2 or just work with signed if we assume orientation
                # However, min area is always positive. 
                # Let's use the absolute value but handle gradient carefully or use a smooth approximation.
                # Actually, log-sum-exp of -area is not smooth if area crosses 0.
                # Better: Minimize log(sum(exp(-beta * area_i))) where area_i > 0.
                # If points coincide, area is 0.
                
                # To ensure differentiability, we can use:
                # f = -1/beta * log( sum(exp(-beta * area_i)) )
                # grad f = -1/beta * ( sum(-beta * exp(-beta * area_i) * grad area_i) ) / sum(exp(-beta * area_i))
                #        = sum( exp(-beta * area_i) * grad area_i ) / sum(exp(-beta * area_i))
                
                # We need grad area_i with respect to barycentric coords.
                # This is complex to derive analytically for each triple.
                # Alternative: Use finite differences for gradient? Too slow.
                # Alternative: Use Cartesian coords for optimization? 
                # The inspiration suggests barycentric. 
                # Let's stick to a simpler objective that is differentiable:
                # Minimize sum of (1/area_i^2) or similar? No, that blows up.
                
                # Let's use the "soft min" approach but compute gradient via automatic differentiation logic manually?
                # Or just use Cartesian coordinates with bounds?
                # Cartesian bounds are piecewise linear. SLSQP can handle inequality constraints.
                
                # Let's switch strategy: Optimize in Cartesian coordinates using SLSQP.
                # This avoids the complexity of barycentric gradients.
                pass
                
    # Since the loop above is just a placeholder for thought, let's re-implement objective for Cartesian.
    # But the function signature takes bary. 
    # Let's define the objective in terms of bary, but compute gradient using chain rule from Cartesian.
    # dCart/dBary is a constant linear map.
    # dArea/dCart can be computed analytically for each triangle.
    
    # Let's precompute the indices of triangles
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
                
    areas = np.zeros(len(tri_indices))
    grads_cart = np.zeros((len(tri_indices), n, 2))
    
    for idx, (i, j, k) in enumerate(tri_indices):
        p1, p2, p3 = cart[i], cart[j], cart[k]
        # Area = 0.5 * | (p2-p1) x (p3-p1) | / TRI_AREA
        # Let's assume generic position. 
        # Vectorized computation of areas:
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        area = abs(cross) * 0.5 / TRI_AREA
        areas[idx] = area
        
        # Gradient of area w.r.t. p1, p2, p3
        # A = 0.5 * | v1 x v2 |
        # If cross > 0: A = 0.5 * (v1 x v2)
        # dA/dp1 = -0.5 * (v1 x v2)'_p1 ... 
        # Let's use the property: Area = 0.5 * |det([p2-p1, p3-p1])|
        # Gradient of det([p2-p1, p3-p1]) w.r.t p1 is -[v1, v2] rotated?
        # Let's derive:
        # f(p1, p2, p3) = 0.5 * ( (p2x-p1x)(p3y-p1y) - (p2y-p1y)(p3x-p1x) )
        # df/dp1x = 0.5 * ( -(p3y-p1y) - (p2y-p1y) ) = 0.5 * (p1y - p3y + p1y - p2y) ... wait.
        # df/dp1x = 0.5 * ( - (p3y-p1y) - (p2y-p1y) ) = 0.5 * ( -p3y + p1y - p2y + p1y ) = p1y - 0.5*(p2y+p3y)
        # df/dp1y = 0.5 * ( (p2x-p1x) - (p3x-p1x) ) = 0.5 * (p2x - p3x)
        
        # df/dp2x = 0.5 * (p3y - p1y)
        # df/dp2y = 0.5 * ( -(p3x - p1x) )
        
        # df/dp3x = 0.5 * ( -(p2y - p1y) )
        # df/dp3y = 0.5 * (p2x - p1x)
        
        sign = 1.0 if cross > 0 else -1.0
        if cross == 0:
            sign = 1.0 # Degenerate, gradient undefined, pick arbitrary direction to push apart
            
        # Gradient of the signed area function (without 0.5/TRI_AREA scaling first)
        # g1 = [ p1y - 0.5*(p2y+p3y), 0.5*(p2x-p3x) ]
        # g2 = [ 0.5*(p3y-p1y), -0.5*(p3x-p1x) ]
        # g3 = [ -0.5*(p2y-p1y), 0.5*(p2x-p1x) ]
        
        # The area is 0.5/TRI_AREA * sign * (v1 x v2)
        # So grad Area = 0.5/TRI_AREA * sign * grad (v1 x v2)
        
        scale = 0.5 / TRI_AREA * sign
        
        g1 = np.array([p1[1] - 0.5*(p2[1]+p3[1]), 0.5*(p2[0]-p3[0])])
        g2 = np.array([0.5*(p3[1]-p1[1]), -0.5*(p3[0]-p1[0])])
        g3 = np.array([-0.5*(p2[1]-p1[1]), 0.5*(p2[0]-p1[0])])
        
        grads_cart[idx, i] = scale * g1
        grads_cart[idx, j] = scale * g2
        grads_cart[idx, k] = scale * g3
        
    # Soft min: f = -1/beta * log( sum(exp(-beta * areas)) )
    # To avoid overflow, subtract max area? No, we want min area.
    # exp(-beta * area) is largest for smallest area.
    # Let m = min(areas). 
    # sum(exp(-beta * areas)) = exp(-beta * m) * sum(exp(-beta * (areas - m)))
    # log(sum) = -beta * m + log(sum(exp(-beta * (areas - m))))
    # f = m - 1/beta * log(sum(exp(-beta * (areas - m))))
    
    m = np.min(areas)
    shifted = areas - m
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    val = m - (1.0/beta) * np.log(sum_exp)
    
    # Gradient
    # d/dx f = d/dx [ m - 1/beta * log(S) ]
    # S = sum(exp(-beta * areas))
    # dS/dx = sum( -beta * exp(-beta * area_i) * d(area_i)/dx )
    # d/dx log(S) = (1/S) * dS/dx = -beta * sum( exp(-beta * area_i) * d(area_i)/dx ) / S
    # d/dx f = d(m)/dx + 1/beta * (1/S) * sum( exp(-beta * area_i) * d(area_i)/dx )
    # Note: d(m)/dx is non-differentiable, but the soft min is smooth. 
    # The term d(m)/dx is approximated by the weighted average? 
    # Actually, the formula f = -1/beta log(sum(exp(-beta a_i))) is the smooth approximation.
    # Its gradient is:
    # grad f = sum( (exp(-beta a_i) / S) * grad a_i )
    # Wait, let's check signs.
    # f = -1/beta log S
    # df = -1/beta * (1/S) dS
    # dS = sum( d(exp(-beta a_i)) ) = sum( -beta exp(-beta a_i) da_i )
    # df = -1/beta * (1/S) * sum( -beta exp(-beta a_i) da_i )
    # df = (1/S) * sum( exp(-beta a_i) da_i )
    # So grad f = sum( w_i * grad a_i ) where w_i = exp(-beta a_i) / S.
    
    weights = exp_terms / sum_exp
    grad_cart = np.zeros((n, 2))
    for idx in range(len(tri_indices)):
        w = weights[idx]
        if w > 1e-12:
            grad_cart += w * grads_cart[idx]
            
    # Map gradient from Cartesian to Barycentric
    # cart = bary @ M where M = [[0,0], [1,0], [0.5, S3H]]
    # dCart = dBary @ M
    # grad_bary = M^T @ grad_cart
    # M is 2x3. M^T is 3x2.
    # M columns are V0, V1, V2.
    # M = [[0, 1, 0.5], [0, 0, S3H]]
    # M^T = [[0, 0], [1, 0], [0.5, S3H]]
    
    grad_bary = np.zeros((n, 3))
    grad_bary[:, 0] = 0.0
    grad_bary[:, 1] = grad_cart[:, 0]
    grad_bary[:, 2] = 0.5 * grad_cart[:, 0] + S3H * grad_cart[:, 1]
    
    return val, grad_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_bary = None
    best_score = -1.0
    
    # Initializations
    # 1. Random
    # 2. Grid
    # 3. Vertices + Midpoints + Centroid
    
    inits = []
    
    # Random init
    np.random.seed(42)
    for _ in range(3):
        bary = np.random.dirichlet(np.ones(N) * 1.0, size=N)
        inits.append(bary)
        
    # Grid-like init (Barycentric grid)
    # Sample points on a grid in barycentric space
    # i+j+k = 3 for 4 levels? No, we need 11 points.
    # Let's just use a fixed set of "good" points in barycentric coords
    # Center: (1/3, 1/3, 1/3)
    # Vertices: (1,0,0), (0,1,0), (0,0,1)
    # Midpoints: (0.5, 0.5, 0), etc.
    
    fixed_bary = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0/3.0, 1.0/3.0, 1.0/3.0],
        [0.75, 0.125, 0.125],
        [0.125, 0.75, 0.125],
        [0.125, 0.125, 0.75],
        [0.25, 0.25, 0.5]
    ]
    inits.append(np.array(fixed_bary))
    
    # Another structured init: 3 rings
    # Inner ring: 1 point (center)
    # Middle ring: 3 points
    # Outer ring: 7 points?
    
    for init_bary in inits:
        if time.time() > deadline:
            break
            
        # Normalize to ensure sum is 1
        init_bary = init_bary / init_bary.sum(axis=1, keepdims=True)
        
        # SLSQP Optimization
        # Constraints: sum of bary coords = 1 for each point
        cons = []
        for i in range(N):
            cons.append({
                'type': 'eq',
                'fun': lambda x, i=i: x[i*3:(i+1)*3].sum() - 1.0,
                'jac': lambda x, i=i: np.concatenate([np.zeros(i*3), np.ones(3), np.zeros((N-i-1)*3)])
            })
        
        # Bounds: 0 <= bary <= 1
        bounds = [(0, 1)] * (N * 3)
        
        x0 = init_bary.flatten()
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < 0: # res.fun is -min_area approx
                # Convert back
                bary_opt = res.x.reshape(N, 3)
                # Ensure validity
                bary_opt = np.clip(bary_opt, 0, 1)
                bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
                
                cart_opt = bary_to_cart(bary_opt)
                score = compute_min_area_cartesian(cart_opt)
                
                if score > best_score:
                    best_score = score
                    best_bary = bary_opt.copy()
                    
        except Exception as e:
            pass
            
    if best_bary is None:
        # Fallback
        best_bary = inits[0]
        
    # Final polish with higher beta? 
    # Or just return the best found.
    
    # Ensure we return Cartesian coordinates as per interface
    final_cart = bary_to_cart(best_bary)
    
    return final_cart

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area (normalized):", compute_min_area_cartesian(pts))
