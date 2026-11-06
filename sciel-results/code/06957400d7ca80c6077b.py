"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in radians for a set of unit-sphere points.
    Uses the circumradius of each facet of the convex hull."""
    from scipy.spatial import ConvexHull, Delaunay
    
    # Use ConvexHull to get facets
    hull = ConvexHull(points)
    max_rad = 0.0
    
    for simplex in hull.simplices:
        # Get the 3 points forming this facet (triangle)
        p1, p2, p3 = points[simplex]
        
        # Compute circumradius of the spherical triangle
        # The circumradius R satisfies: for a spherical triangle with sides a,b,c,
        # cos(R) = (cos(a) - cos(b)cos(c)) / (sin(b)sin(c)) ... but this is for the circumcircle
        # Actually, for a spherical triangle, the circumradius R is given by:
        # tan(R) = (sin(a)sin(b)sin(c)) / (sin(a+b+c) ... ) 
        # Simpler: find the point on the sphere equidistant from p1, p2, p3.
        # The circumcenter on the sphere is the normalized vector:
        # v = p1 * sin(2*angle1) + p2 * sin(2*angle2) + p3 * sin(2*angle3) ... 
        # Actually, the circumcenter of a spherical triangle with vertices p1,p2,p3 is:
        # v = (p1 × p2) × (p2 × p3) ... normalized? No.
        
        # The circumcenter c satisfies c·p1 = c·p2 = c·p3 = cos(R)
        # So c is proportional to p1 + p2 + p3 only if they form a right triangle? No.
        # The correct formula: the circumcenter is along the vector:
        # v = (p1 × p2) × (p2 × p3) + (p2 × p3) × (p3 × p1) + (p3 × p1) × (p1 × p2)
        # This is the cross product of the normals... actually simpler:
        # The circumcenter is the normalized sum of the three points weighted by the sine of twice the opposite angles.
        # But an easier approach: solve for c in the plane spanned by p1,p2,p3 such that c·p1=c·p2=c·p3 and |c|=1.
        
        # Let c = a*p1 + b*p2 + c_coeff*p3
        # c·p1 = c·p2 => a + b*(p2·p1) + c_coeff*(p3·p1) = a*(p1·p1) + b*(p2·p1) + c_coeff*(p3·p1)
        # Since |p1|=|p2|=|p3|=1:
        # c·p1 = a + b*d12 + c_coeff*d13
        # c·p2 = a*d12 + b + c_coeff*d23
        # Set equal: a(1-d12) + c_coeff(d13-d23) = 0 ... this is getting complex.
        
        # Alternative: the circumcenter of a spherical triangle is given by:
        # v = (p1 * sin(b+c-a) + p2 * sin(a+c-b) + p3 * sin(a+b-c)) / norm
        # where a,b,c are the side lengths (angular distances).
        
        a = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        b = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        c = np.arccos(np.clip(np.dot(p3, p1), -1, 1))
        
        # Check if the triangle is degenerate or has an issue
        if a + b + c > 2 * np.pi:
            # This shouldn't happen for a proper spherical triangle on the convex hull
            # But let's handle it
            continue
            
        # The circumcenter vector
        w1 = np.sin(b + c - a)
        w2 = np.sin(a + c - b)
        w3 = np.sin(a + b - c)
        
        v = w1 * p1 + w2 * p2 + w3 * p3
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-10:
            # Degenerate case, use the center of the three points
            v = (p1 + p2 + p3) / 3.0
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
        
        c_point = v / norm_v
        
        # The circumradius is the angular distance from c_point to any of p1, p2, p3
        cos_R = np.dot(c_point, p1)
        cos_R = np.clip(cos_R, -1, 1)
        R = np.arccos(cos_R)
        
        if R > max_rad:
            max_rad = R
    
    return max_rad


def _objective_and_gradient(x):
    """Objective function and gradient for optimization.
    x is a flattened array of 3*N coordinates."""
    points = x.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    # Compute covering radius
    cr = _covering_radius(points)
    
    return cr, None  # Gradient is complex, use numerical or L-BFGS-B with finite diff


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # Start with a Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()
    
    # Use L-BFGS-B to optimize
    # We need to handle the fact that points must stay on the unit sphere.
    # We'll use a penalty method or project back.
    
    def objective(x):
        points = x.reshape(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        cr = _covering_radius(points)
        # Add a penalty for points not being well-separated (to avoid coincident points)
        # Compute pairwise distances
        dots = np.dot(points, points.T)
        np.fill_diagonal(dots, 1.0)
        min_dot = np.min(dots)
        # Penalty if any two points are too close
        penalty = 0.0
        if min_dot > 0.999:  # angular distance < ~1 degree
            penalty = 100.0 * (min_dot - 0.999)
        return cr + penalty
    
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    
    points = result.x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
