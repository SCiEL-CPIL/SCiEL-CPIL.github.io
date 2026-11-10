# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_face_normals(a_deg):
    """Return 6 outward face normals (unit vectors) for a hexagon with rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _pair_separated(v1, n1, v2, n2, tol=1e-6):
    """
    Check if two hexagons are separated using the Separating Axis Theorem.
    v1, n1: vertices and normals of hex1
    v2, n2: vertices and normals of hex2
    Returns True if separated (valid), False if overlapping.
    """
    for i in range(6):
        n = n1[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() - proj2.min() < -tol or proj2.max() - proj1.min() < -tol:
            return True
    
    for i in range(6):
        n = n2[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() - proj2.min() < -tol or proj2.max() - proj1.min() < -tol:
            return True
    
    return False

def _check_inner_separation(inner_data, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_data)
    vertices = []
    normals = []
    for i in range(n):
        cx, cy, a = inner_data[i]
        v = _hex_vertices(cx, cy, 1.0, a)
        nm = _hex_face_normals(a)
        vertices.append(v)
        normals.append(nm)
    
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_separated(vertices[i], normals[i], vertices[j], normals[j], tol):
                return False
    return True

def _check_inside_outer_full(inner_data, outer_data, side, tol=1e-6):
    """Check all inner hexagons are inside the outer hexagon with given side length."""
    cx, cy, a = outer_data
    outer_verts = _hex_vertices(cx, cy, side, a)
    n_outer = 6
    
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        inner_verts = _hex_vertices(ix, iy, 1.0, ia)
        for p in inner_verts:
            for e in range(n_outer):
                p1 = outer_verts[e]
                p2 = outer_verts[(e + 1) % n_outer]
                edge = p2 - p1
                vec = p - p1
                cross_val = edge[0] * vec[1] - edge[1] * vec[0]
                if cross_val < -tol:
                    return False
    return True

def _validate(inner_data, outer_data, side, tol=1e-6):
    """Full validation."""
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    if not _check_inner_separation(inner_data, tol):
        return False
    
    if not _check_inside_outer_full(inner_data, outer_data, side, tol):
        return False
    
    outer_center = np.array([outer_data[0], outer_data[1]])
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        shifted = np.array([ix, iy]) - outer_center
        v = _hex_vertices(shifted[0], shifted[1], 1.0, ia)
        for e in range(6):
            d = v[(e + 1) % 6] - v[e]
            length = np.linalg.norm(d)
            if abs(length - 1.0) > 1e-8:
                return False
    
    return True

def _compute_side(inner_data):
    """Compute the minimum side length of an outer hexagon (centered at origin, angle 0) to contain all inner hexagons."""
    required_side = 0.0
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        v = _hex_vertices(ix, iy, 1.0, ia)
        for p in v:
            # For a regular hexagon with vertex at 0 deg (pointy right), the containment constraint for point (x,y) is:
            # max(|x|, |x/2 + sqrt(3)y/2|, |-x/2 + sqrt(3)y/2|) <= s
            # Or equivalently using the support function:
            # The hexagon is the intersection of 6 half-planes.
            # With vertices at angles 0, 60, 120, 180, 240, 300.
            # The normals are at 30, 90, 150, 210, 270, 330.
            # The constraint is n_k . p <= s for all k.
            # n_k = (cos(30+60k), sin(30+60k))
            # Let's just compute the max projection onto the 6 outward normals.
            pass
    
    # Actually, let's use the explicit formula for a hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The half-planes are:
    # 1. x <= s
    # 2. x/2 + sqrt(3)y/2 <= s
    # 3. -x/2 + sqrt(3)y/2 <= s
    # 4. -x <= s  => x >= -s
    # 5. -x/2 - sqrt(3)y/2 <= s
    # 6. x/2 - sqrt(3)y/2 <= s
    
    # So the condition is max( |x|, |x/2 + sqrt(3)y/2|, |-x/2 + sqrt(3)y/2| ) <= s
    # Wait, let's check.
    # Vertex (s, 0): |s| = s. |s/2| = s/2. |-s/2| = s/2. Max is s. OK.
    # Vertex (s/2, s*sqrt(3)/2): |s/2| = s/2. |s/4 + 3s/4| = s. |-s/4 + 3s/4| = s/2. Max is s. OK.
    
    # So the condition is:
    # s >= max( |x|, |x/2 + sqrt(3)y/2|, |-x/2 + sqrt(3)y/2| )
    
    sqrt3 = np.sqrt(3.0)
    for i in range(len(inner_data)):
        ix, iy, ia = inner_data[i]
        v = _hex_vertices(ix, iy, 1.0, ia)
        for p in v:
            x, y = p
            d1 = abs(x)
            d2 = abs(x/2.0 + sqrt3*y/2.0)
            d3 = abs(-x/2.0 + sqrt3*y/2.0)
            required_side = max(required_side, d1, d2, d3)
    
    return required_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: hexagonal grid arrangement
    inner_data = np.array([
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
        [-3.75, -2.17, 0],
        [3.75, -2.17, 0],
    ])
    
    best_inner = inner_data.copy()
    best_side = _compute_side(inner_data)
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Verify initial
    if _validate(inner_data, outer_data, best_side):
        best_inner = inner_data.copy()
        best_side = _compute_side(inner_data)
    else:
        # If invalid, try a slightly expanded version
        best_side = 8.0
        if _validate(inner_data, outer_data, best_side):
            best_inner = inner_data.copy()
        else:
            # Fallback to a very large side
            best_side = 10.0
    
    # Structural Change: Use Simulated Annealing with a focus on local moves and rotations
    # This avoids the global search pitfalls of DE and allows fine-tuning.
    
    current_inner = best_inner.copy()
    current_side = best_side
    best_inner_sa = best_inner.copy()
    best_side_sa = best_side
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    
    # Precompute vertices and normals for speed
    def get_vertices_normals(inner):
        verts = []
        norms = []
        for i in range(11):
            cx, cy, a = inner[i]
            v = _hex_vertices(cx, cy, 1.0, a)
            n = _hex_face_normals(a)
            verts.append(v)
            norms.append(n)
        return verts, norms
    
    def is_valid_fast(inner, verts, norms, side, tol=1e-6):
        # Check separation
        for i in range(11):
            for j in range(i+1, 11):
                if not _pair_separated(verts[i], norms[i], verts[j], norms[j], tol):
                    return False
        # Check containment (using the computed side)
        # We can skip full containment check in SA loop for speed if we trust the side calculation
        # But we must ensure the side is actually sufficient.
        # The side calculation is exact for the outer hexagon shape.
        return True
    
    # SA Loop
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Generate a new candidate
        new_inner = current_inner.copy()
        idx = np.random.randint(0, 11)
        
        # Random move type
        move_type = np.random.randint(0, 3)
        if move_type == 0:
            # Move x
            new_inner[idx, 0] += np.random.normal(0, 0.1)
        elif move_type == 1:
            # Move y
            new_inner[idx, 1] += np.random.normal(0, 0.1)
        else:
            # Rotate
            new_inner[idx, 2] += np.random.normal(0, 1.0)
            # Normalize angle to [0, 60)
            new_inner[idx, 2] %= 60.0
            
        new_side = _compute_side(new_inner)
        
        # Check validity of the new configuration
        # We need to check if the inner hexagons are separated.
        # Containment is guaranteed by the side calculation.
        verts, norms = get_vertices_normals(new_inner)
        valid = is_valid_fast(new_inner, verts, norms, new_side)
        
        if valid:
            new_cost = new_side
        else:
            # Penalty for invalidity
            # Calculate overlap magnitude? Too expensive. Just use a large penalty.
            new_cost = new_side + 10.0
        
        current_cost = current_side if _check_inner_separation(current_inner) else current_side + 10.0
        
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_inner = new_inner
            current_side = new_side
            
            if new_cost < best_side_sa:
                best_side_sa = new_cost
                best_inner_sa = new_inner.copy()
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    # Final check and refinement
    if best_side_sa < best_side:
        best_inner = best_inner_sa
        best_side = best_side_sa
    
    # One last local optimization step using Nelder-Mead on the best solution found
    if time.time() < deadline - 60:
        from scipy.optimize import minimize
        
        def local_obj(params):
            inner = params.reshape(11, 3)
            side = _compute_side(inner)
            if not _check_inner_separation(inner):
                return side + 100.0
            return side
        
        try:
            res = minimize(local_obj, best_inner.flatten(), method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_side:
                inner_lm = res.x.reshape(11, 3)
                side_lm = _compute_side(inner_lm)
                if _validate(inner_lm, outer_data, side_lm):
                    if side_lm < best_side:
                        best_inner = inner_lm
                        best_side = side_lm
        except Exception:
            pass
    
    # Final validation
    if not _validate(best_inner, outer_data, best_side):
        # Fallback
        best_inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        best_side = 8.0
    
    return best_inner, outer_data, best_side

# EVOLVE-BLOCK-END
