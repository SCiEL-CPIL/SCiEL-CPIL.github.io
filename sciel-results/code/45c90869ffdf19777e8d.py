import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    
    # Quaternion helpers
    def q_multiply(q1, q2):
        """Hamiltonian product of two quaternions (w, x, y, z)"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def q_conjugate(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def q_norm(q):
        return np.sqrt(np.sum(q**2))
    
    def q_normalize(q):
        n = q_norm(q)
        if n < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n
    
    def q_to_point(q):
        """Extract 3D point from quaternion (vector part normalized)"""
        vec = q[1:4]
        n = np.linalg.norm(vec)
        if n < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return vec / n
    
    def point_to_q(p):
        """Convert 3D unit point to quaternion (w=0, vector part = point)"""
        return np.array([0.0, p[0], p[1], p[2]])
    
    def rotation_q(axis, angle):
        """Create quaternion for rotation by angle around axis"""
        axis = axis / np.linalg.norm(axis)
        return np.array([
            np.cos(angle/2),
            axis[0]*np.sin(angle/2),
            axis[1]*np.sin(angle/2),
            axis[2]*np.sin(angle/2)
        ])
    
    # Initialize with icosahedron + 2 points, converted to quaternions
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p13 = np.array([1.0, 0.0, 0.0])
    p14 = np.array([-1.0, 0.0, 0.0])
    initial_points = np.vstack([ico, p13, p14])
    
    # Convert to quaternions
    quats = np.array([point_to_q(p) for p in initial_points])
    
    def get_points_from_quats(qs):
        pts = np.array([q_to_point(q) for q in qs])
        return pts
    
    def min_max_ratio(pts):
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2
    
    # Optimize using quaternion-based perturbations
    np.random.seed(42)
    best_score = min_max_ratio(get_points_from_quats(quats))
    quats = quats.copy()
    
    for iteration in range(5000):
        improved = False
        for i in range(n):
            orig_q = quats[i].copy()
            best_q = orig_q
            best_local_score = min_max_ratio(get_points_from_quats(quats))
            
            # Try random small rotations
            for _ in range(15):
                # Random axis and small angle
                axis = np.random.randn(3)
                angle = np.random.uniform(0.01, 0.2)
                rot_q = rotation_q(axis, angle)
                
                # Apply rotation: new_q = rot_q * orig_q
                new_q = q_multiply(rot_q, orig_q)
                new_q = q_normalize(new_q)
                
                quats[i] = new_q
                score = min_max_ratio(get_points_from_quats(quats))
                
                if score > best_local_score:
                    best_local_score = score
                    best_q = new_q
                    improved = True
            
            quats[i] = best_q
        
        if not improved:
            break
    
    # Final refinement with smaller steps
    for _ in range(2000):
        improved = False
        for i in range(n):
            orig_q = quats[i].copy()
            best_q = orig_q
            best_local_score = min_max_ratio(get_points_from_quats(quats))
            
            for _ in range(10):
                axis = np.random.randn(3)
                angle = np.random.uniform(0.005, 0.05)
                rot_q = rotation_q(axis, angle)
                new_q = q_multiply(rot_q, orig_q)
                new_q = q_normalize(new_q)
                
                quats[i] = new_q
                score = min_max_ratio(get_points_from_quats(quats))
                
                if score > best_local_score:
                    best_local_score = score
                    best_q = new_q
                    improved = True
            
            quats[i] = best_q
        
        if not improved:
            break
    
    return get_points_from_quats(quats)
