import numpy as np
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 15.0
_lock = threading.Lock()

def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN

def _validate_positions(positions):
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Vectorized distance check
    dists = np.linalg.norm(positions[:, np.newaxis] - positions[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.35:
        return False
    return True

def _generate_icosahedron_points(n):
    """
    Generate points for an icosahedral cluster.
    For n=13, it's perfect. For others, we add points in shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    points.extend(list(verts))
    
    # If we need more, add 2nd shell (face centers and edge midpoints)
    if n > 13:
        # 20 faces. Face centers.
        # We need the faces. Hardcoding is tedious, so we approximate with a dense sphere.
        # Better: Use a known set of 55 points for Mackay icosahedron if n=55.
        # For general n, we'll use a "spherical shell" addition that respects symmetry roughly.
        
        # Let's just use a Fibonacci sphere for the outer shell if n > 13
        # This breaks symmetry but is a good start for relaxation.
        
        # Actually, let's try to generate the 55-atom Mackay icosahedron explicitly if n=55.
        if n == 55:
            # 1st shell: 12 (radius 1)
            # 2nd shell: 42 (radius ~ 1.9? No, Mackay ratio is such that 2nd shell is at sqrt(5+2*phi)? No.)
            # In Mackay icosahedron, the 2nd shell is at distance sqrt(5+2*phi) * r1? 
            # Actually, the 2nd shell consists of 12 face centers and 30 edge midpoints.
            # Face centers are at distance sqrt(5+2*phi)/phi? 
            # Let's just generate 42 points on a sphere of radius R2.
            # R2 is chosen such that the distance to 1st shell is reasonable.
            
            R2 = 1.9 # Approximate
            N_outer = 42
            z = (2 * np.arange(N_outer) + 1) / N_outer
            r = np.sqrt(1 - z**2)
            phi_angle = np.arange(N_outer) * (np.pi * (3 - np.sqrt(5)))
            
            x = r * np.cos(phi_angle)
            y = r * np.sin(phi_angle)
            
            outer_pts = np.column_stack([x, y, z]) * R2
            points.extend(list(outer_pts))
            
    # Scale to n
    if len(points) < n:
        # Add random points on a larger sphere
        deficit = n - len(points)
        R3 = 2.5
        z = (2 * np.arange(deficit) + 1) / deficit
        r = np.sqrt(1 - z**2)
        phi_angle = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
        x = r * np.cos(phi_angle)
        y = r * np.sin(phi_angle)
        outer_pts = np.column_stack([x, y, z]) * R3
        points.extend(list(outer_pts))
        
    if len(points) > n:
        points = points[:n]
        
    pts = np.array(points)
    # Center
    pts -= pts.mean(axis=0)
    # Scale to unit radius roughly? No, let relax handle scale.
    # But we want a compact start.
    norm = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norm)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm) # Scale to radius 3
        
    return pts

def _generate_fcc_sphere(n):
    """
    Generate an FCC-like sphere.
    """
    a = 1.0
    pts = []
    # Generate a cube of FCC points
    L = int(np.cbrt(n / 2.0)) + 2
    for i in range(L):
        for j in range(L):
            for k in range(L):
                # FCC positions in unit cell
                # (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    pts.append([x, y, z])
                    
    pts = np.array(pts)
    # Center
    center = pts.mean(axis=0)
    pts -= center
    
    # Keep closest n points
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    # Scale to radius 3
    norm = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norm)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
        
    return pts

def _generate_bcc_sphere(n):
    """
    Generate a BCC-like sphere.
    """
    a = 1.0
    pts = []
    L = int(np.cbrt(n / 2.0)) + 2
    for i in range(L):
        for j in range(L):
            for k in range(L):
                # BCC: (0,0,0) and (0.5, 0.5, 0.5)
                pts.append([i*a, j*a, k*a])
                if i < L-1 and j < L-1 and k < L-1:
                    pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
                    
    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    
    norm = np.linalg.norm(pts, axis=1)
    max_norm = np.max(norm)
    if max_norm > 0:
        pts = pts * (3.0 / max_norm)
        
    return pts

def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    seed = n * 100 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)
    
    best_energy = np.inf
    best_positions = None
    
    # Initial diverse starts
    starts = []
    
    # 1. Icosahedral
    ico = _generate_icosahedron_points(n)
    if _validate_positions(ico):
        starts.append(ico)
        
    # 2. FCC
    fcc = _generate_fcc_sphere(n)
    if _validate_positions(fcc):
        starts.append(fcc)
        
    # 3. BCC
    bcc = _generate_bcc_sphere(n)
    if _validate_positions(bcc):
        starts.append(bcc)
        
    # 4. Random Compact (Fallback/Extra diversity)
    for _ in range(2):
        R = 3.0
        pts = rng.uniform(-R, R, (n, 3))
        # Simple rejection to ensure compactness? No, just random.
        # Actually, let's do a quick "greedy" compact placement
        pts = np.zeros((n, 3))
        pts[0] = 0
        for i in range(1, n):
            # Place near existing
            idx = rng.integers(0, i)
            offset = rng.normal(0, 0.5, 3)
            pts[i] = pts[idx] + offset
        if _validate_positions(pts):
            starts.append(pts)

    # Relax initial starts in parallel
    def _relax_one(pos):
        try:
            energy, positions = relax(pos, 'sutton_chen', kind)
            if _validate_positions(positions):
                return energy, positions
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_one, start): start for start in starts}
        for future in as_completed(futures):
            if _get_time_left() < 5.0:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    with _lock:
                        best_energy = energy
                        best_positions = positions.copy()
            except Exception:
                pass

    # Basin Hopping
    # We will perform a series of kicks on the best structure found.
    # Each kick creates a perturbed structure, which is then relaxed.
    
    num_kicks = 200 # Max kicks
    kick_amplitudes = [0.1, 0.2, 0.5, 1.0]
    
    for i in range(num_kicks):
        if _get_time_left() < 10.0:
            break
            
        if best_positions is None:
            break
            
        # Choose a kick amplitude
        amp = kick_amplitudes[i % len(kick_amplitudes)]
        
        # Generate perturbed structure
        # Gaussian noise
        noise = rng.normal(0, amp, best_positions.shape)
        perturbed = best_positions + noise
        
        # Check validity before relaxing
        if not _validate_positions(perturbed):
            # Try a smaller kick
            noise = rng.normal(0, amp * 0.5, best_positions.shape)
            perturbed = best_positions + noise
            if not _validate_positions(perturbed):
                continue
                
        # Relax
        try:
            energy, positions = relax(perturbed, 'sutton_chen', kind)
            if _validate_positions(positions):
                if energy < best_energy:
                    with _lock:
                        best_energy = energy
                        best_positions = positions.copy()
        except Exception:
            pass

    # Final validation
    if best_positions is None:
        # Fallback to first start if something went wrong
        best_positions = starts[0]
        
    return best_positions
