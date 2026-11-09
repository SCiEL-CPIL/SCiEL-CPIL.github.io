import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    def get_hex_init(r):
        # Generate hex grid
        pts = []
        # Scale factor for hex grid
        # Distance between centers in hex grid is 2r
        # Row height is 2r * sqrt(3)/2 = r*sqrt(3)
        # Offset for odd rows is r
        # We need enough points to cover the unit disk
        # Max radius 1.
        # Number of rows ~ 1 / (r*sqrt(3))
        max_r = 1.0
        row_h = r * np.sqrt(3)
        num_rows = int(max_r / row_h) + 2
        for j in range(-num_rows, num_rows + 1):
            y = j * row_h
            offset = r if j % 2 != 0 else 0
            num_cols = int((max_r - abs(y)) / (2*r)) + 2 # Approx
            for i in range(-num_cols, num_cols + 1):
                x = i * 2*r + offset
                if x*x + y*y <= (1.0)**2: # Rough check
                    pts.append([x, y])
        pts = np.array(pts)
        # Select N closest to center
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)
        return pts[idx[:N]]

    def relax(points, r, max_iters=5000):
        pts = points.copy()
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            # Compute pairwise differences
            # Shape (N, N, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Mask for pairs i < j
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            dist_masked = dist[mask]
            diff_masked = diff[mask]
            
            # Find violations
            violations = dist_masked < 2*r
            if np.any(violations):
                # Get indices
                # We need to apply forces
                # Force magnitude: (2r - dist) / 2
                # Direction: normalized diff
                # dist_masked[violations] might be 0
                d_v = dist_masked[violations]
                d_v[d_v == 0] = 1e-6 # Avoid div by 0
                force_mag = (2*r - d_v) / (2 * d_v)
                
                # Apply to pts
                # We need to map back to original indices
                # This is tricky with boolean masking in a single vectorized step for updates
                # It's easier to loop over the violating pairs if there are few, 
                # or use a loop if N is small.
                # For N=50, a loop over 1225 pairs is fast enough.
                # Let's stick to the loop for clarity and safety, but optimize the inner part.
                pass
            
            # Actually, a simple loop is safer and likely fast enough for N=50.
            # 1225 pairs * 5000 iters = 6M checks. Python can do ~10-50M simple ops/sec.
            # So 6M ops might take 0.1-0.5s.
            # This allows ~200-1000 relaxations per second.
            # Binary search takes ~10-20 steps.
            # Total time < 10s.
            
            # Let's use the loop version inside relax for correctness.
            
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 2*r:
                        if dist < 1e-9:
                            # Random kick
                            d = np.array([np.random.randn(2)])
                            dist = np.linalg.norm(d)
                        push = (2*r - dist) / (2*dist)
                        pts[i] += push * d
                        pts[j] -= push * d
                        moved = True
            
            for i in range(N):
                dist_c = np.linalg.norm(pts[i])
                if dist_c > 1 - r:
                    if dist_c < 1e-9:
                        continue
                    pts[i] = pts[i] * ((1 - r) / dist_c)
                    moved = True
            
            if not moved:
                break
                
        # Check final validity
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                min_dist = min(min_dist, np.linalg.norm(pts[i]-pts[j]))
            min_dist = min(min_dist, 1 - np.linalg.norm(pts[i]))
        
        # The score is min_dist / 2 ? No.
        # Score = min( min_pair_dist / 2, min_boundary_dist )
        # If we satisfied constraints for r, then min_pair_dist >= 2r and min_boundary_dist >= r.
        # So score >= r.
        return pts, min_dist / 2 # This is the actual radius achieved

    best_r = 0.0
    best_pts = None
    
    # Binary search for max r
    lo = 0.1
    hi = 0.126
    
    # Initializations
    inits = []
    # 1. Hex
    inits.append(get_hex_init(0.125))
    # 2. Random
    inits.append(np.random.randn(N, 2) * 0.5)
    # 3. Ring structure
    ring_pts = []
    ring_pts.append([0,0])
    for k, count in enumerate([6, 12, 18, 13]):
        # Rough radii
        r_k = (k+1) * 0.2 
        for m in range(count):
            angle = 2*np.pi*m/count
            ring_pts.append([r_k*np.cos(angle), r_k*np.sin(angle)])
    inits.append(np.array(ring_pts))
    
    # Shuffle inits
    np.random.shuffle(inits)
    
    while lo < hi - 1e-6:
        if time.time() - start_time > time_limit - 10:
            break
        mid = (lo + hi) / 2
        
        success = False
        for init in inits:
            if time.time() - start_time > time_limit - 5:
                break
            # Scale init to fit roughly
            # The relax function handles the geometry, but starting scale matters.
            # If init points are too far apart, relax might not converge to the dense state.
            # If too close, it might get stuck.
            # The hex init is scaled to r=0.125. 
            # We should probably rescale the init to match the target r roughly.
            # But relax pushes them to exactly 2r distance.
            # So starting scale doesn't matter as much as topology.
            
            pts, achieved_r = relax(init, mid)
            if achieved_r >= mid - 1e-6:
                success = True
                best_r = mid
                best_pts = pts
                break
        
        if success:
            lo = mid
        else:
            hi = mid
            
    # Final check
    if best_pts is None:
        # Fallback to hex
        pts, _ = relax(get_hex_init(0.125), 0.125)
        best_pts = pts
        best_r = 0.125
        
    return best_pts
