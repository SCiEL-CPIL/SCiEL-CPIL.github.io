<<<<<<< SEARCH
    rng = np.random.RandomState(42)
    
    inits = []
    
    # Random inits
    for _ in range(5):
        inits.append(rng.randn(28) * 2.0)
        
    # Hexagonal init
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    d_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(d_center)[:16]
    hex_init = hex_pts[idx]
    hex_init -= np.mean(hex_init, axis=0)
    
    # Try to convert hex_init to params
    # We need to pick two points to be P0 and P1.
    # Let's pick the two closest points in hex_init as P0 and P1.
    dists_hex = spd.pdist(hex_init)
    # Find min distance pair
    min_idx = np.argmin(dists_hex)
    # Convert linear index to (i, j)
    # 16 points -> 120 pairs
    # This is tedious. Let's just pick P0=hex_init[0], P1=hex_init[1] and hope they are close.
    # Or better, sort by x, then y.
    hex_sorted = hex_init[np.lexsort((hex_init[:, 1], hex_init[:, 0]))]
    p0_hex = hex_sorted[0]
    p1_hex = hex_sorted[1]
    
    # Construct a candidate with P0, P1 fixed
    cand = hex_sorted.copy()
    cand[0] = p0_hex
    cand[1] = p1_hex
    # Translate and rotate
    p0 = cand[0]
    p1 = cand[1]
    cand_trans = cand - p0
    d = np.linalg.norm(p1)
    if d > 1e-9:
        angle = np.arctan2(p1[1], p1[0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        cand_rot = cand_trans @ rot.T
        cand_scaled = cand_rot / d
        params_hex = cand_scaled[2:].flatten()
        inits.append(params_hex)

    # Ring init
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    r = 2.0
    circle_init = np.column_stack((r*np.cos(angles), r*np.sin(angles))).flatten()
    inits.append(circle_init)
=======
    rng = np.random.RandomState(42)
    
    inits = []
    
    # Random inits
    for _ in range(3):
        inits.append(rng.randn(28) * 2.0)
        
    def make_params_from_points(pts):
        """
        Takes a 16x2 array, finds the closest pair, sets them as P0(0,0) and P1(1,0),
        and returns the 28-dim param vector.
        """
        # Find closest pair
        dists = spd.pdist(pts)
        min_idx = np.argmin(dists)
        # Convert linear index to (i, j)
        # n=16, pairs = 120
        # i = min_idx // 15 ? No.
        # Pairs are (0,1), (0,2)...(0,15), (1,2)...
        # Index k corresponds to pair (i, j) where:
        # i = (n-1) - int(sqrt(2*k + 0.25) + 0.5)
        # j = i + 1 + int(k - (n-1)*i + i*(i-1)/2)
        # Let's use a simpler lookup or just brute force for 16 points
        n_pts = 16
        i = 0
        j = 0
        count = 0
        for ii in range(n_pts - 1):
            for jj in range(ii + 1, n_pts):
                if count == min_idx:
                    i = ii
                    j = jj
                    break
                count += 1
            if count == min_idx:
                break
        
        # Move point i to 0, point j to 1
        new_pts = pts.copy()
        # Swap i to 0
        if i != 0:
            new_pts[[0, i]] = new_pts[[i, 0]]
            if j != 0:
                j = i if j == 0 else (0 if j == i else j) # This logic is messy.
        
        # Easier: just reorder
        order = [0] * 16
        order[0] = i
        order[1] = j
        used = {i, j}
        for k in range(2, 16):
            for m in range(16):
                if m not in used:
                    order[k] = m
                    used.add(m)
                    break
        
        new_pts = pts[order]
        
        # Now new_pts[0] and new_pts[1] are the closest pair.
        # Translate new_pts[0] to (0,0)
        new_pts -= new_pts[0]
        # Rotate so new_pts[1] is on x-axis
        d = np.linalg.norm(new_pts[1])
        if d < 1e-9:
            return None
        angle = np.arctan2(new_pts[1][1], new_pts[1][0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        new_pts = new_pts @ rot.T
        # Scale so new_pts[1] is (1,0)
        new_pts = new_pts / d
        
        # Verify
        if not np.allclose(new_pts[0], [0,0]) or not np.allclose(new_pts[1], [1,0]):
             return None
             
        return new_pts[2:].flatten()

    # 1. Hexagonal Lattice Init
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Center the cluster
    hex_pts -= np.mean(hex_pts, axis=0)
    # Select 16 closest to center
    d_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(d_center)[:16]
    hex_init = hex_pts[idx]
    
    p_hex = make_params_from_points(hex_init)
    if p_hex is not None:
        inits.append(p_hex)
        
    # 2. Circle Init (16 points on a circle)
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    p_circ = make_params_from_points(circle_pts)
    if p_circ is not None:
        inits.append(p_circ)
        
    # 3. Square Grid Init (4x4)
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i, j])
    grid_pts = np.array(grid_pts)
    grid_pts -= np.mean(grid_pts, axis=0)
    p_grid = make_params_from_points(grid_pts)
    if p_grid is not None:
        inits.append(p_grid)
>>>>>>> REPLACE

