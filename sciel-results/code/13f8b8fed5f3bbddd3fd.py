import numpy as np
import itertools
import math

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute planes with >= 5 points
    max_coeff = 8
    seen_planes = set()
    planes = [] # List of lists of point indices
    point_to_planes = [[] for _ in range(N)]
    
    for a in range(-max_coeff, max_coeff + 1):
        for b in range(-max_coeff, max_coeff + 1):
            for c in range(-max_coeff, max_coeff + 1):
                if a == 0 and b == 0 and c == 0:
                    continue
                g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g > 0:
                    a, b, c = a//g, b//g, c//g
                # Normalize sign
                if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
                    a, b, c = -a, -b, -c
                
                # Calculate d for each point
                # d = ax + by + cz
                ds = grid @ np.array([a, b, c])
                unique_ds = np.unique(ds)
                for d in unique_ds:
                    mask = ds == d
                    pts = np.where(mask)[0]
                    if len(pts) >= 5:
                        plane_key = (a, b, c, d)
                        if plane_key not in seen_planes:
                            seen_planes.add(plane_key)
                            plane_idx = len(planes)
                            planes.append(pts)
                            for p in pts:
                                point_to_planes[p].append(plane_idx)

    # Precompute plane masks
    plane_masks = []
    for pts in planes:
        mask = 0
        for p in pts:
            mask |= (1 << p)
        plane_masks.append(mask)

    # Order points by degree (number of planes) descending
    degrees = [len(point_to_planes[i]) for i in range(N)]
    order = np.argsort(degrees)[::-1]

    best_size = 0
    best_mask = 0

    import sys
    sys.setrecursionlimit(10000)

    def solve(idx, selected_mask, selected_count):
        nonlocal best_size, best_mask
        
        # Pruning
        remaining = N - idx
        if selected_count + remaining <= best_size:
            return
            
        if idx == N:
            if selected_count > best_size:
                best_size = selected_count
                best_mask = selected_mask
            return
            
        p = order[idx]
        
        # Option 1: Skip
        solve(idx + 1, selected_mask, selected_count)
        
        # Option 2: Add
        # Check validity
        valid = True
        for plane_idx in point_to_planes[p]:
            inter = selected_mask & plane_masks[plane_idx]
            if bin(inter).count('1') >= 4:
                valid = False
                break
                
        if valid:
            solve(idx + 1, selected_mask | (1 << p), selected_count + 1)

    solve(0, 0, 0)

    # Reconstruct points
    best_points = [i for i in range(N) if (best_mask >> i) & 1]
    
    # Verify cospherical constraint and fix if necessary
    def is_cospherical(p1, p2, p3, p4, p5):
        pts = np.array([p1, p2, p3, p4, p5], dtype=np.float64)
        M = np.zeros((5, 5))
        M[:, 0:3] = pts
        M[:, 3] = (pts ** 2).sum(axis=1)
        M[:, 4] = 1
        return abs(np.linalg.det(M)) < 1e-9
        
    def check_and_fix(points):
        pts = set(points)
        while True:
            found = False
            for combo in itertools.combinations(pts, 5):
                if is_cospherical(*combo):
                    pts.remove(combo[-1])
                    found = True
                    break
            if not found:
                break
        return list(pts)
        
    final_points = check_and_fix(best_points)
    
    return grid[final_points].astype(np.int64)

no_5_on_a_sphere8()
