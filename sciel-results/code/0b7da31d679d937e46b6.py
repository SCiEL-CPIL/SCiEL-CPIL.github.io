# EVOLVE-BLOCK-START
import numpy as np
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Optimizes a convex 20-gon to maximize the minimum vertex score.
    The vertex score is based on how equidistant the 4 nearest neighbors are.
    We start with a regular polygon and perform a local search on the radial distances
    of the vertices to improve the equidistance of the nearest neighbors.
    """
    n = 20
    # Fixed angles for the vertices
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial radii (regular polygon)
    radii = np.ones(n)
    
    def get_polygon(radii):
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    def vertex_score(polygon, v_idx):
        """
        Calculate the score for a single vertex.
        Returns 0.0 if invalid or score is 0.
        """
        v = polygon[v_idx]
        # Distances to all other vertices
        dists = np.linalg.norm(polygon - v, axis=1)
        # Exclude self
        dists[v_idx] = np.inf
        # Sort distances
        d = np.sort(dists)
        
        # Check for repeated points (distance < 0.05)
        if d[0] < 0.05:
            return 0.0
            
        # Find k: first index minimizing d_{k+1} - d_k
        gaps = d[1:] - d[:-1]
        # We look for the minimum gap. 
        # Note: The problem says "first index minimizing".
        k = np.argmin(gaps)
        
        # Check D
        D = (d[k] + d[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        # Select 4 distances
        if k == 0:
            d_sel = d[0:4]
        elif k == 17: # 19-1-1? No, indices 0..18. k max is 18? 
            # d has 19 elements. gaps has 18 elements. k in 0..17.
            # If k=17, d[k]=d[17], d[k+1]=d[18].
            # "d_15..d_18 if k=17"
            d_sel = d[15:19]
        else:
            # d_{k-1}..d_{k+2}
            d_sel = d[k-1:k+3]
            
        # Calculate raw score
        # max(d/D, D/d)
        ratios = np.maximum(d_sel / D, D / d_sel)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        return score

    def total_score(polygon):
        """
        Calculate the minimum vertex score over all vertices.
        Also checks for convexity and validity.
        """
        # Check for repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(polygon[i] - polygon[j]) < 0.05:
                    return 0.0
                    
        # Check convexity
        # Cross products of consecutive edges should have the same sign and be non-zero
        cross_signs = []
        for i in range(n):
            p0 = polygon[i]
            p1 = polygon[(i+1)%n]
            p2 = polygon[(i+2)%n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross == 0:
                return 0.0
            cross_signs.append(np.sign(cross))
            
        if len(set(cross_signs)) != 1:
            return 0.0
            
        # Calculate min vertex score
        min_score = 1.0
        for i in range(n):
            s = vertex_score(polygon, i)
            if s < min_score:
                min_score = s
                if min_score == 0.0:
                    return 0.0
        return min_score

    # Initial polygon
    best_polygon = get_polygon(radii)
    best_score = total_score(best_polygon)
    
    # Optimization loop
    start_time = time.time()
    deadline = 600.0 # seconds
    # Keep a safety margin
    while time.time() < deadline - 10.0:
        # Perturb radii
        # Pick a few random vertices to adjust
        num_perturb = np.random.randint(1, 4)
        idxs = np.random.choice(n, num_perturb, replace=False)
        
        new_radii = radii.copy()
        for idx in idxs:
            # Small perturbation
            delta = np.random.uniform(-0.1, 0.1)
            new_radii[idx] += delta
            # Ensure radius is positive
            if new_radii[idx] < 0.1:
                new_radii[idx] = 0.1 + np.random.uniform(0, 0.1)
                
        new_polygon = get_polygon(new_radii)
        new_score = total_score(new_polygon)
        
        if new_score > best_score:
            best_score = new_score
            best_polygon = new_polygon
            radii = new_radii
        else:
            # If score is 0, it's invalid, so revert
            pass

    return best_polygon

# EVOLVE-BLOCK-END
