# EVOLVE-BLOCK-START
import numpy as np
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized for the equidistance scoring metric.
    Uses a radial function r(theta) = 1 + sum of Fourier terms to create
    distance distributions that maximize the minimum vertex score.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin
    
    def compute_scores(polygon):
        """Compute the minimum vertex score for a given polygon."""
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(polygon, axis=0)
        centered = polygon - centroid
        dists_to_centroid = np.linalg.norm(centered, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        normalized = centered / mean_dist
        
        # Check strict convexity
        # No repeated points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                    return 0.0
        
        # Check turns
        signs = []
        total_turn = 0.0
        for i in range(n):
            p0 = normalized[i]
            p1 = normalized[(i+1) % n]
            p2 = normalized[(i+2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(cross) < 1e-12:
                return 0.0
            signs.append(1 if cross > 0 else -1)
            total_turn += np.arctan2(cross, np.dot(v1, v2))
        
        if len(set(signs)) != 1:
            return 0.0
        if abs(abs(total_turn) - 2*np.pi) > 1e-6:
            return 0.0
        
        # Compute vertex scores
        min_score = 1.0
        for i in range(n):
            vi = normalized[i]
            # Distances to all other vertices
            diffs = normalized - vi
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf  # exclude self
            sorted_d = np.sort(dists)
            
            # Check if any < 0.05
            if sorted_d[0] < 0.05:
                min_score = 0.0
                break
            
            # Find k: first index minimizing d_{k+1} - d_k
            gaps = np.diff(sorted_d)
            k = np.argmin(gaps)
            D = (sorted_d[k] + sorted_d[k+1]) / 2.0
            if D < 0.1:
                min_score = 0.0
                break
            
            # Take 4 distances
            if k == 0:
                indices = [0, 1, 2, 3]
            elif k == 17:
                indices = [15, 16, 17, 18]
            else:
                indices = [k-1, k, k+1, k+2]
            
            selected_d = sorted_d[indices]
            ratios = np.maximum(selected_d / D, D / selected_d)
            raw = np.mean(ratios)
            vertex_score = min(1.0, 1.0 / raw)
            min_score = min(min_score, vertex_score)
        
        return min_score
    
    def make_polygon(params):
        """Create polygon from radial function parameters."""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        r = np.ones(n)
        for freq, amp in params:
            r += amp * np.cos(freq * angles)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)
    
    # Start with regular polygon
    best_params = []
    best_score = compute_scores(make_polygon(best_params))
    
    # Try single Fourier modes
    for freq in [2, 3, 4, 5, 6, 8, 10]:
        for eps in np.linspace(-0.15, 0.15, 15):
            if time.time() > deadline - 60:
                break
            params = [(freq, eps)]
            poly = make_polygon(params)
            score = compute_scores(poly)
            if score > best_score:
                best_score = score
                best_params = params
    
    # Try combinations of two modes
    for freq1 in [2, 3, 4, 5]:
        for freq2 in [4, 5, 6, 8]:
            if freq1 == freq2:
                continue
            for eps1 in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                for eps2 in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    if time.time() > deadline - 30:
                        break
                    params = [(freq1, eps1), (freq2, eps2)]
                    poly = make_polygon(params)
                    score = compute_scores(poly)
                    if score > best_score:
                        best_score = score
                        best_params = params
    
    # Fine-tune best parameters with local search
    if best_params and time.time() < deadline - 10:
        # Simple local refinement
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, (freq, amp) in enumerate(best_params):
                for delta in [-0.02, -0.01, 0.01, 0.02]:
                    new_params = list(best_params)
                    new_amp = amp + delta
                    new_params[i] = (freq, new_amp)
                    poly = make_polygon(new_params)
                    score = compute_scores(poly)
                    if score > best_score:
                        best_score = score
                        best_params = new_params
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Return the best polygon
    if best_params:
        return make_polygon(best_params)
    else:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)

# EVOLVE-BLOCK-END
