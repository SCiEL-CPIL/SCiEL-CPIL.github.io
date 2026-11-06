import numpy as np
import time
import sys

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30, 30]^2
    to maximize the largest multiplicity of a single positive doubled area.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s limit
    
    # 1. Fast Vectorized Scorer
    def fast_score(pts):
        """
        pts: numpy array of shape (30, 2) with integer coordinates.
        Returns: max multiplicity of any positive doubled area.
        """
        # Compute cross products for all pairs (i, j) with i < j
        # We want area of triangle (i, j, k).
        # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
        # Let v_ij = p_j - p_i.
        # The term inside abs is det(v_ij, p_k - p_i).
        
        # Efficient way:
        # For each pair (i, j), compute v = p[j] - p[i].
        # Then for all k > j (or all k != i, j), compute cross(v, p[k] - p[i]).
        
        # Since N=30 is small, we can do a full O(N^3) vectorized approach 
        # or a smarter O(N^2) pair-iteration with vectorized k-loop.
        # O(N^3) vectorized:
        # p: (30, 2)
        # p_i: (30, 1, 2)
        # p_j: (1, 30, 2)
        # p_k: (1, 1, 30, 2) -- too big memory? 30*30*30*2*8 = 432KB. Fine.
        
        # Actually, simpler:
        # Iterate i from 0 to 27.
        # For each i, compute vectors from i to all j > i.
        # Then for each such vector, compute cross with all k > j.
        
        # Even faster for N=30:
        # Just compute all pairwise differences once? No, area depends on triple.
        
        # Let's use the broadcasting trick for all triples at once.
        # p_i = pts[:, np.newaxis, np.newaxis, :] # (30, 1, 1, 2)
        # p_j = pts[np.newaxis, :, np.newaxis, :] # (1, 30, 1, 2)
        # p_k = pts[np.newaxis, np.newaxis, :, :] # (1, 1, 30, 2)
        
        # v_ij = p_j - p_i  # (30, 30, 2)
        # v_ik = p_k - p_i  # (30, 30, 2)
        # det = v_ij[..., 0] * v_ik[..., 1] - v_ij[..., 1] * v_ik[..., 0]
        # We need i < j < k to avoid duplicates and self-loops.
        
        # Memory: 30x30x30x2 floats = 432KB. Very fast.
        
        p = pts
        # Expand dimensions for broadcasting
        pi = p[:, np.newaxis, np.newaxis, :]  # (30, 1, 1, 2)
        pj = p[np.newaxis, :, np.newaxis, :]  # (1, 30, 1, 2)
        pk = p[np.newaxis, np.newaxis, :, :]  # (1, 1, 30, 2)
        
        # Compute vectors
        vij = pj - pi  # (30, 30, 2)
        vik = pk - pi  # (30, 30, 2)
        
        # Compute determinant (doubled area * 2? No, det is 2*Area)
        # det = vij_x * vik_y - vij_y * vik_x
        det = vij[..., 0] * vik[..., 1] - vij[..., 1] * vik[..., 0]
        
        # We only care about triples (i, j, k) with i < j < k
        # Create a mask for i < j < k
        # i index is axis 0, j is axis 1, k is axis 2
        # We need i < j AND j < k
        
        # Generate indices
        i_idx, j_idx, k_idx = np.indices((30, 30, 30))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        
        # Extract valid determinants
        valid_dets = det[mask]
        
        # We want absolute values, exclude 0
        abs_dets = np.abs(valid_dets)
        abs_dets = abs_dets[abs_dets > 0]
        
        if len(abs_dets) == 0:
            return 0
            
        # Find max multiplicity
        # Since values are integers, we can use np.unique
        values, counts = np.unique(abs_dets, return_counts=True)
        
        return int(np.max(counts))

    # 2. Initialization
    # Start with a strong baseline: 15x2 grid
    # Points: (x, 0) for x in 0..14, (x, 1) for x in 0..14
    # Centered in [-30, 30]
    base_points = []
    for x in range(15):
        base_points.append((x - 7, 0))
        base_points.append((x - 7, 1))
    
    current_pts = np.array(base_points, dtype=np.int64)
    current_score = fast_score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Pre-compute all valid points in [-30, 30]^2
    # 61 * 61 = 3721 points
    xs = np.arange(-30, 31)
    ys = np.arange(-30, 31)
    all_points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    
    # 3. Simulated Annealing
    # Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1.0
    
    # Helper to check if a point is already in the set
    # For N=30, a set of tuples is fast enough for the check during move generation
    # But we need to update it.
    # Let's maintain a set of tuples for O(1) lookup.
    current_set = set(map(tuple, current_pts))
    
    # Main loop
    while time.time() < deadline and T > min_T:
        # Generate a move
        # Strategy: Pick a random index i, and a random new point p_new.
        # If p_new is in current_set, skip.
        # Otherwise, swap.
        
        # To ensure diversity, occasionally swap two points?
        # Single point jump is usually sufficient for this density.
        
        idx = np.random.randint(0, 30)
        old_pt = current_pts[idx]
        
        # Pick a random candidate from all_points
        # We can just pick a random index into all_points
        new_pt = all_points[np.random.randint(0, len(all_points))]
        
        # Check if new_pt is already in the set
        if tuple(new_pt) in current_set:
            continue
            
        # Apply move
        current_pts[idx] = new_pt
        current_set.remove(tuple(old_pt))
        current_set.add(tuple(new_pt))
        
        # Score
        new_score = fast_score(current_pts)
        
        # Acceptance
        if new_score > current_score:
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            # Reject or accept with probability exp(-delta/T)
            delta = current_score - new_score
            if delta == 0:
                # Tie: accept to explore
                pass # Keep new state
            else:
                prob = np.exp(-delta / T)
                if np.random.random() > prob:
                    # Revert
                    current_pts[idx] = old_pt
                    current_set.remove(tuple(new_pt))
                    current_set.add(tuple(old_pt))
        
        # Cool down
        T *= cooling_rate
        
        # If we are stuck at the same score for a long time, maybe restart?
        # Or just let it cool.
        
    # Return best points found
    return [tuple(p) for p in best_pts]

# Wrapper to ensure correct return type if needed, though construct returns list of tuples
if __name__ == "__main__":
    pts = construct()
    print(f"Score: {fast_score(np.array(pts))}")
    print(f"Points: {len(pts)}")
