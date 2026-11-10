import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0
    
    # 1. Vectorized Scoring and Helper Functions
    def score_config(pts):
        """Compute the max multiplicity of any positive doubled area."""
        n = len(pts)
        if n < 3: return 0
        P = np.array(pts, dtype=np.int64)
        # Generate indices for all triples
        i = np.repeat(np.arange(n), n*(n-1)//2)
        j = np.repeat(np.arange(n)[:, None], n-1, axis=1)
        j = np.concatenate([j[i < k] for k in range(n) if i[i==k].size > 0]) # This is complex, let's use itertools for clarity in logic but vectorize the math
        
        # Simpler vectorized approach for n=30:
        # There are 4060 triples.
        # Let's just compute all areas in a vectorized block.
        # For speed, we can compute areas relative to a fixed point? No, need all triples.
        
        # Efficient way:
        # For each pair (i,j), the area of triangle (i,j,k) is 0.5 * |cross(Pj-Pi, Pk-Pi)|
        # Let's compute all cross products.
        
        # Create arrays of all unique pairs (i, j) with i < j
        pairs_i, pairs_j = np.triu_indices(n, k=1)
        v1 = P[pairs_j] - P[pairs_i] # shape (m, 2)
        
        # For each pair, compute cross product with all other points
        # cross(v1, Pk - Pi) = v1[0]*(Pk[1]-Pi[1]) - v1[1]*(Pk[0]-Pi[0])
        # = (v1[0]*Pk[1] - v1[1]*Pk[0]) - (v1[0]*Pi[1] - v1[1]*Pi[0])
        
        # v1 is (m, 2), P is (n, 2)
        # term1 = v1[:, 0][:, None] * P[1][None, :] - v1[:, 1][:, None] * P[0][None, :]
        # term2 = v1[:, 0] * P[pairs_i, 1] - v1[:, 1] * P[pairs_i, 0]
        
        # This results in a (m, n) matrix of doubled areas (signed)
        # We need absolute values, and exclude k == i or k == j
        
        m = len(pairs_i)
        # term1: (m, n)
        t1 = v1[:, 0][:, None] * P[1][None, :]
        t2 = v1[:, 1][:, None] * P[0][None, :]
        cross_vals = t1 - t2
        
        # term2: (m,)
        const = v1[:, 0] * P[pairs_i, 1] - v1[:, 1] * P[pairs_i, 0]
        
        # doubled_area = |cross_vals - const[:, None]|
        doubled_areas = np.abs(cross_vals - const[:, None])
        
        # Mask out k == i and k == j
        # For each row r (corresponding to pair i_r, j_r), 
        # col i_r and col j_r should be 0.
        # We can just set them to 0 and then filter out 0s.
        # Note: k can be any index 0..n-1.
        # The triangle is (i, j, k). If k==i or k==j, area is 0.
        # So we can just flatten, take abs, remove 0s, and count.
        
        flat_areas = doubled_areas.flatten()
        # The entries where k==i or k==j are exactly 0.
        # Are there other zero areas? Yes, collinear points.
        # We only care about positive areas.
        positive_areas = flat_areas[flat_areas > 0]
        
        if len(positive_areas) == 0:
            return 0, None, Counter()
        
        counts = Counter(positive_areas.tolist())
        max_mult = max(counts.values())
        best_area = [a for a, c in counts.items() if c == max_mult][0]
        return max_mult, best_area, counts

    def generate_candidates():
        """Generate diverse initial candidate pools."""
        candidates = []
        
        # 1. Rectangular Grids
        factors = [1, 2, 3, 5, 6, 10, 15, 30]
        for w in factors:
            if 30 % w != 0: continue
            h = 30 // w
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    # Check bounds
                    if (w-1)*dx > 60 or (h-1)*dy > 60: continue
                    ox = -((w-1)*dx)//2
                    oy = -((h-1)*dy)//2
                    pts = []
                    for i in range(w):
                        for j in range(h):
                            x = ox + i*dx
                            y = oy + j*dy
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                pts.append((x, y))
                    if len(pts) == 30:
                        candidates.append(pts)
        
        # 2. Random Clusters (to simulate "dense" regions which might share areas)
        random.seed(42)
        for _ in range(5):
            cx = random.randint(-20, 20)
            cy = random.randint(-20, 20)
            pts = set()
            while len(pts) < 30:
                x = cx + random.randint(-10, 10)
                y = cy + random.randint(-10, 10)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
            candidates.append(list(pts))
            
        # 3. Hexagonal Lattice subset
        pts = []
        for i in range(-15, 16):
            for j in range(-15, 16):
                x = i
                y = j + (i % 2)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) >= 30:
            random.shuffle(pts)
            candidates.append(pts[:30])
            
        return candidates

    # 2. Greedy Set-Cover / Combinatorial Selection
    def greedy_select(pool, target_area, n_select=30):
        """
        Select n_select points from pool to maximize the number of triples
        with area == target_area.
        """
        if len(pool) < n_select:
            return pool
        
        # Precompute areas for all triples in the pool?
        # If pool is large (e.g., 100 points), 100 choose 3 is 161k triples.
        # We can compute a matrix of "compatibility" or just use a heuristic.
        
        # Heuristic: Start with the subset of the pool that has the highest
        # local density of target_area triangles.
        
        # Let's try a simpler greedy:
        # 1. Pick a random starting point.
        # 2. At each step, pick the point from remaining pool that maximizes
        #    the number of new target_area triangles formed with existing points.
        
        # To make this efficient, we need to compute areas between new candidate
        # and existing points.
        
        current_set = []
        remaining = list(pool)
        
        # Initial point: pick one that has many neighbors forming target_area?
        # Just pick random to break symmetry, or the one with highest degree in "target graph".
        
        # Let's build a graph where edge (u,v) has weight 1 if there exists w in pool
        # such that area(u,v,w) == target_area.
        # This is expensive to build fully.
        
        # Alternative:
        # Sample 3 points from pool. If area == target_area, it's a good seed.
        # Let's just use a standard greedy:
        
        # To speed up, we restrict the pool to points "close" to a centroid?
        # Or just run greedy on the whole pool if pool < 100.
        
        if len(pool) > 150:
             # Reduce pool to top 100 points based on some heuristic
             # e.g. points that form many small area triangles with others
             # For now, just take first 150
             pool = pool[:150]
             
        P = np.array(pool, dtype=np.int64)
        n_pool = len(P)
        
        # Precompute all pairwise vectors?
        # Let's just do the greedy loop.
        
        selected_indices = []
        available_indices = list(range(n_pool))
        
        # Start with a random point
        first_idx = random.choice(available_indices)
        selected_indices.append(first_idx)
        available_indices.remove(first_idx)
        
        for _ in range(n_select - 1):
            if not available_indices: break
            
            best_idx = None
            best_gain = -1
            
            # For each candidate in available_indices, compute gain
            # Gain = number of pairs (i, j) in selected_indices such that
            # area(candidate, i, j) == target_area
            
            cand_idx = np.array(available_indices)
            cand_pts = P[cand_idx]
            
            # selected_pts is (k, 2)
            # We need to check all pairs in selected.
            # If k is small (early steps), this is fast.
            
            sel_idx = np.array(selected_indices)
            sel_pts = P[sel_idx]
            k = len(sel_pts)
            
            if k >= 2:
                # Generate all pairs in selected
                si, sj = np.triu_indices(k, k=1)
                # v1 = sel_pts[sj] - sel_pts[si]
                v1 = sel_pts[sj] - sel_pts[si]
                
                # For each candidate c, area(c, si, sj) = 0.5 * |cross(v1, c - sel_pts[si])|
                # cross(v1, c - Pi) = v1[0]*(c[1]-Pi[1]) - v1[1]*(c[0]-Pi[0])
                # = (v1[0]*c[1] - v1[1]*c[0]) - (v1[0]*Pi[1] - v1[1]*Pi[0])
                
                # v1 is (p, 2), c is (m, 2), Pi is (p, 2)
                # term_c = v1[:, 0][:, None] * c[1][None, :] - v1[:, 1][:, None] * c[0][None, :]
                # term_p = v1[:, 0] * sel_pts[si, 1] - v1[:, 1] * sel_pts[si, 0]
                
                # Result is (p, m) matrix of doubled areas
                tc1 = v1[:, 0][:, None] * cand_pts[1][None, :]
                tc2 = v1[:, 1][:, None] * cand_pts[0][None, :]
                cross_c = tc1 - tc2
                
                tp1 = v1[:, 0] * sel_pts[si, 1]
                tp2 = v1[:, 1] * sel_pts[si, 0]
                cross_p = tp1 - tp2
                
                areas = np.abs(cross_c - cross_p[:, None])
                
                # Count matches
                matches = (areas == target_area).sum(axis=0)
                
                best_local_idx = np.argmax(matches)
                best_gain = matches[best_local_idx]
                best_idx = available_indices[best_local_idx]
            else:
                # If k < 2, gain is 0 for all. Pick random.
                best_idx = random.choice(available_indices)
                
            selected_indices.append(best_idx)
            available_indices.remove(best_idx)
            
        return [P[i] for i in selected_indices]

    # 3. Main Loop
    candidates = generate_candidates()
    
    best_score = 0
    best_pts = None
    best_area = None
    
    # Evaluate initial candidates to find a good target area
    init_scores = []
    for c in candidates:
        s, a, _ = score_config(c)
        init_scores.append((s, a, c))
        
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Try to improve the top few candidates using the greedy method with their target area
    # Also try a few other common small areas
    
    target_areas_to_try = set()
    for s, a, c in init_scores[:5]:
        if a is not None:
            target_areas_to_try.add(a)
            
    # Add some small areas just in case
    for a in range(1, 10):
        target_areas_to_try.add(a)
        
    # For each candidate and target area, run greedy selection from a larger pool
    # The pool can be the candidate itself plus some random perturbations
    # Or better: generate a larger pool of "good" points.
    
    # Let's create a "super pool" from the best grid candidate
    if init_scores:
        best_init_pts = init_scores[0][2]
        # Expand this grid slightly to create a pool of ~100 points
        # Identify the grid parameters
        # Simple heuristic: take the bounding box of best_init_pts and fill it
        xs = [p[0] for p in best_init_pts]
        ys = [p[1] for p in best_init_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        pool = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pool.append((x, y))
                    
        # Ensure we have enough points in pool
        if len(pool) < 30:
             pool = best_init_pts
             
        # If pool is too big, downsample
        if len(pool) > 200:
            random.shuffle(pool)
            pool = pool[:200]
            
        for target_a in list(target_areas_to_try)[:10]: # Limit targets
            if time.time() > total_deadline - 5: break
            
            # Run greedy selection
            selected = greedy_select(pool, target_a, 30)
            
            # Evaluate
            s, a, _ = score_config(selected)
            if s > best_score:
                best_score = s
                best_pts = selected
                best_area = a
                
    # Fallback: return the best initial candidate if greedy didn't improve
    if best_pts is None:
        best_pts = init_scores[0][2] if init_scores else [(0,0)]
    elif best_score < init_scores[0][0]:
        best_pts = init_scores[0][2]
        
    # 4. Local Search Polish on the best result
    # Use the existing local search logic but faster with NumPy scoring
    def fast_local_search(pts, duration):
        current_pts = [p for p in pts]
        current_score, current_area, _ = score_config(current_pts)
        end_time = time.time() + duration
        no_improve = 0
        
        while time.time() < end_time:
            if no_improve > 50: break
            
            i = random.randint(0, 29)
            old_p = current_pts[i]
            
            # Try a few random moves
            improved = False
            for _ in range(5):
                nx = old_p[0] + random.randint(-3, 3)
                ny = old_p[1] + random.randint(-3, 3)
                new_p = (nx, ny)
                if -30 <= nx <= 30 and -30 <= ny <= 30 and new_p not in current_pts:
                    current_pts[i] = new_p
                    new_score, _, _ = score_config(current_pts)
                    if new_score > current_score:
                        current_score = new_score
                        no_improve = 0
                        improved = True
                        break
                    else:
                        current_pts[i] = old_p
            if not improved:
                no_improve += 1
                
        return current_score, current_pts

    if time.time() < total_deadline - 5:
        s, p = fast_local_search(best_pts, 10)
        if s > best_score:
            best_score = s
            best_pts = p
            
    return best_pts

# For debugging/verification
if __name__ == "__main__":
    pts = construct()
    print(f"Found {len(pts)} points")
    s, a, c = score_config(pts)
    print(f"Score: {s}, Area: {a}")
