# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def radii_to_pts(r):
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    best_score = 0.0
    best_radii = np.ones(n)
    
    def try_candidate(r):
        nonlocal best_score, best_radii
        pts = radii_to_pts(r)
        s = _score_polygon(pts)
        if s > best_score:
            best_score = s
            best_radii = r.copy()
            return True
        return False

    # Initial seeds
    # 1. Regular
    r_current = np.ones(n)
    try_candidate(r_current)
    
    # 2. Alternating
    r_alt = 1.0 + 0.1 * np.array([1, -1] * 10)
    try_candidate(r_alt)
    
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        r_rand = 1.0 + 0.2 * rng.normal(0, 1, n)
        r_rand = np.clip(r_rand, 0.5, 2.0)
        try_candidate(r_rand)

    # Simulated Annealing
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.999
    
    r_current = best_radii.copy()
    current_score = best_score
    
    no_improvement_count = 0
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor
        # Perturb 1 or 2 random radii
        num_perturb = 1 if rng.random() < 0.8 else 2
        indices = rng.choice(n, size=num_perturb, replace=False)
        r_new = r_current.copy()
        for idx in indices:
            r_new[idx] += rng.normal(0, T * 10) # Scale noise by T
            r_new[idx] = np.clip(r_new[idx], 0.5, 2.5)
            
        s_new = _score_polygon(radii_to_pts(r_new))
        
        # Acceptance criterion
        if s_new > current_score:
            # Always accept improvement
            r_current = r_new
            current_score = s_new
            no_improvement_count = 0
            last_improvement_time = time.time()
            if try_candidate(r_new):
                pass # Updated best
        else:
            # Metropolis criterion
            if s_new == 0.0:
                # Invalid, reject unless very hot (to explore?)
                # Actually, invalid scores 0. If current is > 0, delta is negative.
                # If current is 0, delta is 0.
                if current_score == 0.0 and rng.random() < 0.1:
                     r_current = r_new
                     current_score = s_new
                else:
                     no_improvement_count += 1
            else:
                delta = s_new - current_score
                # delta is negative
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    r_current = r_new
                    current_score = s_new
                    no_improvement_count += 1
                else:
                    no_improvement_count += 1
        
        # Cooling
        T *= cooling_rate
        
        # Stagnation check
        if no_improvement_count > 5000:
            # Reheat and kick
            T = max(T, 0.02)
            # Kick: large random perturbation
            kick_idx = rng.choice(n)
            r_current[kick_idx] += rng.uniform(-0.5, 0.5)
            r_current = np.clip(r_current, 0.5, 2.5)
            s_new = _score_polygon(radii_to_pts(r_current))
            current_score = s_new
            if try_candidate(r_current):
                no_improvement_count = 0
            else:
                no_improvement_count = 0 # Reset counter after kick
                # If kick made it worse, accept it anyway to escape local min?
                # Yes, SA should accept bad moves to escape.
                # But if it's invalid (0), and best is > 0, we might be stuck in 0 region.
                # Let's just reset counter and continue.
                
    # Final polish with local search
    if best_score > 0 and time.time() < deadline:
        current = best_radii.copy()
        step = 0.01
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    new_r = current.copy()
                    new_r[i] += delta
                    if new_r[i] < 0.5 or new_r[i] > 2.5:
                        continue
                    s_new = _score_polygon(radii_to_pts(new_r))
                    if s_new > best_score:
                        best_score = s_new
                        best_radii = new_r.copy()
                        current = new_r
                        improved = True
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break

    return radii_to_pts(best_radii)

# EVOLVE-BLOCK-END
