import time
import random
from itertools import combinations
from collections import Counter
import numpy as np

def _max_single_area(pts):
    """Max multiplicity of any positive doubled-area among all triples."""
    n = len(pts)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _compute_area_counter(pts):
    """Compute Counter of all positive doubled-areas."""
    n = len(pts)
    cnt = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if a > 0:
                    cnt[a] += 1
    return cnt

def _refined_local_search(start_pts, target, time_budget, rng):
    """
    Optimized local search using vectorized delta calculation and guided moves.
    """
    pts = list(start_pts)
    n = len(pts)
    
    # Initial full count to get baseline
    cnt = _compute_area_counter(pts)
    current_count = cnt.get(target, 0)
    best_count = current_count
    best_pts = pts[:]
    occupied = set(pts)
    
    # Precompute vectors from each point to all others? 
    # Actually, we need vectors relative to the point being moved.
    # Let's represent points as a list of tuples.
    
    end_time = time.time() + time_budget
    no_improve_iter = 0
    max_no_improve = 100
    
    while time.time() < end_time:
        if no_improve_iter > max_no_improve:
            break
            
        idx = rng.randrange(n)
        old_pos = pts[idx]
        old_x, old_y = old_pos
        
        # Prepare vectors from old_pos to all other points
        # v_j = P_j - P_old
        # Area(i, j, k) = |det(v_j, v_k)|
        # We need to count how many pairs (j, k) have |det(v_j, v_k)| == target
        
        # To speed up delta calculation for a new position P_new:
        # w_j = P_j - P_new
        # Area_new(i, j, k) = |det(w_j, w_k)|
        
        # Let's collect candidate positions.
        candidates = set()
        
        # 1. Guided Moves: 
        # Pick two other points j, k. 
        # We want P_new such that |det(P_j - P_new, P_k - P_new)| == target.
        # Let u = P_j - P_k.
        # det(P_j - P_new, P_k - P_new) = det(P_j - P_new, P_k - P_j + P_j - P_new)
        # = det(P_j - P_new, P_k - P_j) + det(P_j - P_new, P_j - P_new)
        # = det(P_j - P_new, P_k - P_j)
        # So we need det(P_j - P_new, u) = +/- target.
        # Let P_new = (x, y). P_j = (xj, yj). u = (ux, uy).
        # det((xj-x, yj-y), (ux, uy)) = (xj-x)uy - (yj-y)ux = xj*uy - x*uy - yj*ux + y*ux
        # = (xj*uy - yj*ux) - (x*uy - y*ux)
        # Let C = xj*uy - yj*ux.
        # We need C - (x*uy - y*ux) = +/- target
        # x*uy - y*ux = C +/- target
        # This is a linear Diophantine equation.
        
        # We'll sample a few pairs (j, k) to generate candidates.
        other_indices = [i for i in range(n) if i != idx]
        rng.shuffle(other_indices)
        num_pairs_to_try = 10
        for p in range(min(num_pairs_to_try, len(other_indices))):
            j = other_indices[p]
            k = other_indices[(p + 1) % len(other_indices)]
            if j == k: continue
            
            xj, yj = pts[j]
            xk, yk = pts[k]
            ux = xj - xk
            uy = yj - yk
            
            # If u is zero, skip (should not happen as points are distinct)
            if ux == 0 and uy == 0: continue
            
            # Equation: x*uy - y*ux = C +/- target
            # C = xj*uy - yj*ux
            C = xj * uy - yj * ux
            
            # We need to find integer solutions (x, y) in [-30, 30]
            # x*uy - y*ux = D, where D is C+target or C-target
            
            # General solution for x*uy - y*ux = D:
            # If gcd(ux, uy) does not divide D, no solution.
            g = np.gcd(abs(ux), abs(uy)) if (ux != 0 or uy != 0) else 1
            if g == 0: continue
            
            for sign in [1, -1]:
                D = C + sign * target
                if D % g != 0:
                    continue
                
                # Find a particular solution (x0, y0)
                # Use extended Euclidean or just brute force small range?
                # Since range is small [-30, 30], we can just iterate x and solve for y?
                # Or iterate y and solve for x?
                # x*uy - y*ux = D
                # If uy != 0: x = (D + y*ux) / uy. Must be integer.
                # If ux != 0: y = (x*uy - D) / ux. Must be integer.
                
                # Let's iterate x in [-30, 30] and check if y is integer and in range.
                # This is fast enough for 10 pairs * 2 signs * 61 x-values.
                for x in range(-30, 31):
                    if uy != 0:
                        num = D + x * uy # Wait, x*uy - y*ux = D => y*ux = x*uy - D => y = (x*uy - D)/ux
                        # Correction:
                        # x*uy - y*ux = D
                        # y*ux = x*uy - D
                        # If ux != 0:
                        if ux != 0:
                            num = x * uy - D
                            if num % ux == 0:
                                y = num // ux
                                if -30 <= y <= 30:
                                    p_new = (x, y)
                                    if p_new not in occupied and p_new != old_pos:
                                        candidates.add(p_new)
                        else:
                            # ux == 0, so uy != 0
                            # x*uy = D => x = D/uy
                            if D % uy == 0:
                                x_sol = D // uy
                                if -30 <= x_sol <= 30:
                                    # y can be anything? No, equation is x*uy = D.
                                    # y is free?
                                    # det((xj-x, yj-y), (0, uy)) = (xj-x)uy - 0 = (xj-x)uy = D
                                    # So x is fixed. y can be any integer?
                                    # Yes, if ux=0, the x-coordinate of the vector u is 0.
                                    # The determinant depends only on x.
                                    # So any y works as long as x is correct.
                                    # Let's pick a few random y's.
                                    for _ in range(3):
                                        y_rand = rng.randint(-30, 30)
                                        p_new = (x_sol, y_rand)
                                        if p_new not in occupied and p_new != old_pos:
                                            candidates.add(p_new)
                    else:
                        # uy == 0, so ux != 0
                        # -y*ux = D => y = -D/ux
                        if D % ux == 0:
                            y_sol = -D // ux
                            if -30 <= y_sol <= 30:
                                for _ in range(3):
                                    x_rand = rng.randint(-30, 30)
                                    p_new = (x_rand, y_sol)
                                    if p_new not in occupied and p_new != old_pos:
                                        candidates.add(p_new)
        
        # 2. Random Moves
        for _ in range(5):
            nx = rng.randint(-30, 30)
            ny = rng.randint(-30, 30)
            p = (nx, ny)
            if p not in occupied and p != old_pos:
                candidates.add(p)
        
        if not candidates:
            no_improve_iter += 1
            continue
            
        # Evaluate all candidates
        # We need to calculate delta for each candidate.
        # Delta = New_Count - Old_Count
        # Old_Count is constant for this idx.
        # New_Count = number of pairs (j, k) from others such that |det(P_j - P_new, P_k - P_new)| == target
        
        # Let's vectorize this.
        # Others: list of points excluding idx.
        others = [pts[i] for i in range(n) if i != idx]
        # M is (n-1) x 2 matrix
        M = np.array(others)
        
        # For each candidate P_new, we compute vectors V = M - P_new
        # Then we need to count pairs (a, b) in V such that |V_a[0]*V_b[1] - V_a[1]*V_b[0]| == target
        
        # This is still O(N^2) per candidate.
        # With ~20 candidates, and N=29, N^2=841. 20*841 = 16820 ops.
        # In Python loops this is slow.
        
        # Let's stick to a simpler optimization:
        # Just compute the count for the best few candidates?
        # Or compute delta for all candidates in a vectorized way?
        
        # Let's try to compute the "score" of a candidate P_new.
        # score(P_new) = sum_{j<k, j,k != idx} [ |det(P_j - P_new, P_k - P_new)| == target ]
        
        # We can compute the matrix of determinants for all pairs in 'others' relative to P_new.
        # But doing this for each candidate separately is O(C * N^2).
        
        # Alternative:
        # Precompute all pair determinants for 'others' relative to old_pos?
        # No, that doesn't help directly for new_pos.
        
        # Let's just iterate candidates and compute count.
        # To speed it up, we can use numpy to compute the determinant matrix for one candidate.
        
        best_candidate = None
        best_delta = 0
        
        # Calculate old count for this idx (once)
        old_count = 0
        # We can compute this from the global counter? No, we need specific idx.
        # Let's compute old_count by iterating pairs of others.
        # This is O(N^2).
        # We can do this once per iteration.
        
        # Actually, we can maintain a global counter and update it?
        # Updating global counter is O(N^2) per move.
        # If we accept a move, we update counter.
        # If we reject, we don't.
        # But we need to evaluate candidates first.
        
        # Let's just compute old_count now.
        others_list = others
        n_others = len(others_list)
        for i in range(n_others):
            x1, y1 = others_list[i]
            for j in range(i + 1, n_others):
                x2, y2 = others_list[j]
                # Area with old_pos
                # det(P1 - P_old, P2 - P_old)
                a = abs((x1 - old_x) * (y2 - old_y) - (y1 - old_y) * (x2 - old_x))
                if a == target:
                    old_count += 1
        
        # Evaluate candidates
        # To make this faster, we can limit the number of candidates evaluated to top 10?
        # Or just evaluate all.
        
        for cand in candidates:
            cx, cy = cand
            new_count = 0
            # Vectorized calculation for this candidate?
            # V = M - cand
            # Dets = V[:, 0][:, None] * V[:, 1][None, :] - V[:, 1][:, None] * V[:, 0][None, :]
            # This creates a (N-1) x (N-1) matrix.
            # Then count abs(Dets) == target.
            
            V = M - np.array([cx, cy])
            # Determinant matrix
            # det(V_i, V_j) = V_i[0]*V_j[1] - V_i[1]*V_j[0]
            # We can compute this as:
            # X = V[:, 0]
            # Y = V[:, 1]
            # Dets = X[:, None] * Y[None, :] - Y[:, None] * X[None, :]
            
            X = V[:, 0]
            Y = V[:, 1]
            Dets = X[:, None] * Y[None, :] - Y[:, None] * X[None, :]
            
            # Count pairs i < j where abs(Dets[i, j]) == target
            # Dets is symmetric with 0 diagonal.
            # We can just take upper triangle.
            # mask = np.triu(Dets, 1)
            # count = np.sum(np.abs(mask) == target)
            
            # This is fast in numpy.
            mask = np.triu(Dets, 1)
            new_count = np.sum(np.abs(mask) == target)
            
            delta = new_count - old_count
            if delta > best_delta:
                best_delta = delta
                best_candidate = cand
        
        if best_candidate is not None and best_delta > 0:
            # Accept move
            occupied.remove(old_pos)
            pts[idx] = best_candidate
            occupied.add(best_candidate)
            
            # Update global counter?
            # We need to update the global counter to keep track of other areas?
            # No, we only care about `target` for the local search objective.
            # But the final score is max over ALL areas.
            # So we must recompute the full counter at the end, or maintain it.
            # Maintaining it is O(N^2) per accepted move.
            # Let's just recompute the full counter every K accepted moves or at the end?
            # No, we need to know if we improved the `target` count.
            # We are tracking `current_count` which is the count of `target`.
            # We update `current_count` by `best_delta`.
            
            current_count += best_delta
            
            if current_count > best_count:
                best_count = current_count
                best_pts = pts[:]
                no_improve_iter = 0
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1
            
    return best_pts, best_count

def construct():
    t0 = time.time()
    deadline = t0 + 85.0 # 5s safety margin for final evaluation
    rng = random.Random(42)
    
    best_pts = None
    best_score = 0
    
    # Initial configurations
    configs = []
    
    # 1. Random
    random_pts = []
    used = set()
    while len(random_pts) < 30:
        p = (rng.randint(-30, 30), rng.randint(-30, 30))
        if p not in used:
            used.add(p)
            random_pts.append(p)
    configs.append(random_pts)
    
    # 2. Grid (6x5)
    grid_pts = []
    for x in range(-10, 11, 4):
        for y in range(-10, 11, 4):
            grid_pts.append((x, y))
    if len(grid_pts) < 30:
        used = set(grid_pts)
        while len(grid_pts) < 30:
            p = (rng.randint(-30, 30), rng.randint(-30, 30))
            if p not in used:
                used.add(p)
                grid_pts.append(p)
    configs.append(grid_pts[:30])
    
    # 3. Dense Cluster (small grid)
    cluster_pts = []
    for x in range(-5, 6, 2):
        for y in range(-5, 6, 2):
            cluster_pts.append((x, y))
    if len(cluster_pts) < 30:
        used = set(cluster_pts)
        while len(cluster_pts) < 30:
            p = (rng.randint(-30, 30), rng.randint(-30, 30))
            if p not in used:
                used.add(p)
                cluster_pts.append(p)
    configs.append(cluster_pts[:30])
    
    # Targets to try
    # Small areas are more likely to be frequent in dense configurations.
    targets_to_try = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24]
    
    # Time budget allocation
    # We have ~85s.
    # 3 configs * 10 targets = 30 runs.
    # ~2.8s per run.
    
    for config in configs:
        for target in targets_to_try:
            if time.time() > deadline - 5.0:
                break
            
            time_budget = min(2.5, deadline - 5.0 - time.time())
            if time_budget <= 0:
                break
            
            res_pts, res_score = _refined_local_search(config, target, time_budget, rng)
            
            # Evaluate true score
            true_score = _max_single_area(res_pts)
            
            if true_score > best_score:
                best_score = true_score
                best_pts = res_pts[:]
                
    # Final fallback
    if best_pts is None:
        best_pts = configs[0]
        
    return best_pts

if __name__ == "__main__":
    result = construct()
    assert len(result) == 30
    assert len(set(result)) == 30
    for x, y in result:
        assert -30 <= x <= 30 and -30 <= y <= 30
    score = _max_single_area(result)
    print(f"Score: {score}")
