import random
import time
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin

    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    # For each point, precompute all squared distances to other points
    # and map squared distance -> set of points at that distance from this point
    # Actually, let's think differently:
    # A set S is "no-isosceles" if for every three points p1, p2, p3 in S,
    # the three distances d(p1,p2), d(p1,p3), d(p2,p3) are all distinct.
    # Equivalently: no isosceles triangle means for every pair (p1, p2) in S,
    # and every other point p3 in S, d(p1,p3) != d(p2,p3) and d(p1,p3) != d(p1,p2) and d(p2,p3) != d(p1,p2).
    # Wait, that's not quite right. Let me reconsider.
    # Three points form an isosceles triangle if at least two of the three pairwise distances are equal.
    # So S is valid iff for every triple (a,b,c) in S, all three pairwise distances are distinct.
    
    # This is equivalent to: for every pair (a,b) in S, and every c in S\{a,b}, 
    #   d(a,c) != d(b,c) and d(a,c) != d(a,b) and d(b,c) != d(a,b)
    
    # Let's use a greedy approach with multiple random restarts and local search.
    # Represent the set of selected points as a bitmask or set.
    
    best_result = []
    best_size = 0
    
    # Precompute distance matrix (squared distances)
    n_pts = N * N
    dist_matrix = [[0] * n_pts for _ in range(n_pts)]
    point_to_idx = {}
    for i, (x, y) in enumerate(all_points):
        point_to_idx[(x, y)] = i
        for j in range(i + 1, n_pts):
            dx = x - all_points[j][0]
            dy = y - all_points[j][1]
            d2 = dx * dx + dy * dy
            dist_matrix[i][j] = d2
            dist_matrix[j][i] = d2
    
    # For each point i, and for each possible squared distance d2,
    # precompute the set of points j such that dist_matrix[i][j] == d2
    # This will help in checking validity quickly.
    # But this might be too large. Let's use a different approach.
    
    # Greedy with random ordering + local search
    def is_valid_set(selected):
        """Check if selected set has no isosceles triangle."""
        n = len(selected)
        if n < 3:
            return True
        # Check all triples
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    d_ij = dist_matrix[selected[i]][selected[j]]
                    d_ik = dist_matrix[selected[i]][selected[k]]
                    d_jk = dist_matrix[selected[j]][selected[k]]
                    if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                        return False
        return True
    
    def try_greedy(order, max_time):
        """Greedy: add points in the given order if valid."""
        selected = []
        # Use a set for O(1) lookup
        selected_set = set()
        # To check validity when adding point p, we need to check against all existing points
        # For each existing point q, and each other existing point r, we need to ensure no isosceles
        # This is O(k^2) where k is current size, so total is O(n^3) in worst case
        # Let's just do it naively for now
        
        for idx in order:
            if time.time() > max_time:
                break
            # Check if adding idx to selected is valid
            # For each pair (i, j) in selected, check if adding idx creates isosceles with i, j, idx
            # Also check for each i in selected, if d(i, idx) == d(i, j) for any j in selected\{i}
            # And d(j, idx) == d(i, j) for any j in selected\{i}
            valid = True
            for i in range(len(selected)):
                si = selected[i]
                d_sidx = dist_matrix[si][idx]
                for j in range(i + 1, len(selected)):
                    sj = selected[j]
                    d_sjsi = dist_matrix[si][sj]
                    d_sjidx = dist_matrix[sj][idx]
                    if d_sidx == d_sjsi or d_sidx == d_sjidx or d_sjsi == d_sjidx:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(idx)
        
        return selected
    
    # Try multiple random restarts
    random.seed(42)
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random ordering
        order = list(range(n_pts))
        random.shuffle(order)
        
        selected = try_greedy(order, deadline - 2)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = [all_points[i] for i in selected]
            
            # Try local search: try to add more points
            # Or try swapping
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try to add each unselected point
                selected_set = set(selected)
                for idx in range(n_pts):
                    if time.time() > deadline - 10:
                        break
                    if idx in selected_set:
                        continue
                    # Check if adding idx is valid
                    valid = True
                    for i in range(len(selected)):
                        si = selected[i]
                        d_sidx = dist_matrix[si][idx]
                        for j in range(i + 1, len(selected)):
                            sj = selected[j]
                            d_sjsi = dist_matrix[si][sj]
                            d_sjidx = dist_matrix[sj][idx]
                            if d_sidx == d_sjsi or d_sidx == d_sjidx or d_sjsi == d_sjidx:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        selected.append(idx)
                        selected_set.add(idx)
                        improved = True
        
        # Try a structured approach: points on a line with Sidon-like property
        # Actually, let's also try a simple structured pattern
        if restart % 5 == 0:
            # Try points on diagonal
            selected_diag = []
            for d in range(2 * N - 1):
                if time.time() > deadline - 5:
                    break
                # Points with x + y = d
                points_on_diag = []
                for x in range(N):
                    y = d - x
                    if 0 <= y < N:
                        points_on_diag.append((x, y))
                # Try to greedily add from this diagonal
                for p in points_on_diag:
                    idx = point_to_idx[p]
                    valid = True
                    for i in range(len(selected_diag)):
                        si = selected_diag[i]
                        d_sidx = dist_matrix[si[0] * N + si[1]][idx]
                        for j in range(i + 1, len(selected_diag)):
                            sj = selected_diag[j]
                            d_sjsi = dist_matrix[si[0] * N + si[1]][sj[0] * N + sj[1]]
                            d_sjidx = dist_matrix[sj[0] * N + sj[1]][idx]
                            if d_sidx == d_sjsi or d_sidx == d_sjidx or d_sjsi == d_sjidx:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        selected_diag.append(p)
            
            if len(selected_diag) > best_size:
                best_size = len(selected_diag)
                best_result = selected_diag
    
    # Convert to list of (x, y) tuples
    if not best_result:
        # Fallback: single point
        best_result = [(0, 0)]
    
    return best_result
