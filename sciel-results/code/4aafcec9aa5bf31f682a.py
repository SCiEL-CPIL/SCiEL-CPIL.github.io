import time
import random
import sys

N = 16
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 270  # Safety margin before 300s limit
    
    # Indexing: point (x, y) -> x * N + y
    points = [(x, y) for x in range(N) for y in range(N)]
    idx_to_pt = {i: p for i, p in enumerate(points)}
    pt_to_idx = {p: i for i, p in enumerate(points)}
    
    # Precompute squared distances between all pairs
    # d2[i][j] = squared distance between point i and point j
    d2 = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        ax, ay = idx_to_pt[i]
        for j in range(i + 1, TOTAL_POINTS):
            bx, by = idx_to_pt[j]
            dist = (ax - bx)**2 + (ay - by)**2
            d2[i][j] = dist
            d2[j][i] = dist
    
    # Precompute for each pair (i, j), which points k form an isosceles triangle with i, j
    # iso_conflict[i][j] = set of k such that {i, j, k} is isosceles
    # For efficiency, store as a list of lists
    iso_conflict = [[set() for _ in range(TOTAL_POINTS)] for _ in range(TOTAL_POINTS)]
    
    for i in range(TOTAL_POINTS):
        ax, ay = idx_to_pt[i]
        for j in range(i + 1, TOTAL_POINTS):
            bx, by = idx_to_pt[j]
            d_ij = d2[i][j]
            
            # Find all k where d(i,k) == d(i,j) or d(j,k) == d(i,j) or d(i,k) == d(j,k)
            for k in range(TOTAL_POINTS):
                if k == i or k == j:
                    continue
                cx, cy = idx_to_pt[k]
                d_ik = d2[i][k]
                d_jk = d2[j][k]
                
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    iso_conflict[i][j].add(k)
                    iso_conflict[j][i].add(k)
    
    def is_valid_no_iso(chosen_set):
        """Check if chosen_set has no isosceles triangles"""
        pts = list(chosen_set)
        n = len(pts)
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    i, j, k = pts[a], pts[b], pts[c]
                    # Check if any k in iso_conflict[i][j] equals the third point
                    if k in iso_conflict[i][j]:
                        return False
        return True
    
    def is_maximal(chosen_set, all_points_set):
        """Check if no point can be added to chosen_set without creating isosceles"""
        for p in all_points_set:
            if p in chosen_set:
                continue
            # Check if adding p creates an isosceles triangle
            for q in chosen_set:
                if p in iso_conflict[q][p] or p in iso_conflict[p][q]:
                    # Adding p would create isosceles with q and some other point
                    # Actually, we need to check: does {p, q, r} form isosceles for some r in chosen_set?
                    # Simpler: check if p conflicts with any pair in chosen_set
                    pass
            # Better: check if p can be added without creating isosceles
            # p can be added if for all pairs (q, r) in chosen_set, p is not in iso_conflict[q][r]
            # AND for all q in chosen_set, p is not in iso_conflict[p][q] (which is always true by definition)
            # Actually, the condition is: no three points in (chosen_set ∪ {p}) form isosceles
            # Since chosen_set is already valid, we only need to check triples involving p
            # i.e., for all pairs (q, r) in chosen_set, p not in iso_conflict[q][r]
            # AND for all q in chosen_set, the pair (p, q) doesn't create issue with existing points
            # Wait, that's not right. The condition is just: no isosceles triangle in the new set.
            # Since old set is valid, new isosceles must involve p.
            # So: for all pairs (q, r) in chosen_set, p not in iso_conflict[q][r]
            # AND for all q in chosen_set, no r in chosen_set such that q in iso_conflict[p][r]
            # But iso_conflict[p][r] contains points that form isosceles with p and r.
            # If q is in iso_conflict[p][r], then {p, r, q} is isosceles.
            # So we need: for all q, r in chosen_set with q != r, q not in iso_conflict[p][r]
            # Which is equivalent to: for all r in chosen_set, iso_conflict[p][r] ∩ chosen_set = empty
            can_add = True
            for r in chosen_set:
                if iso_conflict[p][r] & chosen_set:
                    can_add = False
                    break
            if can_add:
                return False  # Can add p, so not maximal
        return True  # Cannot add any point, so maximal
    
    def greedy_construct(order):
        """Build a maximal no-isosceles set using given order"""
        chosen = set()
        for p in order:
            # Check if p can be added
            can_add = True
            for q in chosen:
                if iso_conflict[p][q] & chosen:
                    can_add = False
                    break
            if can_add:
                chosen.add(p)
        return chosen
    
    def try_remove(chosen_set):
        """Try to remove points from chosen_set while maintaining maximality"""
        chosen = set(chosen_set)
        improved = True
        while improved:
            improved = False
            for p in list(chosen):
                # Try removing p
                new_chosen = chosen - {p}
                # Check if new_chosen is still no-isosceles (it should be, since it's a subset)
                # Check if new_chosen is still maximal
                if is_maximal(new_chosen, set(range(TOTAL_POINTS))):
                    chosen = new_chosen
                    improved = True
                    break
        return chosen
    
    # Generate different orderings
    def get_orderings():
        orderings = []
        
        # Row-major
        orderings.append(list(range(TOTAL_POINTS)))
        
        # Column-major
        col_order = [x + y * N for y in range(N) for x in range(N)]
        orderings.append(col_order)
        
        # Diagonal order
        diag_order = []
        for d in range(2 * N - 1):
            for x in range(N):
                y = d - x
                if 0 <= y < N:
                    diag_order.append(x + y * N)
        orderings.append(diag_order)
        
        # Checkerboard (even then odd parity)
        even_pts = [i for i in range(TOTAL_POINTS) if (i % N + i // N) % 2 == 0]
        odd_pts = [i for i in range(TOTAL_POINTS) if (i % N + i // N) % 2 == 1]
        orderings.append(even_pts + odd_pts)
        orderings.append(odd_pts + even_pts)
        
        # Random orders
        for _ in range(20):
            random_order = list(range(TOTAL_POINTS))
            random.shuffle(random_order)
            orderings.append(random_order)
        
        return orderings
    
    best_set = None
    best_size = TOTAL_POINTS + 1
    
    orderings = get_orderings()
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        # Greedy construct
        chosen = greedy_construct(order)
        
        # Local search: try to remove points
        chosen = try_remove(chosen)
        
        size = len(chosen)
        if size < best_size:
            best_size = size
            best_set = set(chosen)
    
    # Additional local search: try swapping points
    if best_set is not None:
        current_best = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in list(current_best):
                # Try replacing p with some other point q
                for q in range(TOTAL_POINTS):
                    if q in current_best:
                        continue
                    new_set = current_best - {p} | {q}
                    # Check if new_set is valid no-isosceles
                    if is_valid_no_iso(new_set):
                        # Check if maximal
                        if is_maximal(new_set, set(range(TOTAL_POINTS))):
                            if len(new_set) < len(current_best):
                                current_best = new_set
                                best_set = new_set
                                best_size = len(new_set)
                                improved = True
                                break
                if improved:
                    break
    
    # Convert to list of tuples
    result = [idx_to_pt[i] for i in best_set]
    return result
