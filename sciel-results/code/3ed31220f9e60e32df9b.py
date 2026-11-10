import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    num_points = N * N
    coords = [(i // N, i % N) for i in range(num_points)]
    
    # Precompute distances
    # To save memory and time, we can compute on the fly or store in a compact way.
    # 4096 * 4096 * 8 bytes = 134 MB. Might be too much?
    # Let's compute on the fly or store as a list of arrays if memory allows.
    # 4096*4096 = 16M entries. Integers are small. 
    # Let's try to avoid storing the full matrix if possible, or use numpy.
    # But standard library is safer for compatibility unless numpy is explicitly imported in the prompt? 
    # "Python standard library, NumPy and SciPy are available."
    # Let's use NumPy for speed if available, else fallback.
    
    try:
        import numpy as np
        use_numpy = True
        xs = np.array([c[0] for c in coords], dtype=np.int32)
        ys = np.array([c[1] for c in coords], dtype=np.int32)
        # dist2[i, j]
        # This creates a 4096x4096 array. 
        # 16M * 4 bytes = 64 MB. Acceptable.
        dist2 = (xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2
    except ImportError:
        use_numpy = False
        # Fallback to list of lists if numpy fails (unlikely but safe)
        dist2 = [[0]*num_points for _ in range(num_points)]
        for i in range(num_points):
            xi, yi = coords[i]
            for j in range(num_points):
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist2[i][j] = dx*dx + dy*dy

    def get_dist(i, j):
        if use_numpy:
            return dist2[i, j]
        else:
            return dist2[i][j]

    # State:
    # selected: set of indices
    # dist_counts: list of dicts. dist_counts[i][d] = count of points j in selected such that dist(i,j)==d
    # Note: dist(i,i) = 0. We ignore self.
    
    def init_state():
        selected = set()
        dist_counts = [{} for _ in range(num_points)]
        return selected, dist_counts

    def calc_violations(selected, dist_counts):
        """
        Returns total number of isosceles triangles (violations).
        A violation exists if for any point i, there is a distance d with count > 1.
        Specifically, if count is k, it contributes (k choose 2) to the number of isosceles triangles with apex i?
        Actually, the problem says "no three points forming an isosceles triangle".
        If d(i, j1) = d(i, j2), then (i, j1, j2) is an isosceles triangle.
        So yes, each pair of points at the same distance from i forms a violation.
        Total violations = sum over i of sum over d of C(count[i][d], 2).
        """
        total = 0
        for i in selected:
            counts = dist_counts[i]
            for d, c in counts.items():
                if c > 1:
                    total += c * (c - 1) // 2
        return total

    def try_add(selected, dist_counts, p):
        """
        Attempt to add point p. Returns (success, delta_violations).
        If success, updates structures.
        """
        if p in selected:
            return False, 0
        
        # Calculate new violations introduced by p
        # 1. Violations at p: distances from p to existing points.
        # 2. Violations at existing points q: distance d(p,q) might match an existing distance at q.
        
        delta_v = 0
        new_counts_for_p = {}
        
        for q in selected:
            d = get_dist(p, q)
            # Check if d already exists in dist_counts[p] (which is empty initially for p)
            # Since p is new, dist_counts[p] is empty. So no conflict at p yet?
            # Wait, dist_counts[p] is updated as we go? 
            # No, we check against current state.
            # For p, all distances are new. So no conflict at p from "previous" points at p.
            # BUT, if we add p, we must check if d(p, q1) == d(p, q2).
            # Since we are adding one point at a time, and p is not in selected, 
            # the distances from p to selected points are all distinct IF no two selected points are at same dist from p.
            # So we need to check if d is already in new_counts_for_p.
            
            if d in new_counts_for_p:
                # This means there is another q' such that d(p, q') == d(p, q).
                # This creates an isosceles triangle with apex p.
                # How many? 
                # If we add p, and there are k points at distance d from p, 
                # the number of pairs is C(k, 2).
                # Before adding p, these points didn't form a triangle with p.
                # So we add C(k, 2) violations?
                # Actually, we just need the change in cost.
                # Let's just compute the cost change properly.
                pass
            
            # Check conflict at q
            # dist_counts[q] has counts of distances from q to other selected points.
            if d in dist_counts[q]:
                # Adding p increases count of d at q by 1.
                # Old count: c. New count: c+1.
                # Old contribution: C(c, 2). New contribution: C(c+1, 2).
                # Delta = C(c+1, 2) - C(c, 2) = c.
                delta_v += dist_counts[q][d]
            
            # Record distance for p
            if d in new_counts_for_p:
                new_counts_for_p[d] += 1
            else:
                new_counts_for_p[d] = 1
        
        # Now handle violations at p itself.
        # For each distance d with count k in new_counts_for_p,
        # the number of isosceles triangles with apex p is C(k, 2).
        for d, k in new_counts_for_p.items():
            if k > 1:
                delta_v += k * (k - 1) // 2
        
        # If we decide to accept, we update.
        # For now, just return delta.
        return True, delta_v, new_counts_for_p

    # We need a way to accept/reject and update.
    # Let's wrap the state in a class or use closures carefully.
    
    def add_point(selected, dist_counts, p):
        if p in selected:
            return
        selected.add(p)
        new_counts_for_p = {}
        for q in selected:
            if q == p: continue
            d = get_dist(p, q)
            # Update q's counts
            if d in dist_counts[q]:
                dist_counts[q][d] += 1
            else:
                dist_counts[q][d] = 1
            
            # Update p's counts
            if d in new_counts_for_p:
                new_counts_for_p[d] += 1
            else:
                new_counts_for_p[d] = 1
        
        dist_counts[p] = new_counts_for_p

    def remove_point(selected, dist_counts, p):
        if p not in selected:
            return
        selected.remove(p)
        for q in selected:
            d = get_dist(p, q)
            # Update q's counts
            dist_counts[q][d] -= 1
            if dist_counts[q][d] == 0:
                del dist_counts[q][d]
        
        # Clear p's counts
        dist_counts[p] = {}

    def get_violations(selected, dist_counts):
        total = 0
        for i in selected:
            counts = dist_counts[i]
            for d, c in counts.items():
                if c > 1:
                    total += c * (c - 1) // 2
        return total

    # Simulated Annealing
    rng = random.Random(42)
    
    # Initialize with a small random set
    initial_size = 10
    selected = set(rng.sample(range(num_points), initial_size))
    dist_counts = [{} for _ in range(num_points)]
    for p in selected:
        add_point(selected, dist_counts, p)
        
    # Wait, add_point modifies selected. I should add to a temp set then copy?
    # My add_point above adds to the set passed in.
    # Let's re-init properly.
    selected = set()
    dist_counts = [{} for _ in range(num_points)]
    init_candidates = rng.sample(range(num_points), initial_size)
    for p in init_candidates:
        add_point(selected, dist_counts, p)
        
    current_violations = get_violations(selected, dist_counts)
    best_selected = set(selected)
    best_violations = current_violations
    
    T = 1000.0
    cooling = 0.995
    min_T = 0.1
    
    time_limit = deadline - 5.0
    
    while time.time() < time_limit and T > min_T:
        # Propose move
        move_type = rng.random()
        if move_type < 0.3:
            # Add
            p = rng.randint(0, num_points - 1)
            if p not in selected:
                # Calculate delta
                # We need to calculate delta_violations efficiently.
                # Re-use logic from try_add
                delta_v = 0
                new_counts_for_p = {}
                for q in selected:
                    d = get_dist(p, q)
                    if d in dist_counts[q]:
                        delta_v += dist_counts[q][d]
                    if d in new_counts_for_p:
                        new_counts_for_p[d] += 1
                    else:
                        new_counts_for_p[d] = 1
                for d, k in new_counts_for_p.items():
                    if k > 1:
                        delta_v += k * (k - 1) // 2
                
                new_violations = current_violations + delta_v
                if new_violations < current_violations or rng.random() < math.exp(-delta_v / T):
                    # Accept
                    add_point(selected, dist_counts, p)
                    current_violations = new_violations
                    if current_violations == 0:
                        if len(selected) > len(best_selected):
                            best_selected = set(selected)
                            best_violations = 0
                            
        elif move_type < 0.6:
            # Remove
            if selected:
                p = rng.choice(list(selected))
                # Calculate delta (negative)
                delta_v = 0
                # Removing p reduces violations.
                # 1. Violations at p disappear.
                for d, k in dist_counts[p].items():
                    if k > 1:
                        delta_v -= k * (k - 1) // 2
                # 2. Violations at other points q reduce.
                for q in selected:
                    if q == p: continue
                    d = get_dist(p, q)
                    # Current count at q is c. New count is c-1.
                    # Old contrib: C(c,2). New: C(c-1,2).
                    # Delta: C(c-1,2) - C(c,2) = - (c-1).
                    delta_v -= (dist_counts[q][d] - 1)
                
                new_violations = current_violations + delta_v
                if new_violations < current_violations or rng.random() < math.exp(-delta_v / T):
                    # Accept
                    remove_point(selected, dist_counts, p)
                    current_violations = new_violations
                    if current_violations == 0:
                        if len(selected) > len(best_selected):
                            best_selected = set(selected)
                            best_violations = 0
                            
        else:
            # Swap
            if selected:
                p_old = rng.choice(list(selected))
                p_new = rng.randint(0, num_points - 1)
                if p_new not in selected:
                    # Remove p_old, Add p_new
                    # Calculate delta for removal
                    delta_rem = 0
                    for d, k in dist_counts[p_old].items():
                        if k > 1:
                            delta_rem -= k * (k - 1) // 2
                    for q in selected:
                        if q == p_old: continue
                        d = get_dist(p_old, q)
                        delta_rem -= (dist_counts[q][d] - 1)
                    
                    # Calculate delta for addition (p_new to set without p_old)
                    # This is tricky because the set changes.
                    # Easier: Just execute, check cost, revert if bad?
                    # Or calculate carefully.
                    # Let's just do a "trial" add/remove and check.
                    # For simplicity in SA, let's just do Add and Remove separately in the loop.
                    # Swap is complex to calculate delta for.
                    pass
                    
        T *= cooling
        
    # Final Greedy Polish
    # Try to add any point that doesn't increase violations
    improved = True
    while improved:
        improved = False
        for p in range(num_points):
            if p not in selected:
                # Check if adding p keeps violations at 0 (or reduces them)
                # If current_violations > 0, we want to reduce.
                # If current_violations == 0, we want to keep it 0.
                
                # Quick check: if current_violations == 0, only add if delta_v == 0
                # If current_violations > 0, add if delta_v < 0
                
                # Calculate delta_v
                delta_v = 0
                new_counts_for_p = {}
                for q in selected:
                    d = get_dist(p, q)
                    if d in dist_counts[q]:
                        delta_v += dist_counts[q][d]
                    if d in new_counts_for_p:
                        new_counts_for_p[d] += 1
                    else:
                        new_counts_for_p[d] = 1
                for d, k in new_counts_for_p.items():
                    if k > 1:
                        delta_v += k * (k - 1) // 2
                
                if delta_v < 0 or (current_violations == 0 and delta_v == 0):
                    add_point(selected, dist_counts, p)
                    current_violations += delta_v
                    improved = True
                    if current_violations == 0:
                        if len(selected) > len(best_selected):
                            best_selected = set(selected)
                            
    # Ensure best_selected is valid
    # Re-validate best_selected to be absolutely sure
    final_set = set(best_selected)
    # Quick validation of final_set
    valid = True
    if len(final_set) > 2:
        # Re-compute dist counts for final_set to check
        temp_counts = [{} for _ in range(num_points)]
        for p in final_set:
            for q in final_set:
                if p != q:
                    d = get_dist(p, q)
                    if d in temp_counts[p]:
                        temp_counts[p][d] += 1
                    else:
                        temp_counts[p][d] = 1
        for p in final_set:
            for d, c in temp_counts[p].items():
                if c > 1:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to a small valid set
        # (0,0), (1,2), (4,1) is valid?
        # d(0,0->1,2)=5, d(0,0->4,1)=17, d(1,2->4,1)=10.
        # Dist from 0,0: 5, 17 (distinct)
        # Dist from 1,2: 5, 10 (distinct)
        # Dist from 4,1: 17, 10 (distinct)
        # Yes.
        final_set = {0, N+2, 4*N+1} # (0,0), (1,2), (4,1)
        
    return [(i // N, i % N) for i in final_set]

# EVOLVE-BLOCK-END
