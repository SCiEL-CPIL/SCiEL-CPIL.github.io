import time
import random
import itertools

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    # 1. Generate initial candidate
    # Try a perturbed grid
    points = []
    # 5x4 grid
    for i in range(5):
        for j in range(4):
            # Add some noise to break symmetries and 4-collinearities
            # But keep it close to grid to maintain many triples
            # Actually, let's start with a random valid set first to be safe,
            # then try to improve.
            pass
            
    # Better: Random initialization with validity check
    def is_valid(pts):
        n = len(pts)
        # Check no 4 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        # Check collinearity of p1, p2, p3
                        # (x2-x1)(y3-y1) == (y2-y1)(x3-x1)
                        if (p2[0]-p1[0])*(p3[1]-p1[1]) != (p2[1]-p1[1])*(p3[0]-p1[0]):
                            continue
                        # p1, p2, p3 collinear. Check if p4 is on the line
                        if (p4[0]-p1[0])*(p3[1]-p1[1]) == (p4[1]-p1[1])*(p3[0]-p1[0]):
                            return False
        return True

    def random_valid_point_set():
        while True:
            pts = []
            seen = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    pts.append([x, y])
                    seen.add((x, y))
            if is_valid(pts):
                return pts

    # Heuristic: Try to start with a grid-like structure
    # Grid 5x4 has 4-collinearities. We will fix them.
    def generate_perturbed_grid():
        pts = []
        for i in range(5):
            for j in range(4):
                x = i
                y = j
                # Perturb slightly
                if random.random() < 0.5:
                    x += random.randint(-2, 2)
                if random.random() < 0.5:
                    y += random.randint(-2, 2)
                pts.append([x, y])
        # Remove duplicates
        unique_pts = []
        seen = set()
        for p in pts:
            if (p[0], p[1]) not in seen:
                unique_pts.append(p)
                seen.add((p[0], p[1]))
        # If we lost points, add random ones
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                unique_pts.append([x, y])
                seen.add((x, y))
        # Fix 4-collinearities
        # Simple local repair
        for _ in range(100):
            if is_valid(unique_pts):
                break
            # Find a 4-collinear set and move one point
            # This is slow. Just restart if not valid quickly?
            # Or use random_valid_point_set as fallback
            pass
        if not is_valid(unique_pts):
            return random_valid_point_set()
        return unique_pts

    best_points = generate_perturbed_grid()
    
    # Fallback if grid generation fails
    if not is_valid(best_points):
        best_points = random_valid_point_set()
        
    # Score Calculation
    def calc_score(pts):
        n = 20
        # Precompute collinear triples
        # triple_mask[i] is a bitmask of points that form a triple with i? No.
        # Let's store triples as a list of (i, j, k)
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0]):
                        triples.append((i, j, k))
        
        # Find Max Independent Set
        # State: current_mask, available_mask
        # We want to maximize popcount(current_mask)
        
        best_size = 0
        
        # Precompute for each point i, which points j cannot be with i in the set
        # if there is a k such that i, j, k is a triple?
        # No, the constraint is: if i, j are in set, then no k can be in set if (i,j,k) is a triple.
        # So if we add i, we must remove from available any j such that there exists k in current such that (i,j,k) is a triple?
        # And also any k such that there exists j in current such that (i,j,k) is a triple?
        # Actually, if we add i, we check against all pairs in current.
        # If a pair (j,k) in current forms a triple with i, we cannot add i.
        # If we add i, we must remove from available any point m such that there exists a pair (j,k) in current? No.
        # If we add i, and later add m, we need to ensure no triple (i, m, x) for any x in current.
        # So if (i, m, x) is a triple and x is in current, m cannot be added.
        # So when we add i, we should remove from available any m such that there exists x in current with (i, m, x) being a triple.
        
        # Let's build a conflict graph for the B&B
        # conflict[i][j] is True if i and j cannot be in the same set? No, they can, unless there is a k.
        # The constraint is 3-uniform.
        
        def solve_bnb(current_mask, available_mask, current_size):
            nonlocal best_size
            if current_size + bin(available_mask).count('1') <= best_size:
                return
            
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick a point from available
            # Heuristic: pick the one with most conflicts?
            # Or just the first one.
            i = (available_mask & -available_mask).bit_length() - 1
            # Wait, bit_length gives position + 1?
            # If mask is 1 (0b1), bit_length is 1. Index 0.
            # If mask is 2 (0b10), bit_length is 2. Index 1.
            # So i = bit_length - 1.
            
            # Option 1: Exclude i
            solve_bnb(current_mask, available_mask ^ (1 << i), current_size)
            
            # Option 2: Include i
            # Check if i can be included
            valid = True
            new_available = available_mask ^ (1 << i)
            
            # Check conflicts with current set
            # For every pair (j, k) in current_mask, if (i, j, k) is a triple, invalid.
            # This is expensive if done naively.
            # Precompute: for each i, list of pairs (j, k) that form a triple with i.
            # Then check if current_mask contains any such pair.
            
            # Also, update new_available:
            # For every j in current_mask, if (i, j, m) is a triple, remove m from new_available.
            
            # Let's precompute these structures outside
            pass

        # Precompute for B&B
        # triples is a list of (i, j, k)
        # pairs_with_third[i] = set of (j, k) such that (i, j, k) is a triple
        # Actually, we need:
        # 1. For a given i and current_mask, can we add i?
        #    Check if any (j, k) in current_mask is in pairs_with_third[i].
        # 2. If we add i, which points m become unavailable?
        #    For any j in current_mask, if (i, j, m) is a triple, remove m.
        #    So we need: for given i and j, what is m?
        #    third_point[i][j] = m or None.
        
        third_point = [[None]*20 for _ in range(20)]
        for i, j, k in triples:
            third_point[i][j] = k
            third_point[i][k] = j
            third_point[j][i] = k
            third_point[j][k] = i
            third_point[k][i] = j
            third_point[k][j] = i
            
        # pairs_with_third[i] = set of masks? No, set of pairs.
        # To check if current_mask contains a pair (j, k) from pairs_with_third[i]:
        # We can iterate through pairs in pairs_with_third[i] and check if (current_mask & (1<<j)) and (current_mask & (1<<k)).
        # Or precompute a bitmask for each pair?
        # There are at most 1140 triples. So at most 1140 pairs per i? No, 3 pairs per triple.
        
        # Let's just implement the check inline for simplicity, N=20 is small.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def solve_bnb_impl(current_mask, available_mask, current_size):
            nonlocal best_size
            if current_size + bin(available_mask).count('1') <= best_size:
                return
            
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick lowest set bit
            i = (available_mask & -available_mask).bit_length() - 1
            bit_i = 1 << i
            
            # Exclude i
            solve_bnb_impl(current_mask, available_mask ^ bit_i, current_size)
            
            # Include i
            # Check validity
            valid = True
            # Check against current set
            # Iterate through bits in current_mask?
            # We need to check if any pair in current_mask forms a triple with i.
            # Let's iterate through all j < k in current_mask?
            # Or iterate through all j in current_mask, and check if third_point[i][j] is in current_mask.
            for j in range(20):
                if current_mask & (1 << j):
                    k = third_point[i][j]
                    if k is not None and (current_mask & (1 << k)):
                        valid = False
                        break
            
            if valid:
                # Update available
                new_available = available_mask ^ bit_i
                # For each j in current_mask, if (i, j, m) is a triple, remove m.
                for j in range(20):
                    if current_mask & (1 << j):
                        m = third_point[i][j]
                        if m is not None:
                            new_available &= ~(1 << m)
                
                solve_bnb_impl(current_mask | bit_i, new_available, current_size + 1)
                
        solve_bnb_impl(0, (1 << 20) - 1, 0)
        return best_size

    best_score = calc_score(best_points)
    current_points = best_points
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.99
    
    while time.time() - start_time < timeout:
        # Propose move
        new_points = current_points.copy()
        idx = random.randint(0, 19)
        old_x, old_y = new_points[idx]
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        new_points[idx] = [new_x, new_y]
        
        # Check validity
        if not is_valid(new_points):
            # Reject
            continue
            
        new_score = calc_score(new_points)
        
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = new_points
                T = 10.0 # Reset temperature? Or keep cooling
        else:
            # Accept with probability
            delta = new_score - current_score
            if random.random() < pow(2.71828, -delta / T):
                current_points = new_points
                current_score = new_score
        
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best_points

# Test locally? No, just output.
