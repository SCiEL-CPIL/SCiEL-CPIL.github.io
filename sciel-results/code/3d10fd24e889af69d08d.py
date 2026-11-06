import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                            if count >= 4:
                                return True
        return False

    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinearity masks
        # coll_mask[i] is a bitmask of points j such that i and j are collinear with some k?
        # No, for the branch and bound, we need to know: if we pick point i, which points j are "forbidden"
        # because there exists a k already picked such that i, j, k are collinear?
        
        # Actually, the standard approach is:
        # State: A (selected), C (candidates)
        # When adding p to A, we must remove from C any q such that there exists r in A where p, q, r are collinear.
        
        # Precompute for each pair (i, j) the mask of k such that i, j, k are collinear.
        coll_pair_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll_pair_mask[i][j] = mask
                coll_pair_mask[j][i] = mask
        
        best = [0]
        
        # Heuristic: order points by "collinearity degree" (number of other points collinear with them in any triple?)
        # Actually, just pick the first available point in a sorted order.
        # Let's sort indices by the number of collinear partners.
        # deg[i] = number of j such that there exists k with i,j,k collinear.
        deg = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if coll_pair_mask[i][j]:
                    deg[i] += 1
                    deg[j] += 1
        
        order = sorted(range(n), key=lambda x: -deg[x])
        # Map original index to position in order? Or just iterate in this order?
        # The bitmask approach usually iterates bits. Let's stick to bitmask but maybe sort the bits?
        # It's easier to just use the bitmask and pick the lowest set bit, but we can pre-sort the points.
        
        # Let's reorder the points themselves to put high-degree points first.
        # This complicates the bitmask indices. Let's just use the bitmask and a simple heuristic:
        # Pick the point in C with the highest degree.
        
        def backtrack(A, C, depth):
            if depth + bin(C).count('1') <= best[0]:
                return
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Pick a point p from C
            # Heuristic: pick p with max degree in the remaining graph?
            # For speed, let's just pick the lowest set bit, but maybe we can do better.
            # Let's try picking the point that conflicts with the most points in C.
            # Conflict of p with q is if there exists r in A such that p, q, r collinear?
            # No, when we add p, we remove q from C if p, q, r collinear for some r in A.
            # This depends on A.
            
            # Simple heuristic: pick the first bit.
            p_bit = C & -C
            p = p_bit.bit_length() - 1
            
            # Option 1: Exclude p
            backtrack(A, C ^ p_bit, depth)
            
            # Option 2: Include p
            # Check compatibility: is there any pair (a1, a2) in A such that p, a1, a2 are collinear?
            # This is equivalent to: for any a in A, is there another a' in A such that a' is in coll_pair_mask[p][a]?
            # coll_pair_mask[p][a] contains k such that p, a, k collinear.
            # So we need (coll_pair_mask[p][a] & A) to be 0 for all a in A?
            # Wait, if a is in A, and k is in A, and p, a, k collinear, then adding p creates a triple.
            # So we need: for all a in A, (coll_pair_mask[p][a] & A) == 0.
            
            compatible = True
            temp_A = A
            while temp_A:
                a_bit = temp_A & -temp_A
                a = a_bit.bit_length() - 1
                if coll_pair_mask[p][a] & A:
                    compatible = False
                    break
                temp_A ^= a_bit
            
            if compatible:
                new_C = C ^ p_bit
                # Remove from C any q such that p, q, a are collinear for some a in A.
                # i.e., q is in coll_pair_mask[p][a] for some a in A.
                temp_A = A
                while temp_A:
                    a_bit = temp_A & -temp_A
                    a = a_bit.bit_length() - 1
                    new_C &= ~coll_pair_mask[p][a]
                    temp_A ^= a_bit
                
                backtrack(A | p_bit, new_C, depth + 1)
        
        backtrack(0, (1 << n) - 1, 0)
        return best[0]

    best_score = 20
    best_config = None
    
    # Generate candidates
    # Strategy 1: Perturbed Grid
    # Strategy 2: Random in small box
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        if attempts % 1000 == 0:
            # Periodic check
            pass
        
        if attempts <= 2000:
            # Perturbed 5x4 grid
            pts = []
            for i in range(5):
                for j in range(4):
                    # Small random shift
                    di = random.choice([-1, 0, 0, 1])
                    dj = random.choice([-1, 0, 0, 1])
                    pts.append([i + di, j + dj])
        else:
            # Random in [-10, 10]
            pts = []
            for _ in range(20):
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                pts.append([x, y])
        
        # Check distinct
        if len(set(map(tuple, pts))) < 20:
            continue
        
        # Check no 4 collinear
        if has_4_collinear(pts):
            continue
        
        # Compute score
        score = max_no_3_collinear(pts)
        if score < best_score:
            best_score = score
            best_config = pts
            if best_score <= 9:
                # Likely optimal or near optimal for small n, but 20 is larger.
                # Known values: n=19 max no-3 is 10? 
                # If we get 10, that's good.
                break
    
    if best_config is None:
        # Fallback: Parabola (score 20)
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_config
