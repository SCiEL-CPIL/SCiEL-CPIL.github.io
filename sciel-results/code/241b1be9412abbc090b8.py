import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_points = None
    best_score = 20 # Worst case
    
    # Helper: Check no 4 collinear
    def is_valid(points):
        # Check all 4-subsets? 20C4 = 4845. Fast.
        for i in range(20):
            for j in range(i+1, 20):
                for k in range(j+1, 20):
                    for l in range(k+1, 20):
                        if are_collinear(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def are_collinear(p1, p2, p3, p4):
        # Check if any 3 are collinear? No, if 4 are collinear, any 3 are.
        # But we just need to check if all 4 are on the same line.
        # Cross product check.
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Check if p1,p2,p3 are collinear AND p1,p2,p4 are collinear
        # Or simpler: check if the area of triangle p1p2p3 is 0 and p1p2p4 is 0?
        # If p1,p2,p3 collinear, they define a line. Check if p4 is on it.
        # If p1,p2,p3 not collinear, they can't all be collinear with p4 unless p1,p2,p4 are?
        # Actually, if 4 points are collinear, then any subset of 3 is collinear.
        # So we just need to check if p1,p2,p3 are collinear AND p1,p2,p4 are collinear?
        # No, if p1,p2,p3 are collinear, and p4 is not, then not 4 collinear.
        # If p1,p2,p3 are NOT collinear, then 4 cannot be collinear.
        # So: return is_collinear(p1,p2,p3) and is_collinear(p1,p2,p4)
        
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            
        return cross(p1, p2, p3) == 0 and cross(p1, p2, p4) == 0

    def get_triples(points):
        triples = []
        for i in range(20):
            for j in range(i+1, 20):
                for k in range(j+1, 20):
                    if are_collinear(points[i], points[j], points[k], None): # Need 3-point check
                        triples.append((i, j, k))
        return triples

    def are_collinear_3(p1, p2, p3):
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return cross(p1, p2, p3) == 0

    # Exact Max No-3-Collinear Solver
    def max_no_three_collinear(points):
        n = 20
        # Build list of triples (masks)
        triple_masks = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear_3(points[i], points[j], points[k]):
                        mask = (1 << i) | (1 << j) | (1 << k)
                        triple_masks.append(mask)
        
        if not triple_masks:
            return n
            
        # Branch and Bound
        best = 0
        # Sort triples by mask value? Not strictly necessary.
        
        def dfs(mask, count):
            nonlocal best
            if count + bin(mask).count('1') <= best:
                return
            if mask == 0:
                if count > best:
                    best = count
                return
            
            # Pick a vertex
            # Heuristic: Pick the vertex involved in the most remaining triples?
            # For speed, just pick the lowest set bit.
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            
            # Option 1: Include idx
            # If we include idx, we must remove all triples containing idx?
            # No, we are finding a set of vertices with NO triple.
            # So if we include idx, we must ensure no triple containing idx is fully selected.
            # This is equivalent to: We select a set S. S is valid if no triple is a subset of S.
            # This is the Independent Set problem in a 3-uniform hypergraph.
            
            # Standard MIS B&B:
            # 1. Pick a vertex v.
            # 2. Branch: Include v.
            #    If we include v, we cannot include any other vertex that forms a triple with v AND another selected vertex?
            #    Actually, if we include v, we must remove from the candidate set any vertex u such that {v, u, w} is a triple for some w?
            #    No, that's for graphs. For hypergraphs, it's more complex.
            #    
            #    Let's use a simpler bound.
            #    If we include v, we must check all triples containing v.
            #    This path is expensive.
            
            # Alternative: Use the fact that N=20 is small.
            # We can iterate over all subsets? 2^20 = 1M.
            # For each subset, check if it contains a triple.
            # 1M * (check cost). Check cost: iterate triples (max 1140) and see if mask & triple_mask == triple_mask.
            # 1M * 1140 ops = 1.14B ops. Too slow in Python.
            
            # Back to B&B.
            # Pick a triple T = {a, b, c}.
            # At least one of a, b, c must NOT be in the independent set.
            # Branch on which one is excluded?
            # Or: Branch on including one of them?
            
            # Let's try:
            # Pick the first triple T.
            # We must exclude at least one vertex from T.
            # Branch 1: Exclude a. Recurse on mask without a.
            # Branch 2: Exclude b. Recurse on mask without b.
            # Branch 3: Exclude c. Recurse on mask without c.
            # This guarantees we don't pick the full triple.
            
            if not triple_masks:
                 best = max(best, count + bin(mask).count('1'))
                 return
            
            # Pick a triple that intersects mask
            # To make it fast, precompute for each vertex the list of triples it's in?
            # Or just pick the first triple in the list that is a subset of mask?
            # If no triple is a subset of mask, then mask is a valid independent set!
            valid = True
            t_mask = 0
            for t in triple_masks:
                if (t & mask) == t:
                    valid = False
                    t_mask = t
                    break
                    
            if valid:
                best = max(best, count + bin(mask).count('1'))
                return
            
            # Exclude one vertex from t_mask
            # t_mask has 3 bits set.
            a_bit = t_mask & -t_mask
            b_bit = (t_mask >> a_bit.bit_length()) & 1
            # Get the 3 bits
            bits = []
            tmp = t_mask
            while tmp:
                lsb = tmp & -tmp
                bits.append(lsb)
                tmp ^= lsb
                
            for bit in bits:
                new_mask = mask & ~bit
                dfs(new_mask, count)
                
        dfs((1 << n) - 1, 0)
        return best

    # Search
    # Random initialization
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Generate random points
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append((x, y))
        
        # Check validity
        if not is_valid(points):
            continue
            
        score = max_no_three_collinear(points)
        if score < best_score:
            best_score = score
            best_points = points[:]
            
        # Local Search
        temp = 10.0
        current_points = points[:]
        current_score = score
        
        for step in range(5000):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 19)
            old_x, old_y = current_points[i]
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            if (new_x, new_y) == (old_x, old_y):
                continue
                
            # Check if new point collides
            if (new_x, new_y) in [p for j, p in enumerate(current_points) if j != i]:
                continue
                
            candidate = current_points[:]
            candidate[i] = (new_x, new_y)
            
            if not is_valid(candidate):
                continue
                
            new_score = max_no_three_collinear(candidate)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / temp):
                current_points = candidate
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
                    
            temp *= 0.995
            
    # Convert to list of lists
    if best_points:
        return [list(p) for p in best_points]
    else:
        # Fallback: Parabola
        return [[x, x * x] for x in range(-10, 10)]
