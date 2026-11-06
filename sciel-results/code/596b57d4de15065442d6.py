import time
import random
import numpy as np

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
        # Build a set of lines (normalized) and count points on each
        # For 20 points, O(n^3) is fine
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

    def count_3_collinear(points):
        """Count the number of collinear triples. More triples = lower potential score."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def max_no_3_collinear(points):
        """Exact branch and bound for largest subset with no three collinear."""
        n = len(points)
        
        # Precompute collinearity masks
        # coll_pair_mask[i][j] = bitmask of k such that i, j, k are collinear (k != i, j)
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
        
        # Precompute a heuristic ordering: points with more collinear partners first
        deg = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if coll_pair_mask[i][j]:
                    deg[i] += 1
                    deg[j] += 1
        
        # We'll use a bitmask-based branch and bound
        # State: A (selected set as bitmask), C (candidate set as bitmask)
        # When we add p to A, we remove from C any q such that there exists r in A 
        # where p, q, r are collinear. This is equivalent to: 
        # new_C = C & ~ (union over a in A of coll_pair_mask[p][a])
        
        # Also, we need to ensure p is compatible with A: 
        # for all a, b in A, p, a, b are not collinear.
        # This is equivalent to: for all a in A, (coll_pair_mask[p][a] & A) == 0
        
        def backtrack(A, C, depth):
            # Pruning: even if we take all remaining, we can't beat best
            remaining = bin(C).count('1')
            if depth + remaining <= best[0]:
                return
            
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Heuristic: pick the point in C with highest degree (most conflicts)
            # This helps prune earlier
            # For speed, we'll just pick the lowest set bit, but let's try a slightly better heuristic
            # Find point with max deg in C
            max_deg = -1
            p = -1
            temp_C = C
            while temp_C:
                bit = temp_C & -temp_C
                idx = bit.bit_length() - 1
                if deg[idx] > max_deg:
                    max_deg = deg[idx]
                    p = idx
                temp_C ^= bit
            
            if p == -1:
                return
            
            p_bit = 1 << p
            
            # Option 1: Exclude p
            backtrack(A, C ^ p_bit, depth)
            
            # Option 2: Include p (if compatible)
            # Check compatibility: for all a in A, (coll_pair_mask[p][a] & A) == 0
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
                # Remove from C any q that would form a triple with p and some a in A
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
    best_triples = 0  # Track number of collinear triples for tie-breaking
    
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        # Strategy: Perturbed grid with more aggressive perturbation
        # Try different grid sizes and perturbation patterns
        
        if attempts <= 5000:
            # 5x4 grid with random shifts
            pts = []
            for i in range(5):
                for j in range(4):
                    di = random.choice([-2, -1, 0, 0, 1, 2])
                    dj = random.choice([-2, -1, 0, 0, 1, 2])
                    pts.append([i + di, j + dj])
        elif attempts <= 10000:
            # 4x5 grid
            pts = []
            for i in range(4):
                for j in range(5):
                    di = random.choice([-2, -1, 0, 0, 1, 2])
                    dj = random.choice([-2, -1, 0, 0, 1, 2])
                    pts.append([i + di, j + dj])
        elif attempts <= 15000:
            # 6x4 grid (24 points, pick 20)
            pts = []
            for i in range(6):
                for j in range(4):
                    di = random.choice([-1, 0, 0, 1])
                    dj = random.choice([-1, 0, 0, 1])
                    pts.append([i + di, j + dj])
            random.shuffle(pts)
            pts = pts[:20]
        elif attempts <= 20000:
            # Random in [-8, 8]
            pts = []
            for _ in range(20):
                x = random.randint(-8, 8)
                y = random.randint(-8, 8)
                pts.append([x, y])
        else:
            # More aggressive: try to create many collinear triples
            # Start with a grid and add controlled perturbations
            base = []
            for i in range(5):
                for j in range(4):
                    base.append([i, j])
            
            pts = base[:]
            # Randomly swap some points
            for _ in range(random.randint(2, 6)):
                idx = random.randint(0, 19)
                pts[idx] = [random.randint(-5, 5), random.randint(-5, 5)]
        
        # Check distinct
        if len(set(map(tuple, pts))) < 20:
            continue
        
        # Check no 4 collinear
        if has_4_collinear(pts):
            continue
        
        # Count triples as a heuristic for quality
        triples = count_3_collinear(pts)
        
        # Compute exact score
        score = max_no_3_collinear(pts)
        
        if score < best_score or (score == best_score and triples > best_triples):
            best_score = score
            best_config = pts
            best_triples = triples
            if best_score <= 10:
                break
    
    if best_config is None:
        # Fallback: Try a fixed configuration known to have many collinear triples
        # 5x4 grid
        return [[i, j] for i in range(5) for j in range(4)]
    
    return best_config
