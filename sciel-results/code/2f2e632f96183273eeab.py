<<<<<<< SEARCH
    # ---- Greedy Construction: Add points one by one to maximize edges ----
    def build_greedy_seed():
        pts = [(0, 0)]
        # Search radius for new points
        # The cluster grows roughly as sqrt(n). For n=32, radius ~ 5-6.
        # We'll search in a box of size [-10, 10] to be safe.
        
        for _ in range(31):
            best_cand = None
            best_score = -1
            best_dist_center = 999
            
            # Generate candidates
            # We only need to consider points that are "close" to the current cluster
            # to ensure connectivity.
            # Current bounding box of pts
            min_a = min(p[0] for p in pts)
            max_a = max(p[0] for p in pts)
            min_b = min(p[1] for p in pts)
            max_b = max(p[1] for p in pts)
            
            # Expand box by 2
            for a in range(min_a - 2, max_a + 3):
                for b in range(min_b - 2, max_b + 3):
                    if abs(a) > 64 or abs(b) > 64:
                        continue
                    cand = (a, b)
                    if cand in pts:
                        continue
                    # Check distance constraint
                    valid = True
                    for p in pts:
                        if dist_sq(cand, p) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: number of edges to existing points
                    edges = 0
                    for p in pts:
                        if dist_sq(cand, p) == 1:
                            edges += 1
                    
                    # Tie-breaker: closer to center (0,0) is better for compactness
                    # Use hex distance approximation: max(|a|, |b|, |a+b|) is not quite right for this coord system
                    # Just use euclidean dist squared to origin
                    d_center = a*a + a*b + b*b
                    
                    if edges > best_score or (edges == best_score and d_center < best_dist_center):
                        best_score = edges
                        best_dist_center = d_center
                        best_cand = cand
            
            if best_cand is None:
                # Fallback: random nearby point
                # This shouldn't happen if the box is large enough
                for _ in range(100):
                    da = random.randint(-5, 5)
                    db = random.randint(-5, 5)
                    cand = (pts[-1][0] + da, pts[-1][1] + db)
                    if abs(cand[0]) <= 64 and abs(cand[1]) <= 64 and cand not in pts:
                        valid = True
                        for p in pts:
                            if dist_sq(cand, p) < 1:
                                valid = False
                                break
                        if valid:
                            best_cand = cand
                            break
            
            if best_cand:
                pts.append(best_cand)
            else:
                # Error recovery
                break
                
        return pts[:32]
=======
    # ---- Spiral Construction: Deterministic hexagonal spiral seed ----
    def build_spiral_seed():
        pts = []
        # Generate points in hexagonal spiral order
        # Rings: 0 (1), 1 (6), 2 (12), 3 (18)
        # Total 37 points, we take first 32
        
        def add_ring(r):
            if r == 0:
                pts.append((0, 0))
                return
            # Traverse 6 sides of the hexagon ring r
            # Side 1: (r, 0) to (0, r)
            for i in range(r):
                pts.append((r - i, i))
            # Side 2: (0, r) to (-r, r)
            for i in range(r):
                pts.append((-i, r))
            # Side 3: (-r, r) to (-r, 0)
            for i in range(r):
                pts.append((-r, r - i))
            # Side 4: (-r, 0) to (0, -r)
            for i in range(r):
                pts.append((-r + i, -i))
            # Side 5: (0, -r) to (r, -r)
            for i in range(r):
                pts.append((i, -r))
            # Side 6: (r, -r) to (r, 0)
            for i in range(r):
                pts.append((r, -r + i))

        for r in range(0, 5):
            add_ring(r)
            if len(pts) >= 32:
                break
        
        return pts[:32]
>>>>>>> REPLACE

