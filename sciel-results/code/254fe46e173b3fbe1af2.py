    def generate_candidates(points, idx):
        candidates = set()
        p_old = points[idx]
        candidates.add(p_old)
        
        other_indices = [i for i in range(16) if i != idx]
        # Sample a subset of pairs to keep it fast? No, 105 is fine.
        for i, j in itertools.combinations(other_indices, 2):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
            
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            # Find one random integer point on this line within bounds
            # x = p1[0] + t * dx_n
            # y = p1[1] + t * dy_n
            
            # Determine valid t range
            min_t = float('-inf')
            max_t = float('inf')
            
            if dx_n > 0:
                min_t = max(min_t, math.ceil((-1000 - p1[0]) / dx_n))
                max_t = min(max_t, math.floor((1000 - p1[0]) / dx_n))
            elif dx_n < 0:
                min_t = max(min_t, math.ceil((1000 - p1[0]) / dx_n))
                max_t = min(max_t, math.floor((-1000 - p1[0]) / dx_n))
            
            if dy_n > 0:
                min_t = max(min_t, math.ceil((-1000 - p1[1]) / dy_n))
                max_t = min(max_t, math.floor((1000 - p1[1]) / dy_n))
            elif dy_n < 0:
                min_t = max(min_t, math.ceil((1000 - p1[1]) / dy_n))
                max_t = min(max_t, math.floor((-1000 - p1[1]) / dy_n))
            
            if max_t < min_t:
                continue
                
            min_t = int(min_t)
            max_t = int(max_t)
            
            # Pick a random t
            t = random.randint(min_t, max_t)
            x = p1[0] + t * dx_n
            y = p1[1] + t * dy_n
            candidates.add((x, y))
            
        # Add some random points
        for _ in range(5):
            candidates.add((random.randint(-1000, 1000), random.randint(-1000, 1000)))
            
        return list(candidates)
