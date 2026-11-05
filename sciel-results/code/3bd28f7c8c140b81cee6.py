<<<<<<< SEARCH
    def make_initial_config():
        # Try to create many 3-point lines
        # Use a pattern based on small integer coordinates
        configs = []
        
        # Configuration 1: Modified grid with some collinearities
        # 4x5 grid with some points shifted to create triples
        base = []
        for x in range(-4, 6):
            for y in range(-3, 3):
                base.append([x, y])
        # Shuffle and take 20
        random.shuffle(base)
        configs.append(base[:20])
        
        # Configuration 2: Lines of 3
        # Create 6 lines of 3 points each, sharing some points
        # Line 1: (0,0), (1,0), (2,0)
        # Line 2: (0,1), (1,1), (2,1)
        # Line 3: (0,2), (1,2), (2,2)
        # Line 4: (3,0), (3,1), (3,2)
        # Line 5: (4,0), (4,1), (4,2)
        # Line 6: (0,3), (1,3), (2,3)
        # That's 18 points, need 2 more
        configs.append([
            [0,0], [1,0], [2,0],
            [0,1], [1,1], [2,1],
            [0,2], [1,2], [2,2],
            [3,0], [3,1], [3,2],
            [4,0], [4,1], [4,2],
            [0,3], [1,3], [2,3],
            [5,0], [5,1]
        ])
        
        # Configuration 3: More dense collinear structure
        # Use a 5x4 grid
        configs.append([[x, y] for x in range(-2, 3) for y in range(-2, 2)][:20])
        
        # Filter valid configs
        valid_configs = []
        for cfg in configs:
            if len(cfg) == 20 and is_valid(cfg):
                valid_configs.append(cfg)
        
        if valid_configs:
            return random.choice(valid_configs)
        
        # Fallback: random
        pts = []
        for _ in range(100):
            pts = []
            while len(pts) < 20:
                p = random_point()
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return pts
        
        # Last resort: parabola (score 20, but valid)
        return [[x, x * x] for x in range(-10, 10)]
=======
    def make_initial_config():
        """Create initial configurations with many collinear triples."""
        configs = []
        
        # Configuration 1: 5x4 grid (20 points)
        configs.append([[x, y] for x in range(-2, 3) for y in range(-2, 2)])
        
        # Configuration 2: 4x5 grid
        configs.append([[x, y] for x in range(-2, 2) for y in range(-3, 2)])
        
        # Configuration 3: Lines of 3 with shared points (Fano-like)
        configs.append([
            [0,0], [1,0], [2,0],
            [0,1], [1,1], [2,1],
            [0,2], [1,2], [2,2],
            [3,0], [3,1], [3,2],
            [4,0], [4,1], [4,2],
            [0,3], [1,3], [2,3],
            [5,0], [5,1]
        ])
        
        # Configuration 4: Dense small grid
        configs.append([[x, y] for x in range(-3, 3) for y in range(-2, 2)][:20])
        
        # Filter valid configs and pick the one with most triples
        valid_configs = []
        best_cfg = None
        best_triples = -1
        for cfg in configs:
            if len(cfg) == 20 and is_valid(cfg):
                valid_configs.append(cfg)
                sc = calc_score(cfg)
                if sc < best_triples:  # more negative = more triples
                    best_triples = sc
                    best_cfg = cfg
        
        if best_cfg:
            return [p[:] for p in best_cfg]
        
        if valid_configs:
            return [p[:] for p in random.choice(valid_configs)]
        
        # Fallback: random
        for _ in range(100):
            pts = []
            while len(pts) < 20:
                p = random_point()
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return pts
        
        # Last resort: parabola
        return [[x, x * x] for x in range(-10, 10)]
>>>>>>> REPLACE

