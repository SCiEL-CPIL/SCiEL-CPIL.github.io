<<<<<<< SEARCH
    best_score = float('inf')
    best_adj = None
    
    # Try multiple initial configurations
    # 1. K_8,8 + 1 edge
    # 2. Random graphs with 65 edges
    
    num_restarts = 20
    random.seed(42)
    
    for r in range(num_restarts):
=======
    best_score = float('inf')
    best_adj = None
    
    # Try multiple initial configurations
    # 1. K_8,8 + 1 edge
    # 2. Random graphs with 65 edges
    
    num_restarts = 50
    random.seed(42)
    
    for r in range(num_restarts):
>>>>>>> REPLACE

