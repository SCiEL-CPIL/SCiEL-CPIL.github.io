import random
import time

N = 64

def construct():
    start_time = time.time()
    time_limit = 290  # Safety margin
    
    # Precompute all points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = len(points)
    
    best_size = 0
    best_set = []
    
    # Helper to check if P can be added to S
    # S is a list of points
    # dist_sets is a dict: point -> set of squared distances to other points in S
    def try_add(P, S, dist_sets):
        # 1. Check distances from P to all Q in S
        dists_from_P = []
        for Q in S:
            dx = P[0] - Q[0]
            dy = P[1] - Q[1]
            d2 = dx*dx + dy*dy
            dists_from_P.append((d2, Q))
        
        # Check if dists_from_P are distinct
        seen_dists = set()
        for d2, Q in dists_from_P:
            if d2 in seen_dists:
                return False
            seen_dists.add(d2)
            
        # Check if d2 is new for Q
        for d2, Q in dists_from_P:
            if d2 in dist_sets[Q]:
                return False
                
        return True

    # Main search loop
    for seed in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        S = []
        dist_sets = {} # point -> set of squared distances
        
        candidates = points[:]
        random.shuffle(candidates)
        
        for P in candidates:
            if try_add(P, S, dist_sets):
                # Add P
                S.append(P)
                dist_sets[P] = set()
                # Update dist_sets for existing points
                for Q in S[:-1]:
                    dx = P[0] - Q[0]
                    dy = P[1] - Q[1]
                    d2 = dx*dx + dy*dy
                    dist_sets[Q].add(d2)
                    dist_sets[P].add(d2)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
