import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Precompute all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def greedy_construct(seed):
        """
        Greedy construction with random shuffling.
        Maintains dist_map: point -> set of squared distances to other points in current set.
        """
        random.seed(seed)
        shuffled_points = all_points[:]
        random.shuffle(shuffled_points)
        
        current_set = []
        dist_map = {}
        
        for p in shuffled_points:
            valid = True
            dists_from_p = []
            
            # Check against existing points
            for q in current_set:
                d = sq_dist(p, q)
                dists_from_p.append(d)
                
                # If q already has a point at distance d, adding p creates isosceles with apex q
                if d in dist_map[q]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # If p has two points at the same distance, adding p creates isosceles with apex p
            if len(dists_from_p) != len(set(dists_from_p)):
                continue
            
            # Accept p
            current_set.append(p)
            dist_map[p] = set(dists_from_p)
            
            # Update dist_map for all existing points
            # current_set[-1] is p, so iterate up to -1
            for i in range(len(current_set) - 1):
                q = current_set[i]
                dist_map[q].add(dists_from_p[i])
                
        return current_set

    best_set = []
    best_len = 0
    
    seed = 0
    while time.time() < deadline:
        # Run greedy
        result = greedy_construct(seed)
        
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
            
        seed += 1
        
    return best_set
# EVOLVE-BLOCK-END
