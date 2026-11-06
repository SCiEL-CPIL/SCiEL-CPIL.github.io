import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 285.0
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def greedy_construct(seed):
        random.seed(seed)
        shuffled_points = all_points[:]
        random.shuffle(shuffled_points)
        
        current_set = []
        dist_map = {}
        
        for p in shuffled_points:
            valid = True
            dists_from_p = []
            
            for q in current_set:
                d = sq_dist(p, q)
                dists_from_p.append(d)
                if d in dist_map[q]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if len(dists_from_p) != len(set(dists_from_p)):
                continue
            
            current_set.append(p)
            dist_map[p] = set(dists_from_p)
            
            for i in range(len(current_set) - 1):
                q = current_set[i]
                dist_map[q].add(dists_from_p[i])
                
        return current_set, dist_map

    def try_add_point(p, current_set, dist_map):
        """Try to add point p. Returns (success, dists_from_p)"""
        dists_from_p = []
        for q in current_set:
            d = sq_dist(p, q)
            dists_from_p.append(d)
            if d in dist_map[q]:
                return False, None
        
        if len(dists_from_p) != len(set(dists_from_p)):
            return False, None
        
        return True, dists_from_p

    def local_search(seed, max_iters=2000):
        """
        Start with greedy, then try to improve by adding points.
        Uses a queue of candidate points and tries to add them.
        """
        random.seed(seed)
        current_set, dist_map = greedy_construct(seed)
        
        # Build a list of points not in current_set
        current_set_set = set(current_set)
        candidates = [p for p in all_points if p not in current_set_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in candidates[:]:
                if p in current_set_set:
                    continue
                success, dists = try_add_point(p, current_set, dist_map)
                if success:
                    current_set.append(p)
                    current_set_set.add(p)
                    dist_map[p] = set(dists)
                    for i in range(len(current_set) - 1):
                        q = current_set[i]
                        dist_map[q].add(dists[i])
                    candidates.remove(p)
                    improved = True
                    break
        
        return current_set

    best_set = []
    best_len = 0
    
    seed = 0
    while time.time() < deadline:
        result = local_search(seed)
        
        if len(result) > best_len:
            best_set = result
            best_len = len(result)
            
        seed += 1
        
    return best_set
# EVOLVE-BLOCK-END
