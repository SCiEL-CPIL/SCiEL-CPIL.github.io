import time
import random

N = 64

def squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    best_set = []
    best_size = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def is_valid_addition(p, s_list, dist_sets):
        """Check if adding p to s_list creates any isosceles triangle."""
        seen_dists = set()
        sd = squared_dist
        for i in range(len(s_list)):
            a = s_list[i]
            d = sd(p, a)
            # Check apex at p: distance to p must be unique among distances to other points in S
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Check apex at a: distance p-a must not be equal to distance a-b for any b in S, b != a
            # dist_sets[i] contains distances from a to all other points in S
            if d in dist_sets[i]:
                return False
        return True

    def add_point(p, s_list, dist_sets):
        """Add point p to s_list and update dist_sets."""
        new_dist_set = set()
        sd = squared_dist
        for i in range(len(s_list)):
            a = s_list[i]
            d = sd(p, a)
            new_dist_set.add(d)
            # Update existing dist_sets
            dist_sets[i].add(d)
        
        s_list.append(p)
        dist_sets.append(new_dist_set)

    num_restarts = 2000 # Increased restarts due to faster check
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering of points
        shuffled = all_points[:]
        random.shuffle(shuffled)
        
        s_list = []
        dist_sets = []
        
        for p in shuffled:
            if time.time() > deadline:
                break
            if is_valid_addition(p, s_list, dist_sets):
                add_point(p, s_list, dist_sets)
        
        if len(s_list) > best_size:
            best_size = len(s_list)
            best_set = s_list[:]
            
    return best_set
