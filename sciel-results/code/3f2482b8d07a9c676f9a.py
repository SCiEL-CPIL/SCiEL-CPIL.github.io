import time
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270  # 60s safety margin for 330s limit
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = N * N
    
    def can_add(point, pts, dist_cache):
        """Check if point can be added to pts without creating isosceles triangle."""
        px, py = point
        for i in range(len(pts)):
            ax, ay = pts[i]
            d_pa = (px - ax) ** 2 + (py - ay) ** 2
            for j in range(i + 1, len(pts)):
                bx, by = pts[j]
                d_pb = (px - bx) ** 2 + (py - by) ** 2
                d_ab = dist_cache[i * len(pts) + j]
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    return False
        return True
    
    def greedy_build(order, pts, dist_cache):
        """Build set greedily following the given order."""
        for p in order:
            if can_add(p, pts, dist_cache):
                pts.append(p)
                # Update dist_cache for new point
                for i in range(len(pts) - 1):
                    ax, ay = pts[i]
                    px, py = p
                    d = (px - ax) ** 2 + (py - ay) ** 2
                    dist_cache[i * len(pts) + len(pts) - 1] = d
                    dist_cache[(len(pts) - 1) * len(pts) + i] = d
        return len(pts)
    
    best_size = 0
    best_set = []
    
    # Multiple random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = all_points[:]
        random.shuffle(order)
        
        pts = []
        # Pre-allocate dist_cache as dict for sparse access
        dist_cache = {}
        
        for p in order:
            if can_add(p, pts, dist_cache):
                pts.append(p)
                idx_new = len(pts) - 1
                for i in range(idx_new):
                    ax, ay = pts[i]
                    px, py = p
                    d = (px - ax) ** 2 + (py - ay) ** 2
                    dist_cache[(i, idx_new)] = d
                    dist_cache[(idx_new, i)] = d
        
        if len(pts) > best_size:
            best_size = len(pts)
            best_set = pts[:]
        
        # If we've found a large set, we can try local search improvements
        if best_size >= 15 and attempts > 10:
            # Try to add more points by trying all remaining points
            current_set = best_set[:]
            current_size = len(current_set)
            # Build dist_cache for current_set
            dc = {}
            for i in range(current_size):
                for j in range(i + 1, current_size):
                    ax, ay = current_set[i]
                    bx, by = current_set[j]
                    d = (ax - bx) ** 2 + (ay - by) ** 2
                    dc[(i, j)] = d
                    dc[(j, i)] = d
            
            # Try adding each point not in the set
            in_set = set(current_set)
            for p in all_points:
                if p not in in_set:
                    if can_add(p, current_set, dc):
                        current_set.append(p)
                        idx_new = len(current_set) - 1
                        for i in range(idx_new):
                            ax, ay = current_set[i]
                            px, py = p
                            d = (px - ax) ** 2 + (py - ay) ** 2
                            dc[(i, idx_new)] = d
                            dc[(idx_new, i)] = d
                        in_set.add(p)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    return best_set
# EVOLVE-BLOCK-END
