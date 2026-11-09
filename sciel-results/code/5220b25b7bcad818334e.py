import random
import time
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    best_set = []
    best_size = 0
    
    # Helper to check if a point can be added
    def can_add(p, S, dists):
        # dists is a dict: point -> set of squared distances to other points in S
        new_dists = set()
        for q in S:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d2 = dx*dx + dy*dy
            if d2 in dists[q]:
                return False
            if d2 in new_dists:
                return False
            new_dists.add(d2)
        return True

    def add_point(p, S, dists):
        S.append(p)
        new_dists = set()
        for q in S[:-1]: # Exclude p itself
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d2 = dx*dx + dy*dy
            new_dists.add(d2)
            dists[q].add(d2)
        dists[p] = new_dists

    def remove_point(p, S, dists):
        S.remove(p)
        for q in S:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d2 = dx*dx + dy*dy
            dists[q].discard(d2)
        del dists[p]

    # Iterated Greedy
    for seed in range(200):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        S = []
        dists = {}
        
        candidates = all_points[:]
        random.shuffle(candidates)
        
        # Greedy fill
        for p in candidates:
            if can_add(p, S, dists):
                add_point(p, S, dists)
                
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S[:]
            
        # Local search / Perturbation
        # Try to improve by removing and re-adding
        # This part is optional if time is tight, but let's do a simple version
        # We'll do a few rounds of: remove k random points, try to re-add them and others
        for _ in range(5):
            if time.time() > deadline:
                break
                
            # Remove a few random points
            remove_count = max(1, current_size // 20)
            to_remove = random.sample(S, min(remove_count, len(S)))
            for p in to_remove:
                remove_point(p, S, dists)
                
            # Try to re-add removed points and others
            # Create a list of points not in S
            S_set = set(S)
            not_in_S = [p for p in all_points if p not in S_set]
            random.shuffle(not_in_S)
            
            # First try to re-add the removed ones
            random.shuffle(to_remove)
            for p in to_remove:
                if can_add(p, S, dists):
                    add_point(p, S, dists)
                    
            # Then try to add other points
            for p in not_in_S:
                if time.time() > deadline:
                    break
                if can_add(p, S, dists):
                    add_point(p, S, dists)
                    
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size
                best_set = S[:]
                
    return best_set
