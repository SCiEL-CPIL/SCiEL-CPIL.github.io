"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320  # Safety margin before 330s kill
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def is_valid(point_set):
        """Check if no three points form an isosceles triangle."""
        n = len(point_set)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = point_set[i], point_set[j], point_set[k]
                    d12 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                    d23 = (p2[0] - p3[0])**2 + (p2[1] - p3[1])**2
                    d13 = (p1[0] - p3[0])**2 + (p1[1] - p3[1])**2
                    if d12 == d23 or d12 == d13 or d23 == d13:
                        return False
        return True
    
    def greedy_with_ordering(order):
        """Greedy construction using a specific ordering of points."""
        selected = []
        selected_set = set()
        
        for p in order:
            if p in selected_set:
                continue
            # Check if adding p creates an isosceles triangle with any two existing points
            valid = True
            n = len(selected)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = selected[i], selected[j]
                    d1p = (p1[0] - p[0])**2 + (p1[1] - p[1])**2
                    d2p = (p2[0] - p[0])**2 + (p2[1] - p[1])**2
                    d12 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                    if d1p == d2p or d1p == d12 or d2p == d12:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(p)
                selected_set.add(p)
        
        return selected
    
    def local_improvement(point_set, max_iters=500):
        """Try to improve the set by swapping points."""
        current = list(point_set)
        current_set = set(current)
        best = current[:]
        best_len = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding a random point not in the set
            candidates = [p for p in all_points if p not in current_set]
            if candidates:
                p = random.choice(candidates)
                trial = current + [p]
                if is_valid(trial):
                    current = trial
                    current_set.add(p)
                    if len(current) > best_len:
                        best = current[:]
                        best_len = len(current)
                    improved = True
            
            # Try removing a random point and adding a different one
            if not improved and len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                trial = current[:idx] + current[idx+1:]
                current_set.discard(removed)
                
                candidates = [p for p in all_points if p not in current_set]
                if candidates:
                    p = random.choice(candidates)
                    trial2 = trial + [p]
                    if is_valid(trial2):
                        current = trial2
                        current_set.add(p)
                        if len(current) > best_len:
                            best = current[:]
                            best_len = len(current)
                        improved = True
                    else:
                        # Revert
                        current = trial + [removed]
                        current_set.add(removed)
            
            if not improved:
                break
        
        return best
    
    # Try multiple different orderings
    random.seed(42)
    
    best_result = []
    best_len = 0
    
    # Strategy 1: Random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        order = all_points[:]
        random.shuffle(order)
        result = greedy_with_ordering(order)
        if len(result) > best_len:
            best_result = result
            best_len = len(result)
    
    # Strategy 2: Distance-based ordering (start from center)
    cx, cy = N // 2, N // 2
    order_by_dist = sorted(all_points, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    result = greedy_with_ordering(order_by_dist)
    if len(result) > best_len:
        best_result = result
        best_len = len(result)
    
    # Strategy 3: Spiral ordering
    order_spiral = []
    for r in range(N // 2 + 1):
        for x in range(max(0, cx-r), min(N, cx+r+1)):
            for y in range(max(0, cy-r), min(N, cy+r+1)):
                if abs(x-cx) == r or abs(y-cy) == r:
                    order_spiral.append((x, y))
    result = greedy_with_ordering(order_spiral)
    if len(result) > best_len:
        best_result = result
        best_len = len(result)
    
    # Strategy 4: Local improvement on best result
    if best_result:
        improved = local_improvement(best_result, max_iters=1000)
        if len(improved) > best_len:
            best_result = improved
            best_len = len(improved)
    
    # Strategy 5: More random trials with local improvement
    for trial in range(20):
        if time.time() > deadline:
            break
        order = all_points[:]
        random.shuffle(order)
        result = greedy_with_ordering(order)
        if len(result) > best_len:
            best_result = result
            best_len = len(result)
        improved = local_improvement(result, max_iters=200)
        if len(improved) > best_len:
            best_result = improved
            best_len = len(improved)
    
    return best_result
# EVOLVE-BLOCK-END
