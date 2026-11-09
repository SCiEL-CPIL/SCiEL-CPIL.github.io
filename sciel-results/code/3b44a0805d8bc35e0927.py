"""Rewritten program for minimum maximal no-isosceles set."""
import time
import random
import numpy as np

N = 16
TIME_LIMIT = 290.0  # Safety margin before 330s kill

def _precompute_conflicts():
    """
    Precompute for each point p, the set of pairs (i, j) of other points
    such that {p, i, j} forms an isosceles triangle.
    Actually, we need: for a candidate set S, point p can be added iff
    for all pairs (a, b) in S, {p, a, b} is NOT isosceles.
    
    So for each point p and each pair (a, b), we check if isosceles.
    But that's O(N^2 * N^2) = O(N^4) which is 16^4 = 65536, fine.
    
    Better: precompute for each point p, a bitmask or set of points a such that
    for any b, if we add p, we must ensure no b in S makes {p, a, b} isosceles.
    
    Let's define: conflict[p][a] = set of b such that {p, a, b} is isosceles.
    Then p can be added to S iff for all a in S, S ∩ conflict[p][a] = empty.
    """
    points = [(x, y) for x in range(N) for y in range(N)]
    n = len(points)
    # conflict[p][a] = set of indices b (b != p, b != a) such that {p, a, b} is isosceles
    conflict = [dict() for _ in range(n)]
    for p in range(n):
        for a in range(n):
            if p == a:
                continue
            conflict[p][a] = set()
            for b in range(n):
                if b == p or b == a:
                    continue
                # Check if {p, a, b} is isosceles
                px, py = points[p]
                ax, ay = points[a]
                bx, by = points[b]
                d_pa = (px - ax)**2 + (py - ay)**2
                d_pb = (px - bx)**2 + (py - by)**2
                d_ab = (ax - bx)**2 + (ay - by)**2
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    conflict[p][a].add(b)
    return points, conflict

def _can_add(points, conflict, p, chosen_set):
    """Check if point p can be added to chosen_set without creating isosceles triangle."""
    for a in chosen_set:
        if a == p:
            continue
        # Check if any b in chosen_set is in conflict[p][a]
        for b in conflict[p][a]:
            if b in chosen_set:
                return False
    return True

def construct():
    start_time = time.time()
    points, conflict = _precompute_conflicts()
    n = len(points)
    
    # Greedy construction in row-major order
    chosen = []
    chosen_set = set()
    for p in range(n):
        if _can_add(points, conflict, p, chosen_set):
            chosen.append(p)
            chosen_set.add(p)
    
    best_size = len(chosen)
    best_chosen = list(chosen)
    
    # Local search: try to remove points and re-add others
    # Use simulated annealing or simple hill climbing with restarts
    rng = random.Random(42)
    
    def get_size(chosen_list):
        return len(chosen_list)
    
    # Try multiple random restarts with greedy + local improvement
    attempts = 0
    while time.time() - start_time < TIME_LIMIT - 5:
        attempts += 1
        
        # Randomized greedy
        order = list(range(n))
        rng.shuffle(order)
        chosen = []
        chosen_set = set()
        for p in order:
            if _can_add(points, conflict, p, chosen_set):
                chosen.append(p)
                chosen_set.add(p)
        
        # Local improvement: try to swap out one point for two others
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 5:
            improved = False
            for i in range(len(chosen)):
                # Try removing chosen[i]
                removed = chosen[i]
                chosen.remove(removed)
                chosen_set.remove(removed)
                
                # Try adding any point not in chosen
                for p in range(n):
                    if p in chosen_set:
                        continue
                    if _can_add(points, conflict, p, chosen_set):
                        # Can we add p? If yes, and if we can also add another, we improve
                        # But adding one point keeps size same. We need to add 2 to improve.
                        # Let's try adding p and then see if we can add another
                        chosen.append(p)
                        chosen_set.add(p)
                        # Try to add one more
                        added_second = False
                        for q in range(n):
                            if q in chosen_set:
                                continue
                            if _can_add(points, conflict, q, chosen_set):
                                chosen.append(q)
                                chosen_set.add(q)
                                added_second = True
                                break
                        if added_second:
                            # Size increased by 1 (removed 1, added 2)
                            improved = True
                            break
                        else:
                            # Revert: remove p
                            chosen.remove(p)
                            chosen_set.remove(p)
                
                if improved:
                    break
                
                # If no improvement, put removed back
                chosen.append(removed)
                chosen_set.add(removed)
            
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10:
            break
    
    # Return the best solution found
    return [points[i] for i in best_chosen]

# Keep the original function name
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
