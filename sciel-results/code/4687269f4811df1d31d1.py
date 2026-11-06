import numpy as np
import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # 1. Greedy Initialization using vectorized checks
    # We want to pick points one by one.
    # A set of points P is valid if for any distinct p1, p2, p3 in P,
    # the triangle (p1, p2, p3) is not isosceles.
    # This is equivalent to: for any p_new, and any pair (a, b) in P,
    # dist(p_new, a) != dist(p_new, b) AND dist(p_new, a) != dist(a, b) AND dist(p_new, b) != dist(a, b).
    
    # To make this fast, we can maintain a set of "forbidden" distance squares? 
    # No, the constraint depends on the specific pair (a,b).
    # If we add p, it creates isosceles triangles with any existing pair (a,b) if:
    # 1. |p-a|^2 == |p-b|^2
    # 2. |p-a|^2 == |a-b|^2
    # 3. |p-b|^2 == |a-b|^2
    
    # Let's use a randomized greedy approach.
    # We maintain a list of points.
    # We also maintain a set of "bad" positions? No, the constraint is global.
    
    # A faster check:
    # When adding p to set P:
    # For every a in P:
    #   d_pa = |p-a|^2
    #   For every b in P, b != a:
    #     d_pb = |p-b|^2
    #     d_ab = |a-b|^2
    #     If d_pa == d_pb or d_pa == d_ab or d_pb == d_ab: fail.
    
    # This is O(k^2) per insertion, where k is current size.
    # With k~20, k^2=400. 4000 candidates * 400 = 160k ops. Fast.
    
    # However, we can be smarter.
    # Let's just use a simple random greedy first to get a base, then improve.
    
    points = []
    candidates = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(candidates)
    
    def is_valid(points, p):
        if not points:
            return True
        px, py = p
        for i in range(len(points)):
            ax, ay = points[i]
            d_pa2 = (px - ax)**2 + (py - ay)**2
            for j in range(len(points)):
                if i == j: continue
                bx, by = points[j]
                d_pb2 = (px - bx)**2 + (py - by)**2
                d_ab2 = (ax - bx)**2 + (ay - by)**2
                if d_pa2 == d_pb2 or d_pa2 == d_ab2 or d_pb2 == d_ab2:
                    return False
        return True

    # Greedy Phase
    for p in candidates:
        if time.time() > deadline:
            break
        if is_valid(points, p):
            points.append(p)
            
    best_points = points[:]
    best_len = len(points)
    
    # 2. Local Search (Swap-based)
    # We try to improve the set by swapping a point in the set with a point not in the set.
    # Or just removing and re-adding? 
    # The lesson says "structured initialization and aggressive swap-based local search may escape local optima better".
    
    # We need a faster validity check for local search.
    # Let's precompute distances? No, points change.
    
    # Optimization for is_valid:
    # We can check constraints against the set more efficiently?
    # For a fixed p, the condition "d_pa2 == d_pb2" means p lies on the perpendicular bisector of ab.
    # The condition "d_pa2 == d_ab2" means p lies on a circle centered at a with radius |ab|.
    
    # Let's stick to the O(k^2) check but optimize the loop in Python by using lists of tuples.
    # Actually, for k~20, Python loops are okay-ish if we do many iterations.
    # But we have 300s. We can do thousands of swaps.
    
    # Let's implement a swap move:
    # Pick a random index i in points.
    # Pick a random candidate c not in points.
    # Check if (points - {points[i]} + {c}) is valid.
    # If valid, accept.
    # If invalid, maybe try a few more c's?
    
    # To speed up, we can pre-compute a set of "blocked" candidates?
    # No, the blockage depends on the whole set.
    
    # Let's try a different angle for local search:
    # Since checking is O(k^2), and k is small, we can afford to check many swaps.
    
    in_set = set(points)
    
    # Pre-compute list of all grid points for easy access
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    while time.time() < deadline:
        # Pick a random point to remove
        i = random.randint(0, len(points) - 1)
        p_remove = points[i]
        
        # Generate a few random candidates to replace it
        # We can bias towards points that are "far" from the rest? 
        # Or just random.
        
        # Let's try 10 random candidates
        improved = False
        for _ in range(10):
            c = random.choice(all_points)
            if c in in_set:
                continue
            
            # Construct new set
            # Instead of copying the whole list, we can check validity of c against (points - {p_remove})
            # And also ensure c doesn't form isosceles with p_remove? 
            # Wait, p_remove is gone. So we just need to check c against the remaining set.
            
            # Let's create a temporary list without p_remove
            # This is O(k).
            temp_points = [p for j, p in enumerate(points) if j != i]
            
            # Check if c is valid against temp_points
            if is_valid(temp_points, c):
                # Accept
                points[i] = c
                in_set.remove(p_remove)
                in_set.add(c)
                improved = True
                break
        
        # If we improved, we might want to try to add more? 
        # The swap keeps size constant.
        # We can occasionally try to add a point if possible?
        # But the greedy phase already tried to add.
        # Swaps help rearrange to allow future additions?
        # Actually, if we can add, we should.
        
        # Let's try to add a point every 50 swaps?
        # No, just do swaps. The size is fixed.
        # But maybe a different configuration allows a larger size?
        # If we swap, size stays same.
        # To increase size, we need to find a point that can be added.
        # Let's try to add a point every 100 iterations of the main loop.
        
        if random.random() < 0.05:
             # Try to add
             for c in random.sample(all_points, 50):
                 if c in in_set: continue
                 if is_valid(points, c):
                     points.append(c)
                     in_set.add(c)
                     break

    return [(int(x), int(y)) for x, y in points]
