import time
import random
import sys

N = 64

# Precompute all grid points
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(ALL_POINTS)

# Precompute squared distances between all pairs?
# 4096 * 4096 / 2 = 8 million pairs.
# Storing in a list of lists or dict might be memory heavy or slow to init.
# Calculating on the fly is $O(1)$ per pair.
# In the inner loop, we calculate $d^2(P, A)$.
# $d^2 = (P_x - A_x)^2 + (P_y - A_y)^2$.
# This is fast.

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    best_set = []
    best_size = 0
    
    # We will run multiple greedy attempts
    # Each attempt shuffles the points and greedily adds them
    # We can also try different initial points or heuristics
    
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        random.shuffle(ALL_POINTS)
        
        S = []  # List of points in the set
        # DistSet[A] is a set of squared distances from A to other points in S
        # We need to map point to its index or just use the point tuple as key?
        # Using point tuple as key in a dict is fine.
        dist_sets = {}
        
        # To speed up the "check if d2 is in dist_sets[A]" and "check if d2 in local_dists",
        # we maintain a local list/set for the current candidate P's distances to S.
        
        for P in ALL_POINTS:
            # Check if P can be added to S
            local_dists = []
            valid = True
            
            # Optimization: If |S| is large, this loop is the bottleneck.
            # We iterate through all A in S.
            for A in S:
                d2 = dist_sq(P, A)
                
                # Check 1: Is d2 already in local_dists? (Apex at P)
                # Using a set for local_dists is O(1) lookup, but insertion is O(1).
                # However, we need to check against the set of distances from A to others.
                
                # We can't just use a set for local_dists if we want to check against dist_sets[A] efficiently?
                # Actually, we just check if d2 is in dist_sets[A].
                # And we check if d2 is in the set of distances we've already calculated for P.
                
                # Let's use a set for local_dists for O(1) duplicate check.
                if d2 in local_dists_set:
                    valid = False
                    break
                
                local_dists_set.add(d2)
                
                # Check 2: Is d2 in dist_sets[A]? (Apex at A)
                # dist_sets[A] contains distances from A to all B in S (B != A).
                if d2 in dist_sets[A]:
                    valid = False
                    break
            
            if valid:
                # Add P to S
                S.append(P)
                
                # Update dist_sets
                # Create dist_sets[P]
                # It should contain distances from P to all A in S (excluding P, but P is new so all A are distinct)
                # We already calculated these distances in the loop above?
                # Wait, if we broke early, we didn't calculate all.
                # If valid, we went through all A.
                # So we have the distances. But we didn't store them in a list, only in a set.
                # We need the list of distances to build dist_sets[P]?
                # Actually, dist_sets[P] is a set. So we just use the local_dists_set.
                dist_sets[P] = local_dists_set
                
                # Update dist_sets for existing A
                for A in S:
                    if A == P:
                        continue
                    d2 = dist_sq(P, A)
                    dist_sets[A].add(d2)
                    
                # Note: The inner loop for updating dist_sets[A] is O(|S|).
                # So adding a point is O(|S|) for check + O(|S|) for update = O(|S|).
                # Total for one greedy run is O(N * |S|).
                
                # Early termination if we exceed best_size? No, we want the full set.
                
                # Optimization: If |S| is already larger than best_size, we can keep going.
                # But we can't stop early because we might add more.
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we found a very large set, we might want to stop early?
        # Or just let it run until deadline.
        
    return best_set

# Need to fix the logic in the loop above:
# 1. local_dists_set needs to be initialized for each P.
# 2. The update loop `for A in S: if A == P: continue` is inefficient because P is the last element.
#    We can just iterate over S[:-1].

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    # Use a list of points for shuffling
    points = ALL_POINTS[:]
    
    while time.time() < deadline:
        random.shuffle(points)
        
        S = []
        dist_sets = {}
        
        for P in points:
            local_dists_set = set()
            valid = True
            
            for A in S:
                d2 = dist_sq(P, A)
                
                if d2 in local_dists_set:
                    valid = False
                    break
                
                local_dists_set.add(d2)
                
                if d2 in dist_sets[A]:
                    valid = False
                    break
            
            if valid:
                S.append(P)
                dist_sets[P] = local_dists_set
                
                # Update existing dist_sets
                # S[:-1] are the previous points
                for A in S[:-1]:
                    d2 = dist_sq(P, A)
                    dist_sets[A].add(d2)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
