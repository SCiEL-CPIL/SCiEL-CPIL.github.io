"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
from collections import Counter

# EVOLVE-BLOCK-START
import random
import numpy as np

class AreaTracker:
    def __init__(self, points):
        self.points = list(points)
        self.n = len(self.points)
        self.area_counts = Counter()
        self._recompute()

    def _recompute(self):
        self.area_counts = Counter()
        n = self.n
        pts = self.points
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    area = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if area > 0:
                        self.area_counts[area] += 1
        self.max_mult = max(self.area_counts.values()) if self.area_counts else 0

    def get_score(self):
        return self.max_mult

    def swap_point(self, idx, new_pt):
        """
        Swap point at idx with new_pt.
        Returns the new score.
        """
        old_pt = self.points[idx]
        
        # 1. Remove contributions of old_pt
        # Triples involving idx: (idx, j, k) where j,k != idx
        # We iterate over all pairs (j, k) from the other points
        others = [p for i, p in enumerate(self.points) if i != idx]
        # This is O(N) to build, but we can just loop over indices
        # To be efficient, we loop over all j, k in [0, n) excluding idx
        
        # Actually, simpler:
        # For every pair (j, k) such that j != idx and k != idx and j < k:
        # The triple {idx, j, k} is removed.
        
        for j in range(self.n):
            if j == idx: continue
            for k in range(j+1, self.n):
                if k == idx: continue
                # The triple is {idx, j, k}
                x1, y1 = old_pt
                x2, y2 = self.points[j]
                x3, y3 = self.points[k]
                area = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if area > 0:
                    self.area_counts[area] -= 1
                    if self.area_counts[area] == 0:
                        del self.area_counts[area]

        # 2. Update point
        self.points[idx] = new_pt
        
        # 3. Add contributions of new_pt
        for j in range(self.n):
            if j == idx: continue
            for k in range(j+1, self.n):
                if k == idx: continue
                # The triple is {idx, j, k}
                x1, y1 = new_pt
                x2, y2 = self.points[j]
                x3, y3 = self.points[k]
                area = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if area > 0:
                    self.area_counts[area] += 1

        # 4. Update max multiplicity
        # We could recalculate max, but that's O(M) where M is num distinct areas.
        # Given small N, M is small.
        if self.area_counts:
            self.max_mult = max(self.area_counts.values())
        else:
            self.max_mult = 0
            
        return self.max_mult

def _try_grid(n=30):
    """Try all grid shapes that give exactly n points and find the best."""
    best_pts = None
    best_score = -1
    # factor pairs of 30
    factors = [(1,30),(2,15),(3,10),(5,6),(6,5),(10,3),(15,2),(30,1)]
    for r, c in factors:
        # check if it fits in [-30,30]^2
        # We'll center it
        offx = -c // 2
        offy = -r // 2
        pts = [(offx + x, offy + y) for x in range(c) for y in range(r)]
        # check bounds
        if any(x < -30 or x > 30 or y < -30 or y > 30 for x,y in pts):
            continue
        tracker = AreaTracker(pts)
        s = tracker.get_score()
        if s > best_score:
            best_score = s
            best_pts = list(pts)
    return best_pts, best_score

def _random_points(n=30, bounds=30):
    """Generate n random distinct integer points in [-bounds, bounds]^2."""
    pts = set()
    while len(pts) < n:
        x = random.randint(-bounds, bounds)
        y = random.randint(-bounds, bounds)
        pts.add((x, y))
    return list(pts)

def _local_search(points, deadline):
    """Perform local search to improve the score using incremental updates."""
    tracker = AreaTracker(points)
    best_pts = list(points)
    best_score = tracker.get_score()
    n = len(best_pts)
    
    pts_set = set(best_pts)
    
    while time.time() < deadline:
        # Try swapping one point with a random new point
        i = random.randint(0, n-1)
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_pt = (new_x, new_y)
        
        # Check distinctness
        if new_pt in pts_set:
            continue
        
        old_pt = best_pts[i]
        
        # Perform swap in tracker
        new_score = tracker.swap_point(i, new_pt)
        
        # Update sets
        pts_set.remove(old_pt)
        pts_set.add(new_pt)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = list(tracker.points)
        else:
            # Revert
            tracker.swap_point(i, old_pt)
            pts_set.remove(new_pt)
            pts_set.add(old_pt)
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # Start with the best grid configuration
    pts, score = _try_grid(30)
    if pts is None:
        # Fallback: 6x5 grid centered
        pts = [(x-2, y-2) for x in range(6) for y in range(5)]
        score = 0 # Will be calculated in tracker
        
    best_pts = list(pts)
    best_score = score
    
    random.seed(42)
    
    # First, do a deep local search from the grid start
    # Use a generous time limit for the first search
    search_deadline = min(deadline, start_time + 20)
    search_pts, search_score = _local_search(best_pts, search_deadline)
    if search_score > best_score:
        best_score = search_score
        best_pts = search_pts

    # Then try random restarts
    # Each restart gets a smaller time slice
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_restarts = int(remaining_time / 2) # ~2s per restart
        for attempt in range(num_restarts):
            if time.time() > deadline - 1:
                break
            
            # Random initial points
            rand_pts = _random_points(30, 30)
            
            # Local search from random start
            # Limit each search to 1.5s
            search_deadline = min(deadline, time.time() + 1.5)
            search_pts, search_score = _local_search(rand_pts, search_deadline)
            
            if search_score > best_score:
                best_score = search_score
                best_pts = search_pts
                # If we found a new best, do a bit more search on it
                if time.time() < deadline - 2:
                    search_deadline2 = min(deadline, time.time() + 2.0)
                    search_pts2, search_score2 = _local_search(best_pts, search_deadline2)
                    if search_score2 > best_score:
                        best_score = search_score2
                        best_pts = search_pts2

    return best_pts
# EVOLVE-BLOCK-END
