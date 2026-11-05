import itertools
import time
from collections import Counter
import random
import numpy as np

# EVOLVE-BLOCK-START

def _doubled_areas(points):
    """Compute doubled areas (abs determinant) for all triples of points."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
                d = abs(d)
                if d > 0:
                    areas.append(d)
    return areas

def _score(points):
    """Return the largest multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _generate_grid(r, c, ox, oy):
    return [(ox + x, oy + y) for x in range(c) for y in range(r)]

def _initial_grid():
    # 5x6 grid centered roughly at 0
    # x in -2..3, y in -2..2
    return _generate_grid(5, 6, -2, -2)

def _incremental_search(points, time_limit):
    """
    Local search with incremental area histogram updates.
    """
    start_time = time.time()
    pts = list(points)
    n = len(pts)
    
    # Initial histogram
    hist = Counter(_doubled_areas(pts))
    if not hist:
        return pts, 0
    best_score = max(hist.values())
    
    # Precompute indices for faster access?
    # We will iterate over points i, and for each, consider moving it.
    
    # To make moves, we need to know which points are occupied.
    # We'll use a set for O(1) lookup.
    
    # Candidate generation:
    # For a point (x,y), try moving to (x+dx, y+dy) for small dx,dy?
    # Or random points in the bounding box?
    # Random points are better for escaping local minima.
    
    # Let's try a mix:
    # 1. Random swap with another point in the set? No, that just permutes.
    # 2. Move one point to a random empty location in [-30,30]^2.
    
    # Optimization:
    # When moving point i from old to new:
    # 1. Remove contributions of triples involving i.
    # 2. Add contributions of triples involving i at new position.
    # 3. Check if max(hist) increased.
    
    # Complexity per move: O(N^2)
    # N=30, N^2=900.
    # We can do ~1000 moves per second easily.
    # In 50s, we can do ~50,000 moves.
    
    # To avoid getting stuck, we can also try to change the "target" area implicitly
    # by accepting moves that don't immediately improve the *current* max but might
    # enable a different area to become dominant.
    # However, simple hill climbing on the max frequency is a good start.
    
    # Let's add a "kick" mechanism: if stuck for X iterations, do a random restart or
    # accept a neutral/worse move temporarily (simulated annealing).
    
    temperature = 0.0
    no_improve_count = 0
    max_no_improve = 500
    
    # List of all possible integer points in bounds
    # 61x61 = 3721 points.
    # We can pre-generate a list of all valid points.
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    # Map point to index? No, set is fine for membership.
    
    while time.time() - start_time < time_limit:
        # Pick a random point index
        i = random.randint(0, n-1)
        old_pt = pts[i]
        
        # Generate a candidate new point
        # Try a few random candidates and pick the best one
        best_candidate = None
        best_candidate_score = best_score - 1 # We want improvement
        
        # Try 5 random candidates
        for _ in range(5):
            new_pt = random.choice(all_points)
            if new_pt == old_pt or new_pt in pts:
                continue
            
            # Calculate new histogram
            # We need to update hist for triples involving i.
            
            # Helper to get area of triple (a, b, c)
            def area(a, b, c):
                ax, ay = a
                bx, by = b
                cx, cy = c
                return abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
            
            # Subtract old areas
            # Triples are (i, j, k) with j<k, j,k != i
            # We need to iterate j, k from 0..n-1, j!=i, k!=i, j<k
            # This is O(N^2)
            
            # To make this faster, we can precompute the list of (j, k) pairs for each i?
            # No, it depends on i.
            
            # Let's just iterate.
            # We need a temporary copy of the histogram?
            # No, we can just count the changes.
            
            # Actually, modifying a Counter in place and then reverting is risky if we reject.
            # Better: Calculate the delta in the max score?
            # The max score depends on the global histogram.
            # It's hard to know the new max without updating the whole histogram.
            
            # Alternative:
            # Just compute the full score for the candidate?
            # O(N^3) = 27000 ops.
            # 5 candidates * 27000 = 135000 ops per iteration.
            # 1000 iterations/s = 135M ops/s. Python is slow, maybe 10M ops/s.
            # So 135000 ops might take 10-20ms.
            # 50s / 20ms = 2500 iterations.
            # This is similar to the current program's speed but with better moves?
            
            # The current program does _score() which is O(N^3).
            # My incremental approach is O(N^2) per move evaluation if we update the histogram.
            # But we need to know the NEW MAX.
            # If we update the histogram, we can maintain a variable `current_max`.
            # But if we remove an area that was the max, and no other area is as high, the max drops.
            # If we add an area that becomes the new max, the max rises.
            
            # Let's try the incremental histogram update.
            
            # 1. Clone the histogram? No, too slow.
            # 2. Update in place, record changes, revert if rejected.
            
            # Let's try this.
            
            # List of changes: (area, delta_count)
            changes = []
            
            # Remove old triples involving i
            for j in range(n):
                if j == i: continue
                for k in range(j+1, n):
                    if k == i: continue
                    # Ensure i, j, k are distinct and ordered?
                    # The triple is {i, j, k}.
                    # We need to be careful not to double count or miss.
                    # Standard triple loop: i < j < k.
                    # If we fix i, the triples are:
                    # 1. i < j < k  (i is first)
                    # 2. j < i < k  (i is middle)
                    # 3. j < k < i  (i is last)
                    
                    # This is messy.
                    
                    # Easier: Iterate all pairs (j, k) from the set of other points.
                    # There are (N-1) choose 2 pairs.
                    # For each pair, calculate area with old_pt and new_pt.
                    
            # Let's restructure:
            # Get list of other points
            others = [pts[m] for m in range(n) if m != i]
            # For each pair in others, calculate area with old_pt
            # and area with new_pt.
            
            # This is O(N^2)
            
            # We'll do this inside the candidate loop.
            
            # To avoid code duplication, let's define a function to apply a move
            # and return the new score, or None if invalid.
            
            # Actually, let's just use the full O(N^3) score for simplicity first,
            # but optimize the candidate selection.
            # If full score is too slow, we'll switch to incremental.
            
            # Let's estimate:
            # N=30. C(30,3) = 4060 triples.
            # _doubled_areas does 4060 iterations.
            # Counter creation is fast.
            # Max is fast.
            # Total ~ 10-20 microseconds? No, Python loops are slow.
            # 4060 iterations * ~100ns/iter (optimistic) = 0.4ms.
            # More likely 2-5ms.
            # 5 candidates * 5ms = 25ms per outer iteration.
            # 50s / 25ms = 2000 iterations.
            # This is comparable to the current program (which does 2000 iter * 2s limit? No, it does 2000 iter with 2s limit each?
            # The current code:
            # for attempt in range(20):
            #    local_search(..., n_iter=2000, time_limit=2)
            # So it tries 20 * 2000 = 40000 moves in 40s?
            # 40000 moves / 40s = 1000 moves/s.
            # My estimate: 1000 moves/s.
            # So speed is similar.
            
            # The key is the QUALITY of the move.
            # Current: Random point swap.
            # Mine: Random point move (to any empty spot).
            # "Move to any empty spot" is a larger neighborhood than "swap with another point".
            # Swapping keeps the set of points fixed in size and location distribution?
            # No, swapping changes the positions.
            # But "move to random empty spot" allows the point to jump anywhere.
            # This is a better exploration.
            
            # Let's implement "Move to random empty spot" with full score evaluation.
            
            pass # Placeholder for logic

    return pts, best_score

def construct():
    deadline = time.time() + 55
    
    # Start with grid
    pts = _initial_grid()
    best_score = _score(pts)
    best_pts = list(pts)
    
    # Run incremental search
    # We will implement the full score version first as it's safer.
    # If it's too slow, we optimize.
    
    start_time = time.time()
    n = 30
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # To speed up score calculation, we can use numpy?
    # But points are integers.
    
    # Let's try a specific heuristic:
    # Focus on increasing the count of the CURRENT best area.
    
    while time.time() < deadline:
        # Pick a random point
        i = random.randint(0, n-1)
        old_pt = best_pts[i]
        
        # Try a few random new positions
        improved = False
        for _ in range(10):
            new_pt = random.choice(all_points)
            if new_pt == old_pt or new_pt in best_pts:
                continue
            
            # Swap
            best_pts[i] = new_pt
            new_score = _score(best_pts)
            
            if new_score > best_score:
                best_score = new_score
                improved = True
                break
            else:
                # Revert
                best_pts[i] = old_pt
        
        if not improved:
            # Occasionally do a random restart or kick
            if random.random() < 0.01:
                # Random restart from grid
                best_pts = list(_initial_grid())
                best_score = _score(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
