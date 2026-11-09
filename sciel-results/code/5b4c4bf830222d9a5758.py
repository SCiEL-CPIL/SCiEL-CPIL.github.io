"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

class NoIsoscelesSet:
    """Data structure for maintaining a no-isosceles set with fast validity checks."""
    
    def __init__(self, n):
        self.n = n
        self.points = []
        self.point_set = set()
        # For each point index, store the set of squared distances to all other points
        self.dist_sets = []
        # For fast lookup: mapping from (point_index, dist) -> True
        # Actually, we need: for each point i, what distances does it have?
        # dist_sets[i] = set of squared distances from point i to all other points
    
    def _can_add(self, pos):
        """Check if pos can be added without creating an isosceles triangle."""
        # Check 1: pos as apex - all distances from pos to existing points must be distinct
        dists_from_new = set()
        dists_list = []
        for p in self.points:
            d2 = _sq_dist(pos, p)
            if d2 == 0:
                return False  # Same position
            if d2 in dists_from_new:
                return False
            dists_from_new.add(d2)
            dists_list.append(d2)
        
        # Check 2: each existing point as apex - the new distance must not already exist
        for i, d2 in enumerate(dists_list):
            if d2 in self.dist_sets[i]:
                return False
        
        return True
    
    def _add(self, pos):
        """Add a point to the set (assumes validity already checked)."""
        idx = len(self.points)
        self.points.append(pos)
        self.point_set.add(pos)
        
        # Compute distances from new point to all existing
        dists_from_new = set()
        for p in self.points[:-1]:
            d2 = _sq_dist(pos, p)
            dists_from_new.add(d2)
        
        # Update existing points' distance sets
        for i, p in enumerate(self.points[:-1]):
            d2 = _sq_dist(pos, p)
            self.dist_sets[i].add(d2)
        
        self.dist_sets.append(dists_from_new)
    
    def _remove(self, pos):
        """Remove a point from the set."""
        if pos not in self.point_set:
            return
        idx = self.points.index(pos)
        
        # Remove from dist_sets: for each other point, remove the distance to pos
        for i, p in enumerate(self.points):
            if i != idx:
                d2 = _sq_dist(p, pos)
                self.dist_sets[i].discard(d2)
        
        # Remove the point and its dist_set
        self.points.pop(idx)
        self.dist_sets.pop(idx)
        self.point_set.discard(pos)
    
    def size(self):
        return len(self.points)

def _greedy_random(n, timeout, seed=None):
    """Random greedy construction."""
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    best = NoIsoscelesSet(n)
    best_size = 0
    
    all_positions = [(x, y) for x in range(n) for y in range(n)]
    
    while time.time() - start_time < timeout:
        current = NoIsoscelesSet(n)
        
        shuffled = all_positions[:]
        random.shuffle(shuffled)
        
        for pos in shuffled:
            if time.time() - start_time > timeout:
                break
            if current._can_add(pos):
                current._add(pos)
        
        if current.size() > best_size:
            best_size = current.size()
            best = NoIsoscelesSet(n)
            best.points = current.points[:]
            best.point_set = current.point_set.copy()
            best.dist_sets = [s.copy() for s in current.dist_sets]
    
    return best.points

def _greedy_structured(n, timeout):
    """Greedy with structured ordering for better initial placement."""
    start_time = time.time()
    best = NoIsoscelesSet(n)
    best_size = 0
    
    all_positions = [(x, y) for x in range(n) for y in range(n)]
    
    while time.time() - start_time < timeout:
        current = NoIsoscelesSet(n)
        
        # Try different ordering strategies
        strategy = random.random()
        if strategy < 0.25:
            # Random
            shuffled = all_positions[:]
            random.shuffle(shuffled)
        elif strategy < 0.5:
            # Spiral from center
            cx, cy = n // 2, n // 2
            shuffled = sorted(all_positions, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            # Add some randomness
            for i in range(len(shuffled)-1, 0, -1):
                j = random.randint(0, i)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        elif strategy < 0.75:
            # Row-major with random offsets
            shuffled = []
            row_offset = random.randint(0, n-1)
            for i in range(n):
                y = (row_offset + i) % n
                row = [(x, y) for x in range(n)]
                if random.random() < 0.5:
                    row.reverse()
                shuffled.extend(row)
        else:
            # Corner-biased
            shuffled = all_positions[:]
            random.shuffle(shuffled)
            corners = [(0,0), (n-1,0), (0,n-1), (n-1,n-1)]
            corner_pts = [c for c in shuffled if c in corners]
            other_pts = [p for p in shuffled if p not in corners]
            shuffled = corner_pts + other_pts
        
        for pos in shuffled:
            if time.time() - start_time > timeout:
                break
            if current._can_add(pos):
                current._add(pos)
        
        if current.size() > best_size:
            best_size = current.size()
            best = NoIsoscelesSet(n)
            best.points = current.points[:]
            best.point_set = current.point_set.copy()
            best.dist_sets = [s.copy() for s in current.dist_sets]
    
    return best.points

def _local_search(points, n, timeout):
    """Local search: try to add more points by scanning all candidates."""
    start_time = time.time()
    current = NoIsoscelesSet(n)
    current.points = list(points)
    current.point_set = set(points)
    
    # Rebuild dist_sets
    for i in range(len(current.points)):
        dists = set()
        for j in range(len(current.points)):
            if i != j:
                dists.add(_sq_dist(current.points[i], current.points[j]))
        current.dist_sets.append(dists)
    
    all_positions = [(x, y) for x in range(n) for y in range(n)]
    candidates = [p for p in all_positions if p not in current.point_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        for pos in candidates[:]:
            if time.time() - start_time > timeout:
                break
            if current._can_add(pos):
                current._add(pos)
                candidates.remove(pos)
                improved = True
    
    return current.points

def _simulated_annealing(points, n, timeout):
    """Simulated annealing: try removing and re-adding points to escape local optima."""
    start_time = time.time()
    current = NoIsoscelesSet(n)
    current.points = list(points)
    current.point_set = set(points)
    
    # Rebuild dist_sets
    for i in range(len(current.points)):
        dists = set()
        for j in range(len(current.points)):
            if i != j:
                dists.add(_sq_dist(current.points[i], current.points[j]))
        current.dist_sets.append(dists)
    
    best_points = current.points[:]
    best_size = current.size()
    
    temperature = 10.0
    cooling = 0.995
    
    while time.time() - start_time < timeout and temperature > 0.1:
        # Try a random perturbation: remove k random points, then try to re-add
        k = random.randint(1, min(3, max(1, current.size() // 5)))
        removed = random.sample(current.points, k)
        
        for p in removed:
            current._remove(p)
        
        # Try to re-add all removed points plus some new ones
        all_positions = [(x, y) for x in range(n) for y in range(n)]
        candidates = [p for p in all_positions if p not in current.point_set]
        # Prioritize the removed points
        removed_set = set(removed)
        candidates = [p for p in candidates if p in removed_set] + [p for p in candidates if p not in removed_set]
        random.shuffle(candidates[3:])  # Keep removed points somewhat prioritized
        
        for pos in candidates:
            if time.time() - start_time > timeout:
                break
            if current._can_add(pos):
                current._add(pos)
        
        if current.size() > best_size:
            best_size = current.size()
            best_points = current.points[:]
        
        # Accept or reject based on simulated annealing
        if current.size() >= best_size:
            pass  # Keep the new state
        else:
            # With probability exp(-(best_size - current.size()) / temperature), keep
            if random.random() < math.exp(-(best_size - current.size()) / temperature):
                pass  # Keep
            else:
                # Revert: this is complex, so just continue with current state
                pass
        
        temperature *= cooling
    
    # Final local search
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 1:
        best_points = _local_search(best_points, n, remaining_time)
    
    return best_points

def _sidon_line(n):
    """Construct points on a line using Sidon-like positions."""
    xs, used = [], set()
    x = 0
    while x < n:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def _two_row_construction(n):
    """Try to construct points on two rows with careful spacing."""
    points = []
    
    # Place some points on row 0 using Sidon
    xs_row0, used_diffs = [], set()
    for x in range(n):
        diffs = {abs(x - y) for y in xs_row0}
        if not (diffs & used_diffs):
            used_diffs |= diffs
            xs_row0.append(x)
    
    for x in xs_row0:
        points.append((x, 0))
    
    # Try to add points on row 1
    forbidden_x = set()
    for i in range(len(xs_row0)):
        for j in range(i+1, len(xs_row0)):
            s = xs_row0[i] + xs_row0[j]
            if s % 2 == 0:
                forbidden_x.add(s // 2)
    
    xs_row1 = []
    for x in range(n):
        if x in forbidden_x:
            continue
        valid = True
        for y in xs_row1:
            for a in xs_row0:
                d1 = _sq_dist((x,1), (y,1))
                d2 = _sq_dist((x,1), (a,0))
                d3 = _sq_dist((y,1), (a,0))
                if d1 == d2 or d1 == d3 or d2 == d3:
                    if d1 != 0 and d2 != 0 and d3 != 0:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            xs_row1.append(x)
            for a in xs_row0:
                s = x + a
                if s % 2 == 0:
                    forbidden_x.add(s // 2)
    
    for x in xs_row1:
        points.append((x, 1))
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave safety margin before 300s
    
    n = N
    
    # Phase 1: Deterministic constructions
    candidates = []
    sidon_pts = _sidon_line(n)
    candidates.append(sidon_pts)
    two_row_pts = _two_row_construction(n)
    candidates.append(two_row_pts)
    
    best_points = max(candidates, key=len)
    
    # Phase 2: Random greedy (multiple attempts)
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        greedy_time = min(60, time_remaining * 0.4)
        greedy_pts = _greedy_random(n, greedy_time)
        if len(greedy_pts) > len(best_points):
            best_points = greedy_pts
    
    # Phase 3: Structured greedy
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        structured_time = min(60, time_remaining * 0.3)
        structured_pts = _greedy_structured(n, structured_time)
        if len(structured_pts) > len(best_points):
            best_points = structured_pts
    
    # Phase 4: Local search polish
    time_remaining = deadline - time.time()
    if time_remaining > 3:
        polish_time = min(30, time_remaining * 0.2)
        polished = _local_search(best_points, n, polish_time)
        if len(polished) > len(best_points):
            best_points = polished
    
    # Phase 5: Simulated annealing / perturbation
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        sa_time = min(80, time_remaining * 0.6)
        sa_pts = _simulated_annealing(best_points, n, sa_time)
        if len(sa_pts) > len(best_points):
            best_points = sa_pts
    
    # Final local search with remaining time
    time_remaining = deadline - time.time()
    if time_remaining > 2:
        final_polish = _local_search(best_points, n, time_remaining - 1)
        if len(final_polish) > len(best_points):
            best_points = final_polish
    
    return best_points

# EVOLVE-BLOCK-END
