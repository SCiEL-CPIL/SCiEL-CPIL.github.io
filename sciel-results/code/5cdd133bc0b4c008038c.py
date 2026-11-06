"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 64
TIME_LIMIT = 280.0

# EVOLVE-BLOCK-START

def _validate_no_isosceles(points):
    """Verify no three points form an isosceles triangle."""
    n = len(points)
    if n < 3:
        return True
    pts = np.array(points, dtype=np.int32)
    # Compute all pairwise squared distances
    for i in range(n):
        d_from_i = (pts[i] - pts[i+1:])**2
        d2_from_i = d_from_i.sum(axis=1)
        # Check for duplicates among distances from point i
        if len(d2_from_i) != len(set(d2_from_i.tolist())):
            return False
        # Check: for each j, dist(i,j) should not equal dist(j,k) for any k
        for j in range(i+1, n):
            dij2 = d2_from_i[j-1]
            # dist(j, k) for k > j
            d_from_j = (pts[j] - pts[j+1:])**2
            d2_from_j = d_from_j.sum(axis=1)
            if dij2 in d2_from_j.tolist():
                return False
    return True


class NoIsoscelesState:
    """
    Efficient state tracker for the no-isosceles constraint.
    
    Key insight: A point p can be added iff:
    1. All distances from p to existing points are distinct (no two existing points
       are equidistant from p).
    2. For each existing point q, dist(p,q) is not already in the set of distances
       from q to other existing points.
    
    We maintain:
    - points: list of selected points
    - dist_sets: dict mapping point -> set of squared distances to other selected points
    - For fast checking, we also maintain a global "forbidden" structure.
    """
    
    def __init__(self):
        self.points = []
        self.point_set = set()
        # dist_sets[pt] = set of squared distances from pt to all other selected points
        self.dist_sets = {}
        # For check 1: we need to know, for a candidate point p, whether any two
        # existing points are at the same distance from p.
        # This is hard to precompute for all candidates, so we check on the fly.
        # For check 2: dist(p, q) must not be in dist_sets[q]
    
    def can_add(self, pt):
        """Check if pt can be added without creating an isosceles triangle."""
        if pt in self.point_set:
            return False
        if len(self.points) < 2:
            return True
        
        nx, ny = pt
        seen_dists = set()
        
        for (px, py) in self.point_set:
            dx = nx - px
            dy = ny - py
            d2 = dx*dx + dy*dy
            
            # Check 1: no two existing points at same distance from pt
            if d2 in seen_dists:
                return False
            seen_dists.add(d2)
            
            # Check 2: dist(pt, existing) not in existing's distance set
            if d2 in self.dist_sets.get((px, py), ()):
                return False
        
        return True
    
    def add(self, pt):
        """Add a point to the state (assumes can_add returned True)."""
        self.point_set.add(pt)
        self.points.append(pt)
        self.dist_sets[pt] = set()
        
        for (px, py) in self.point_set:
            if pt == (px, py):
                continue
            dx = pt[0] - px
            dy = pt[1] - py
            d2 = dx*dx + dy*dy
            self.dist_sets[pt].add(d2)
            self.dist_sets[(px, py)].add(d2)
    
    def remove(self, pt):
        """Remove a point from the state."""
        self.point_set.remove(pt)
        self.points.remove(pt)
        
        for (px, py) in self.point_set:
            dx = pt[0] - px
            dy = pt[1] - py
            d2 = dx*dx + dy*dy
            if d2 in self.dist_sets.get((px, py), ()):
                self.dist_sets[(px, py)].discard(d2)
        
        del self.dist_sets[pt]
    
    def rebuild(self, points):
        """Rebuild state from a list of points."""
        self.points = list(points)
        self.point_set = set(points)
        self.dist_sets = {pt: set() for pt in self.point_set}
        
        for i in range(len(self.points)):
            for j in range(i+1, len(self.points)):
                dx = self.points[i][0] - self.points[j][0]
                dy = self.points[i][1] - self.points[j][1]
                d2 = dx*dx + dy*dy
                self.dist_sets[self.points[i]].add(d2)
                self.dist_sets[self.points[j]].add(d2)
    
    def size(self):
        return len(self.points)


def _vectorized_greedy(seed, deadline=None, strategy='random'):
    """
    Vectorized greedy: use numpy to quickly compute distances from all candidates
    to all selected points, then filter.
    """
    rng = random.Random(seed)
    state = NoIsoscelesState()
    
    # Precompute all grid points
    all_x = np.arange(N, dtype=np.int32)
    all_y = np.arange(N, dtype=np.int32)
    gx, gy = np.meshgrid(all_x, all_y, indexing='ij')
    all_points = list(zip(gx.flatten().tolist(), gy.flatten().tolist()))
    
    if strategy == 'random':
        order = list(range(N*N))
        rng.shuffle(order)
    elif strategy == 'spiral':
        cx, cy = N//2, N//2
        dists = (gx - cx)**2 + (gy - cy)**2
        order = np.argsort(dists.flatten(), kind='stable').tolist()
    elif strategy == 'diagonal':
        # Order by x+y then x
        order = sorted(range(N*N), key=lambda i: (gx.flatten()[i] + gy.flatten()[i], gx.flatten()[i]))
    else:
        order = list(range(N*N))
        rng.shuffle(order)
    
    for idx in order:
        if deadline is not None and time.time() > deadline:
            break
        
        pt = all_points[idx]
        if state.can_add(pt):
            state.add(pt)
    
    return state.points


def _conflict_aware_greedy(seed, deadline=None):
    """
    Greedy that picks the point with minimum "conflict potential" at each step.
    Conflict potential = number of distance conflicts if added.
    """
    rng = random.Random(seed)
    state = NoIsoscelesState()
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    remaining = set(all_points)
    
    while remaining:
        if deadline is not None and time.time() > deadline:
            break
        
        if state.size() < 2:
            pt = rng.choice(list(remaining))
        else:
            # Sample candidates and pick the one with fewest conflicts
            if len(remaining) > 300:
                sample = rng.sample(list(remaining), 300)
            else:
                sample = list(remaining)
            
            best_pt = None
            best_conflicts = float('inf')
            
            for pt in sample:
                conflicts = 0
                seen_dists = set()
                for (px, py) in state.point_set:
                    dx = pt[0] - px
                    dy = pt[1] - py
                    d2 = dx*dx + dy*dy
                    if d2 in seen_dists:
                        conflicts += 1
                        break
                    seen_dists.add(d2)
                    if d2 in state.dist_sets.get((px, py), ()):
                        conflicts += 1
                        break
                
                if conflicts == 0:
                    best_pt = pt
                    break
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_pt = pt
            
            if best_pt is None:
                break
            pt = best_pt
        
        if state.can_add(pt):
            state.add(pt)
            remaining.discard(pt)
    
    return state.points


def _local_search_swap(state_points, seed, max_iters=30, deadline=None):
    """
    Local search: repeatedly try to remove a point and add a different one.
    Accept if size stays the same or increases.
    """
    rng = random.Random(seed)
    state = NoIsoscelesState()
    state.rebuild(state_points)
    current_size = state.size()
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if state.size() < 2:
            break
        
        # Pick a random point to try removing
        idx = rng.randrange(state.size())
        removed = state.points[idx]
        state.remove(removed)
        
        # Try to find a replacement
        found = False
        candidates = [p for p in all_points if p not in state.point_set]
        rng.shuffle(candidates)
        
        for pt in candidates[:800]:
            if state.can_add(pt):
                state.add(pt)
                found = True
                break
        
        if not found:
            # Restore removed point
            state.add(removed)
    
    return state.points


def _multi_start_local_search(initial_points, seed, num_starts=5, deadline=None):
    """
    Try multiple local search restarts from perturbed versions of the initial set.
    """
    rng = random.Random(seed)
    best = list(initial_points)
    best_size = len(best)
    
    for s in range(num_starts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb: remove k random points
        k = rng.randint(1, max(1, len(best) // 10))
        perturbed = list(best)
        remove_idx = rng.sample(range(len(perturbed)), min(k, len(perturbed)))
        for i in sorted(remove_idx, reverse=True):
            perturbed.pop(i)
        
        # Try to fill back up with greedy
        state = NoIsoscelesState()
        state.rebuild(perturbed)
        
        all_points = [(x, y) for x in range(N) for y in range(N)]
        candidates = [p for p in all_points if p not in state.point_set]
        rng.shuffle(candidates)
        
        for pt in candidates:
            if deadline is not None and time.time() > deadline:
                break
            if state.can_add(pt):
                state.add(pt)
        
        # Local search
        result = _local_search_swap(state.points, seed + s * 100, max_iters=15, deadline=deadline)
        
        if len(result) > best_size:
            best_size = len(result)
            best = list(result)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 20
    
    best_points = []
    best_size = 0
    
    # Phase 1: Multiple greedy strategies
    strategies = ['random', 'spiral', 'diagonal']
    for strat in strategies:
        if time.time() > deadline:
            break
        pts = _vectorized_greedy(42, deadline, strategy=strat)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Phase 2: Conflict-aware greedy with different seeds
    for seed in range(8):
        if time.time() > deadline:
            break
        pts = _conflict_aware_greedy(seed, deadline)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = list(pts)
    
    # Phase 3: Local search from best
    if best_size > 0 and time.time() < deadline:
        improved = _local_search_swap(best_points, 0, max_iters=40, deadline=deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_points = list(improved)
    
    # Phase 4: Multi-start perturbation + local search
    if best_size > 0 and time.time() < deadline:
        improved = _multi_start_local_search(best_points, 100, num_starts=10, deadline=deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_points = list(improved)
    
    # Phase 5: More local search iterations if time remains
    while time.time() < deadline - 5:
        if best_size > 0:
            improved = _local_search_swap(best_points, int(time.time()*1000) % 10000, max_iters=25, deadline=deadline)
            if len(improved) > best_size:
                best_size = len(improved)
                best_points = list(improved)
            else:
                break
    
    return best_points

# EVOLVE-BLOCK-END
