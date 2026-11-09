"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1, x1)  # vertical line x = x1
    if dy == 0:
        return (1, 0, -y1)  # horizontal line y = y1
    # Normalize: make gcd(dx, dy) = 1, and ensure sign convention
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx, dy = -dx, -dy
    elif dx == 0 and dy < 0:
        dy = -dy
    # Line: dx*y - dy*x = dx*y1 - dy*x1
    c = dx * y1 - dy * x1
    return (dx, dy, c)

def _find_all_lines(points):
    """Find all lines and which points lie on them."""
    lines = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                lines[key].add(i)
    # Only keep lines with >= 3 points
    return {k: v for k, v in lines.items() if len(v) >= 3}

def _validate(points):
    """Check that points are valid: 20 distinct, |x|,|y|<=100, no 4 collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check no 4 collinear
    lines = _find_all_lines(points)
    for line_points in lines.values():
        if len(line_points) >= 4:
            return False
    return True

def _count_collinear_triples(points):
    """Count all collinear triples (for analysis)."""
    lines = _find_all_lines(points)
    count = 0
    for line_points in lines.values():
        n = len(line_points)
        count += n * (n - 1) * (n - 2) // 6
    return count

def _max_3free_subset(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound with greedy lower bounds.
    """
    n = len(points)
    lines = _find_all_lines(points)
    
    # Build conflict graph: for each point, which other points share a line with it
    # and which triples are forbidden
    # A triple (i,j,k) is forbidden if i,j,k are collinear
    
    # Precompute for each pair (i,j), the set of k that form a collinear triple
    triple_conflicts = defaultdict(set)
    for line_points in lines.values():
        pts = list(line_points)
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                for c in range(b + 1, len(pts)):
                    triple_conflicts[(pts[a], pts[b])].add(pts[c])
                    triple_conflicts[(pts[b], pts[c])].add(pts[a])
                    triple_conflicts[(pts[a], pts[c])].add(pts[b])
    
    # Branch and bound
    best_size = 0
    best_set = set()
    
    def branch(selected, remaining, depth):
        nonlocal best_size, best_set
        
        # Pruning: even if we take all remaining, can we beat best?
        if len(selected) + len(remaining) <= best_size:
            return
        
        if not remaining:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
            return
        
        # Try adding a point from remaining
        # Choose point with most conflicts to prune early
        # Actually, let's try both including and excluding each point
        
        # Greedy upper bound: for each point in remaining, count how many others it conflicts with
        # Use a simple heuristic: try points in order of fewest conflicts first
        
        # For efficiency, pick the first untried point and branch
        idx = 0
        p = remaining[idx]
        
        # Option 1: include p
        new_selected = selected | {p}
        # Remove all points that would form a collinear triple with any two in new_selected
        # Actually, we need to remove points q such that there exist r in new_selected 
        # where (p, r, q) or (r1, r2, q) is collinear for r1,r2 in new_selected
        
        # Simpler: maintain the set of "forbidden" points
        # A point q is forbidden if there exist two points a, b in selected such that a,b,q collinear
        # When we add p, new forbidden points are those q where (p, a, q) collinear for some a in selected
        # OR (a, b, q) collinear for a,b in selected (already handled)
        
        # Let me restructure: track which points are still "available"
        # A point is available if it doesn't form a collinear triple with any two points in selected
        
        # This is getting complex. Let me use a different approach.
        
        pass
    
    # Let me use a cleaner branch and bound
    def solve():
        nonlocal best_size, best_set
        
        # State: selected set, and for each point, whether it's still available
        # A point q is unavailable if there exist a, b in selected such that a, b, q are collinear
        
        # Precompute: for each pair (a,b), the set of points collinear with both
        pair_to_third = {}
        for line_points in lines.values():
            pts = list(line_points)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    for k in range(len(pts)):
                        if k != i and k != j:
                            if (pts[i], pts[j]) not in pair_to_third:
                                pair_to_third[(pts[i], pts[j])] = set()
                            pair_to_third[(pts[i], pts[j])].add(pts[k])
        
        # Branch and bound
        best = 0
        best_subset = set()
        
        def bb(selected, available_mask):
            nonlocal best, best_subset
            
            # Upper bound
            if len(selected) + available_mask.sum() <= best:
                return
            
            if available_mask.sum() == 0:
                if len(selected) > best:
                    best = len(selected)
                    best_subset = set(selected)
                return
            
            # Pick a point to branch on
            # Heuristic: pick point with most conflicts (to prune)
            # Find points in available_mask
            avail = [i for i in range(n) if available_mask[i]]
            
            if not avail:
                if len(selected) > best:
                    best = len(selected)
                    best_subset = set(selected)
                return
            
            # Choose point with maximum number of conflicts among available
            # Conflict: for point p, count how many available points q would become unavailable
            # if we add p (because p and some existing selected point form a line with q)
            
            # Actually, let's just pick the first available point for simplicity, 
            # but try to be smart about ordering
            
            # For each candidate, compute how many available points it would block
            best_candidate = -1
            best_block_count = -1
            for p in avail[:10]:  # Limit to first 10 for speed
                block_count = 0
                for q in avail:
                    if q == p:
                        continue
                    # Check if adding p would make q unavailable
                    # q becomes unavailable if there exists r in selected such that p, r, q collinear
                    # OR if there exist r1, r2 in selected such that r1, r2, q collinear (already handled)
                    # The new blocking is: for some r in selected, (p, r, q) collinear
                    blocked = False
                    for r in selected:
                        key = (min(p, r), max(p, r))
                        if key in pair_to_third and q in pair_to_third[key]:
                            blocked = True
                            break
                    if blocked:
                        block_count += 1
                if block_count > best_block_count:
                    best_block_count = block_count
                    best_candidate = p
            
            if best_candidate == -1:
                best_candidate = avail[0]
            
            p = best_candidate
            
            # Option 1: include p
            new_selected = selected | {p}
            new_available = available_mask.copy()
            new_available[p] = False
            
            # Remove points that would form collinear triple with p and any point in selected
            for r in selected:
                key = (min(p, r), max(p, r))
                if key in pair_to_third:
                    for q in pair_to_third[key]:
                        new_available[q] = False
            
            # Also, for any two points r1, r2 in new_selected, remove points collinear with both
            # This is already handled incrementally if we do it right
            # Actually, we need to check all pairs in new_selected
            # But since we only added p, we only need to check pairs involving p
            # Pairs not involving p were already handled
            
            # Wait, we also need to handle: for r1, r2 in selected, points collinear with r1, r2, p
            # No, that's not right. A point q is blocked if there exist TWO points in new_selected 
            # that are collinear with q.
            
            # Let me redo: after adding p, for each pair (a, b) in new_selected, 
            # any point collinear with a and b is blocked.
            # New pairs are (p, r) for r in selected.
            # For each such pair, block points collinear with p and r.
            
            # This is what we did above. But we also need to ensure that points already 
            # blocked remain blocked. Since we're using a mask, that's fine.
            
            bb(new_selected, new_available)
            
            # Option 2: exclude p
            new_available2 = available_mask.copy()
            new_available2[p] = False
            bb(selected, new_available2)
        
        import numpy as np
        initial_mask = np.ones(n, dtype=bool)
        bb(set(), initial_mask)
        
        return best, best_subset
    
    best_size, best_set = solve()
    return best_size

def construct():
    """
    Construct 20 points with no four collinear, minimizing the largest 3-free subset.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with a known good configuration and try to improve
    # The parabola gives score 20 (no three collinear at all)
    # We want to introduce some collinear triples to reduce the max 3-free subset
    
    # Strategy: use a grid-like pattern but perturb to avoid 4 collinear
    # Try various random configurations and keep the best
    
    best_points = [[x, x * x] for x in range(-10, 10)]
    best_score = 20
    
    random.seed(42)
    
    # Try random configurations
    for trial in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Generate 20 random points
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        
        # Check validity: no 4 collinear
        if not _validate(points):
            continue
        
        # Compute score
        score = _max_3free_subset(points)
        
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            print(f"New best: score={best_score} at trial {trial}")
    
    # Try some structured configurations
    # Grid with perturbations
    for base_x in range(-10, 11, 5):
        for base_y in range(-10, 11, 5):
            if time.time() > deadline - 10:
                break
            
            points = []
            seen = set()
            # Try a 5x4 grid with small perturbations
            for i in range(5):
                for j in range(4):
                    x = base_x + i * 10 + random.randint(-2, 2)
                    y = base_y + j * 10 + random.randint(-2, 2)
                    if (x, y) not in seen and abs(x) <= 100 and abs(y) <= 100:
                        seen.add((x, y))
                        points.append([x, y])
            
            if len(points) != 20:
                continue
            
            if not _validate(points):
                continue
            
            score = _max_3free_subset(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
    
    return best_points

# EVOLVE-BLOCK-END
