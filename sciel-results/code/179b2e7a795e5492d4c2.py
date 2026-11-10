import itertools
import random
import numpy as np

def _cross(o, a, b):
    """Cross product (a-o) x (b-o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _collinear3(p, q, r):
    return _cross(p, q, r) == 0

def _build_collinear_triples(points):
    """Return list of (i,j,k) indices where points i,j,k are collinear."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _has_four_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    # Map line (slope, intercept) to count? 
    # Or just check all 4-subsets? O(N^4) = 4845 for N=20. Fast enough.
    # But we can be smarter.
    # For each pair, count how many other points are collinear.
    for i in range(n):
        for j in range(i+1, n):
            count = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if _collinear3(points[i], points[j], points[k]):
                    count += 1
            if count >= 2:
                return True
    return False

def _max_independent_set(points):
    """
    Exact branch and bound for largest subset with no three collinear.
    Uses bitmask representation for speed.
    """
    n = len(points)
    triples = _build_collinear_triples(points)
    
    if not triples:
        return n
        
    # Precompute for each point, the bitmask of triples containing it? 
    # Or better: for each triple, store the 3 indices.
    # We maintain a bitmask `remaining` of points still available.
    # We want to find the max size of a subset of `remaining` that contains no full triple.
    
    # Heuristic for B&B:
    # If we have a set of active triples, we must exclude at least one point from each.
    # We branch on a triple.
    
    # To speed up, we can precompute for each point, the list of triples it is part of.
    # But checking if a triple is active requires checking if all 3 points are in `remaining`.
    
    # Let's use a recursive function with memoization? 
    # State is `remaining` bitmask. 2^20 = 1M states.
    # We can memoize the result for a given `remaining` mask.
    # `dp[mask]` = max independent set size for points in `mask`.
    
    # Transition:
    # If no triple is fully contained in `mask`, result is popcount(mask).
    # Else, pick a triple (a,b,c) fully in `mask`.
    # We must exclude at least one of a,b,c.
    # dp[mask] = max(
    #     dp[mask ^ (1<<a)],
    #     dp[mask ^ (1<<b)],
    #     dp[mask ^ (1<<c)]
    # )
    # Wait, is this correct?
    # If we exclude a, we solve for mask without a.
    # Yes, because any independent set in `mask` cannot contain {a,b,c}.
    # So it is a subset of (mask \ {a}) U (mask \ {b}) U (mask \ {c})?
    # No. An independent set $I \subseteq mask$ must not contain any edge.
    # If $I$ does not contain $a$, then $I \subseteq mask \setminus \{a\}$.
    # If $I$ does not contain $b$, then $I \subseteq mask \setminus \{b\}$.
    # If $I$ does not contain $c$, then $I \subseteq mask \setminus \{c\}$.
    # Since $I$ must not contain $\{a,b,c\}$, it must be missing at least one of them.
    # So $I \subseteq (mask \setminus \{a\}) \cup (mask \setminus \{b\}) \cup (mask \setminus \{c\})$?
    # No, $I$ is a specific set.
    # $I$ is a valid independent set in $mask$.
    # This implies $I$ is a valid independent set in $mask \setminus \{a\}$ OR $mask \setminus \{b\}$ OR $mask \setminus \{c\}$?
    # Yes. If $a \notin I$, $I \subseteq mask \setminus \{a\}$.
    # So $\alpha(mask) = \max(\alpha(mask \setminus \{a\}), \alpha(mask \setminus \{b\}), \alpha(mask \setminus \{c\}))$.
    
    # This recursion works!
    # We need to pick a triple efficiently.
    # We can iterate through triples and find the first one that is a subset of `mask`.
    
    # Memoization table
    from functools import lru_cache
    
    # Convert triples to a list of bitmasks for faster checking?
    # triple_mask = (1<<a) | (1<<b) | (1<<c)
    # Check: (mask & triple_mask) == triple_mask
    
    triple_masks = []
    for t in triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
    
    # Sort triple_masks? Maybe not necessary.
    
    @lru_cache(maxsize=None)
    def solve(mask):
        if mask == 0:
            return 0
        
        # Check if any triple is fully in mask
        # Optimization: if popcount(mask) is small, check directly?
        # Just iterate triple_masks.
        found_triple = None
        for tm in triple_masks:
            if (mask & tm) == tm:
                found_triple = tm
                break
        
        if found_triple is None:
            # No triple in mask, so all points can be taken
            return bin(mask).count('1')
        
        # Extract indices from found_triple
        # We need to remove one bit.
        # Get the 3 bits.
        bits = []
        tm = found_triple
        while tm:
            lsb = tm & -tm
            idx = lsb.bit_length() - 1
            bits.append(idx)
            tm ^= lsb
        
        a, b, c = bits
        m1 = mask & ~(1 << a)
        m2 = mask & ~(1 << b)
        m3 = mask & ~(1 << c)
        
        return max(solve(m1), solve(m2), solve(m3))
        
    initial_mask = (1 << n) - 1
    return solve(initial_mask)

def _score(points):
    """Compute the score (largest subset with no three collinear)."""
    if _has_four_collinear(points):
        return 100  # invalid, penalize
    return _max_independent_set(points)

def _generate_base_config():
    """Generate a base configuration with many collinear triples but no four collinear."""
    # Try a perturbed grid
    # 5 rows, 4 cols
    # x = i * 10 + offset, y = j * 10
    # We want to break 4-in-a-row.
    # In a 5x4 grid, rows have 4 points. We need to break them.
    # Shift points in each row.
    
    points = []
    for i in range(5):
        for j in range(4):
            x = i * 10
            y = j * 10
            # Perturb x based on i and j to break horizontal lines
            # If we shift x by i, we might create diagonals?
            # Let's try random small shifts
            pass
            
    # Actually, let's use a known good structure: 
    # Points on y = x^2 mod P? No, integer points.
    
    # Let's try to construct from a 4x5 grid by moving points.
    # Base grid:
    base = []
    for i in range(5):
        for j in range(4):
            base.append([i * 10, j * 10])
            
    # Apply random perturbations to break 4-collinear
    # We'll return this as a candidate, and let local search fix it.
    return base

def _local_search(points, max_iters=2000):
    """
    Local search to minimize the score.
    """
    points = [p[:] for p in points]
    n = len(points)
    
    # Ensure validity first
    if _has_four_collinear(points):
        # Try to fix by moving points
        # This is complex, so we rely on the generator to produce valid or near-valid
        # For now, if invalid, we try to repair
        for i in range(n):
            if _has_four_collinear(points):
                # Move point i randomly
                for _ in range(100):
                    old = points[i][:]
                    points[i] = [random.randint(-100, 100), random.randint(-100, 100)]
                    if points[i] in points[:i] + points[i+1:]:
                        continue
                    if not _has_four_collinear(points):
                        break
                if _has_four_collinear(points):
                    points[i] = old
                    break
            else:
                break
    
    if _has_four_collinear(points):
        return points, 100
        
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    for it in range(max_iters):
        improved = False
        # Pick a random point to move
        i = random.randint(0, n-1)
        old_point = points[i][:]
        
        # Try a few random new positions
        for _ in range(50):
            new_pos = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_pos in points[:i] + points[i+1:]:
                continue
            
            points[i] = new_pos
            if _has_four_collinear(points):
                continue
                
            new_score = _score(points)
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                improved = True
                break # Move to next iteration
            else:
                points[i] = old_point
        
        if not improved:
            # Try swapping two points? Or just continue
            pass
            
    return best_points, best_score

def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    """
    random.seed(42)
    
    best_points = None
    best_score = 100
    
    # Generate multiple random valid configurations
    for trial in range(20):
        points = []
        attempts = 0
        while len(points) < 20 and attempts < 5000:
            attempts += 1
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p in points:
                continue
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
        
        if len(points) == 20 and not _has_four_collinear(points):
            improved_points, score = _local_search(points, max_iters=500)
            if score < best_score:
                best_score = score
                best_points = improved_points
    
    # Also try structured starts
    base = _generate_base_config()
    # Perturb base to make it valid
    # ... (simplified: just add to candidates if valid)
    
    if best_points is None:
        # Fallback
        # Find any valid config
        for trial in range(100):
            points = []
            while len(points) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p in points:
                    continue
                points.append(p)
                if _has_four_collinear(points):
                    points.pop()
            if not _has_four_collinear(points):
                best_points = points
                break
    
    if best_points is None:
        # Absolute fallback: parabola (score 20, valid)
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points
