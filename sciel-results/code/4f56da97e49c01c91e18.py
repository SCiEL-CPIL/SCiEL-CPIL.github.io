import time
import random
import sys
from itertools import combinations
from math import gcd

def cross(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def count_collinear_triples(points):
    """Count the number of collinear triples in the set of points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if invalid."""
    n = len(points)
    # For each pair, count points on the line
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def max_independent_set_3_uniform(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses Branch and Bound with bitmasks.
    """
    n = len(points)
    if n <= 2:
        return n

    # Precompute collinear triples as a list of (i, j, k)
    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    if not collinear_triples:
        return n

    # pair_to_third[i][j] = set of k such that {i, j, k} is collinear
    # Using a dictionary of dictionaries for speed
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j: continue
            pair_to_third[i][j] = set()
            
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)

    best_size = 0
    full_mask = (1 << n) - 1

    def solve_recur(current_size, current_mask, remaining_mask):
        nonlocal best_size
        
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Pruning: if current_size + popcount(remaining) <= best_size, prune
        if current_size + bin(remaining_mask).count('1') <= best_size:
            return

        # Pick a point from remaining
        # Heuristic: Pick the point with the most conflicts? 
        # Or just the lowest index.
        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
                
        if p == -1:
            return

        # Option 1: Exclude p
        solve_recur(current_size, current_mask, remaining_mask & ~(1 << p))

        # Option 2: Include p
        # Check if p is compatible with current_mask
        # p is incompatible if there exist r, s in current_mask such that {p, r, s} is collinear.
        # This is equivalent to: for any r in current_mask, if there exists s in current_mask 
        # such that s in pair_to_third[p][r], then incompatible.
        # Actually, if we include p, we must ensure no two points in current_mask form a triple with p.
        # And no point in current_mask forms a triple with p and another point in current_mask.
        
        # Check compatibility with current_mask:
        compatible = True
        # Iterate through points in current_mask
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = (lsb.bit_length() - 1)
            # If r and p are in a triple with some s in current_mask, it's invalid.
            # We need to check if any s in current_mask is in pair_to_third[p][r]
            # pair_to_third[p][r] is a set of indices.
            # We need to check intersection of pair_to_third[p][r] and current_mask.
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
            
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            
            # Remove forbidden points from new_remaining
            # A point q in new_remaining is forbidden if it forms a triple with p and any r in new_current_mask.
            # Actually, if q is in new_remaining, and we include p, q is forbidden if 
            # there exists r in new_current_mask such that {p, r, q} is collinear.
            # Note: r can be p? No. r is in new_current_mask.
            # So for each r in new_current_mask, q is forbidden if q in pair_to_third[p][r].
            
            forbidden_mask = 0
            # Iterate through points in new_current_mask
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = (lsb.bit_length() - 1)
                # Add all q in pair_to_third[p][r] to forbidden_mask
                # But only those in new_remaining
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
                
            new_remaining &= ~forbidden_mask
            solve_recur(current_size + 1, new_current_mask, new_remaining)

    solve_recur(0, 0, full_mask)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 135.0 # Safety margin
    
    best_triple_count = -1
    best_points = None
    best_score = 20 # Max possible score
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    def get_score(points):
        # This is the expensive part. We only call it for promising candidates.
        return max_independent_set_3_uniform(points)

    # Initial construction strategy: Grid-like to promote collinearity
    # We want many collinear triples to force the independent set size down.
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize with a grid-like structure
        # Try different grid parameters
        ox = random.randint(-80, 80)
        oy = random.randint(-80, 80)
        sx = random.randint(1, 10)
        sy = random.randint(1, 10)
        
        candidates = []
        for i in range(8):
            for j in range(8):
                x = ox + i * sx
                y = oy + j * sy
                if -100 <= x <= 100 and -100 <= y <= 100:
                    candidates.append((x, y))
                    
        # Ensure we have at least 20 unique candidates
        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)
                
        while len(unique_candidates) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)
                
        random.shuffle(unique_candidates)
        points = unique_candidates[:20]
        
        # Ensure no 4 collinear
        valid = False
        attempts = 0
        while not valid and attempts < 500:
            if not has_four_collinear(points):
                valid = True
                break
            # Find a violating point and move it
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
            attempts += 1
            
        if not valid:
            continue
        
        current_triple_count = count_collinear_triples(points)
        
        # Heuristic filter: only compute exact score if triple count is high
        # A high triple count suggests a smaller independent set
        # Let's say if triple count > 20, it's worth checking
        if current_triple_count > 15:
            current_score = get_score(points)
            if current_score < best_score:
                best_score = current_score
                best_triple_count = current_triple_count
                best_points = points.copy()
                # If we find a very good score, we can break early or adjust search
                if best_score <= 10:
                    break
        else:
            # Even if triple count is low, it might be good, but less likely.
            # Let's check a few random ones anyway in the first few restarts
            if restart < 5:
                current_score = get_score(points)
                if current_score < best_score:
                    best_score = current_score
                    best_triple_count = current_triple_count
                    best_points = points.copy()
                else:
                    # Update best triple count for reference if no score improvement
                    if current_triple_count > best_triple_count:
                        best_triple_count = current_triple_count
                        best_points = points.copy()

        # Local Search: Maximize collinear triples
        current_points = points.copy()
        current_tc = current_triple_count
        current_sc = best_score if current_points == best_points else 99
        
        # If this was our best, we already have the score
        if current_points == best_points:
            current_sc = best_score
        else:
            # Don't compute score for every local search step, only if tc improves significantly
            current_sc = 99

        for step in range(1000):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            new_p = random_point()
            
            if new_p in current_points:
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = new_p
            
            if has_four_collinear(candidate_points):
                continue
                
            new_tc = count_collinear_triples(candidate_points)
            
            if new_tc > current_tc:
                current_points = candidate_points
                current_tc = new_tc
                
                # Check score only if triple count improves a lot
                if current_tc > best_triple_count:
                    # This is a promising candidate
                    candidate_sc = get_score(current_points)
                    if candidate_sc < best_score:
                        best_score = candidate_sc
                        best_triple_count = current_tc
                        best_points = current_points.copy()
                        current_sc = candidate_sc
                    else:
                        current_sc = candidate_sc
                        # Still keep current_tc as best for this line of search
                        best_triple_count = max(best_triple_count, current_tc)
                        # But if score didn't improve, maybe we shouldn't update best_points
                        # unless we are tracking best_points based on triple count for future?
                        # No, best_points should be based on score.
                        # But we need to track the best triple count to know when to check score.
                        # Let's just update best_points if it's better in score.
                        # If score is same, maybe higher triple count is "better" for search?
                        # Let's stick to score.
                        if candidate_sc < current_sc:
                             # This case is handled above
                             pass
                        # Actually, if candidate_sc == best_score, we might want to keep the one with higher tc?
                        # Or maybe it doesn't matter. Let's just keep the first one found with that score.
                
                # If we haven't checked score recently, maybe check now if tc is high
                elif current_tc > 30 and step % 10 == 0:
                    candidate_sc = get_score(current_points)
                    if candidate_sc < best_score:
                        best_score = candidate_sc
                        best_triple_count = current_tc
                        best_points = current_points.copy()
                    current_sc = candidate_sc

    if best_points is None:
        # Fallback: Return any valid set
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        # Ensure no 4 collinear
        for _ in range(200):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        best_points = points

    # Convert tuples to lists for output
    return [list(p) for p in best_points]
