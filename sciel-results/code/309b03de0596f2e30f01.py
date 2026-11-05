import time
import random
import math
from collections import defaultdict

DEADLINE = 115.0

def collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def has_four_collinear(points):
    n = len(points)
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope = (0, 1)
            elif dy == 0:
                slope = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx, dy = dx // g, dy // g
                if dx < 0:
                    dx, dy = -dx, -dy
                slope = (dx, dy)
            if slope in slopes:
                slopes[slope] += 1
                if slopes[slope] >= 3:
                    return True
            else:
                slopes[slope] = 1
    return False

def max_no_three_collinear(points):
    """
    Exact branch and bound for largest 3-collinear-free subset.
    Uses bitmask for efficiency.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute collinearity
    # collinear[i][j] is a bitmask of k such that (i, j, k) are collinear
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask
            collinear_mask[j][i] = mask

    best = [0]
    
    def search(available, current_size):
        # Pruning
        if current_size + bin(available).count('1') <= best[0]:
            return
        
        if available == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick a point v
        # Heuristic: pick the point with the most conflicts to prune faster
        # Or just the lowest set bit for simplicity and speed
        v_bit = available & -available
        v = (v_bit.bit_length() - 1)
        
        # Option 1: Exclude v
        search(available ^ v_bit, current_size)
        
        # Option 2: Include v
        # Find all points u such that there exists w in available where (v, u, w) is collinear
        # If we include v, we cannot include any such u (because w would be excluded? No, w is also in available)
        # Actually, if we include v, we cannot include ANY pair (u, w) that forms a triple with v.
        # This means we must remove u from available if there is ANY w in available such that (v,u,w) is collinear.
        # Wait, if we remove u, we might break the triple.
        # The condition for an independent set is: no three points are collinear.
        # If we select v, we cannot select both u and w if (v,u,w) is a triple.
        # This is a 3-uniform constraint.
        # Standard B&B for 3-uniform:
        # If we select v, we must ensure no two other selected points form a triple with v.
        # This is hard to prune simply by removing points.
        
        # Alternative approach:
        # The "conflicts" for v are not just single points.
        # However, we can use a simpler bound.
        # If we select v, the remaining problem is to find an independent set in the remaining points,
        # with the constraint that we cannot select any pair (u,w) that is collinear with v.
        
        # Let's use a different B&B strategy:
        # Recursive selection of points.
        # State: `available` (bitmask), `current_set` (bitmask of selected points).
        # If we add v to current_set, we must remove from `available` any point u such that
        # there exists w in current_set such that (v,u,w) is collinear? No, w must be in the FINAL set.
        
        # This is getting complicated. Let's use the standard "Max Independent Set in Hypergraph" B&B.
        # Actually, for n=20, a simpler recursive search with memoization or just brute force with pruning might be enough.
        # But the previous B&B in the seed program was:
        # search(idx, current_set, current_pairs)
        # It checked if adding idx creates a pair that is already in current_pairs.
        # This is correct for 3-uniform.
        # Let's stick to that logic but optimize with bitmasks if possible, or just use the set-based one which is clear.
        
        pass # Placeholder for the actual implementation below

    # Re-implementing the seed's B&B which was correct, but maybe slow.
    # Let's try to speed it up.
    
    coll_pairs = [set() for _ in range(n)]
    for k in range(n):
        for i in range(n):
            if i == k: continue
            for j in range(i + 1, n):
                if j == k: continue
                if collinear(points[i], points[j], points[k]):
                    coll_pairs[k].add((i, j) if i < j else (j, i))
    
    # Use a bitmask for current_pairs? No, pairs are up to 190. Bitmask of pairs is 190 bits.
    # Python ints are arbitrary precision, so we can use an int as a bitmask for pairs.
    # Index of pair (i,j) where i<j: i*n - i*(i+1)//2 + (j-i-1)?
    # Simpler: map (i,j) to a unique int.
    pair_to_idx = {}
    idx_to_pair = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_to_idx[(i,j)] = idx
            idx_to_pair[idx] = (i,j)
            idx += 1
    num_pairs = idx
    
    # coll_pairs_mask[k] is a bitmask of pairs involving k? No, pairs that are "forbidden" if k is added?
    # If we add k, we add all pairs (k, j) for j in current_set.
    # And we check if any of these pairs are already in the set of pairs from current_set?
    # No, the constraint is: the set of selected points must not contain any triple.
    # This is equivalent to: the set of pairs formed by selected points must not contain any pair (i,j) such that there is a k in selected points with (i,j,k) collinear.
    # The seed program's logic:
    # `current_pairs` stores all pairs (i,j) such that i,j are in current_set.
    # When adding k, we check if any pair in `coll_pairs[k]` is in `current_pairs`.
    # `coll_pairs[k]` contains pairs (i,j) such that (i,j,k) is collinear.
    # If (i,j) is in `current_pairs`, it means i and j are already selected.
    # So adding k would complete the triple (i,j,k).
    # This logic is correct.
    
    # Optimization: Represent `current_pairs` as an integer bitmask.
    # Represent `coll_pairs[k]` as an integer bitmask.
    
    coll_pairs_mask = [0] * n
    for k in range(n):
        mask = 0
        for (i, j) in coll_pairs[k]:
            p_idx = pair_to_idx.get((i,j)) or pair_to_idx.get((j,i))
            if p_idx is not None:
                mask |= (1 << p_idx)
        coll_pairs_mask[k] = mask
    
    best_val = [0]
    
    def search(idx, current_pairs_mask):
        # Pruning: max possible size
        # We don't track current_set size explicitly in the signature, 
        # but we can infer it? No.
        # Let's pass current_size.
        pass

    # Let's just use the set-based one from the seed, it might be fast enough for n=20 if we don't call it too often.
    # The seed called it every step of SA.
    # If SA takes 1ms per step, 100s allows 100k steps.
    # Set-based B&B for n=20 is likely < 1ms.
    
    def search_set(idx, current_set, current_pairs):
        if len(current_set) + (n - idx) <= best[0]:
            return
        if idx == n:
            if len(current_set) > best[0]:
                best[0] = len(current_set)
            return
        
        # Exclude
        search_set(idx + 1, current_set, current_pairs)
        
        # Include
        can_include = True
        for pair in coll_pairs[idx]:
            if pair in current_pairs:
                can_include = False
                break
        
        if can_include:
            new_set = current_set | {idx}
            new_pairs = set(current_pairs)
            for j in current_set:
                new_pairs.add((idx, j) if idx < j else (j, idx))
            search_set(idx + 1, new_set, new_pairs)

    search_set(0, set(), set())
    return best[0]

def construct():
    start_time = time.time()
    random.seed(42)
    
    def generate_points_greedy():
        points = []
        used = set()
        
        # Define some candidate lines
        # We want lines with many integer points.
        # Horizontal, Vertical, Slope 1, -1, 2, -2, etc.
        
        lines = []
        for y in range(-50, 51, 10):
            lines.append(('h', y))
        for x in range(-50, 51, 10):
            lines.append(('v', x))
        for m in [1, -1, 2, -2, 3, -3]:
            for b in range(-50, 51, 10):
                lines.append(('s', m, b))
        
        random.shuffle(lines)
        
        for line in lines:
            if len(points) >= 20:
                break
            if line[0] == 'h':
                y = line[1]
                candidates = [(x, y) for x in range(-100, 101) if (x, y) not in used]
            elif line[0] == 'v':
                x = line[1]
                candidates = [(x, y) for y in range(-100, 101) if (x, y) not in used]
            else:
                m, b = line[1], line[2]
                candidates = []
                for x in range(-100, 101):
                    y = m * x + b
                    if abs(y) <= 100 and (x, y) not in used:
                        candidates.append((x, y))
            
            random.shuffle(candidates)
            
            added = 0
            for p in candidates:
                if len(points) >= 20:
                    break
                # Check 4-collinear
                new_pts = points + [p]
                if has_four_collinear(new_pts):
                    continue
                points.append(p)
                used.add(p)
                added += 1
                if added >= 3:
                    break
        
        # Fill remaining if any
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p in used:
                continue
            new_pts = points + [p]
            if has_four_collinear(new_pts):
                continue
            points.append(p)
            used.add(p)
            
        return points

    def generate_structured(seed_val):
        """Generate structured point configurations with many collinear triples."""
        random.seed(seed_val)
        patterns = []
        
        # Pattern 1: Grid-like with perturbations
        for offset in range(5):
            pts = []
            for i in range(4):
                for j in range(5):
                    x = -40 + i * 20 + random.randint(-2, 2)
                    y = -40 + j * 20 + random.randint(-2, 2)
                    x = max(-100, min(100, x))
                    y = max(-100, min(100, y))
                    pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 2: Circle with perturbations
        for angle_off in [0, 0.1, 0.2]:
            pts = []
            for i in range(20):
                angle = 2 * math.pi * i / 20 + angle_off
                r = 50 + 8 * math.sin(i * 1.5)
                x = int(round(r * math.cos(angle)))
                y = int(round(r * math.sin(angle)))
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        # Pattern 3: Clustered
        for cx, cy in [(0,0), (20,20), (-20,-20), (30,-30)]:
            pts = []
            for i in range(20):
                x = cx + random.randint(-25, 25)
                y = cy + random.randint(-25, 25)
                x = max(-100, min(100, x))
                y = max(-100, min(100, y))
                pts.append((x, y))
            unique = list(dict.fromkeys(pts))
            while len(unique) < 20:
                unique.append((random.randint(-100, 100), random.randint(-100, 100)))
            patterns.append(unique[:20])
        
        return patterns

    def count_triples(points):
        """Count collinear triples for use as surrogate."""
        count = 0
        for i in range(20):
            slopes = {}
            xi, yi = points[i]
            for j in range(20):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            for k in slopes.values():
                if k >= 2:
                    count += k * (k - 1) // 2
        return count

    current_points = generate_points_greedy()
    current_score = max_no_three_collinear(current_points)
    best_points = current_points[:]
    best_score = current_score
    print(f"Initial score: {best_score}", flush=True)
    
    # Try structured initializations
    structured = generate_structured(42)
    for pts in structured:
        if time.time() - start_time > DEADLINE:
            break
        if not has_four_collinear(pts):
            score = max_no_three_collinear(pts)
            if score < best_score:
                best_score = score
                best_points = pts[:]
                print(f"Structured best: {best_score} at {time.time() - start_time:.1f}s", flush=True)
    
    T = 5.0
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        # Move: pick a random point and move it
        i = random.randint(0, 19)
        old_p = current_points[i]
        
        # New random position
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        new_p = (new_x, new_y)
        
        if new_p in current_points:
            continue
            
        # Check validity
        new_points = current_points[:]
        new_points[i] = new_p
        
        if has_four_collinear(new_points):
            continue
        
        new_score = max_no_three_collinear(new_points)
        
        if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
            current_points = new_points
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = new_points[:]
                print(f"New best: {best_score} at {time.time() - start_time:.1f}s", flush=True)
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return [[p[0], p[1]] for p in best_points]
