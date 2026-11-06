"""Rewritten program: minimize largest no-three-collinear subset of 20 points."""
import random
import math
import time
from collections import defaultdict

def construct():
    random.seed(42)
    
    def normalize_line(p1, p2):
        """Return normalized (a, b, c) for line through p1, p2."""
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        g = math.gcd(a, math.gcd(b, c))
        if g != 0:
            a //= g
            b //= g
            c //= g
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def get_lines(points):
        """Return dictionary mapping line keys to set of point indices."""
        n = len(points)
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = normalize_line(points[i], points[j])
                lines[key].add(i)
                lines[key].add(j)
        return lines

    def get_triples(lines):
        """Extract triples from lines dict."""
        triples = []
        for key, pts in lines.items():
            if len(pts) == 3:
                p_list = sorted(pts)
                triples.append((p_list[0], p_list[1], p_list[2]))
        return triples

    def is_valid(points):
        """Check distinct, bounds, and no 4 collinear."""
        if len(set(points)) != len(points):
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        
        # Check no 4 collinear
        lines = get_lines(points)
        for pts in lines.values():
            if len(pts) >= 4:
                return False
        return True

    def calc_mis(points):
        """Calculate Maximum Independent Set size for the hypergraph of collinear triples."""
        n = len(points)
        lines = get_lines(points)
        triples = get_triples(lines)
        
        # conflict_mask[v] is a bitmask of vertices that form a triple with v and some other vertex in current_set?
        # Actually, for MIS in 3-uniform hypergraph, we want to select max vertices such that no triple is fully contained.
        # This is equivalent to finding the max subset S such that for every triple T, S does not contain all of T.
        # Wait, the problem asks for "largest subset with no three collinear".
        # This means we are looking for the size of the largest subset S of points such that no three points in S are collinear.
        # This is the independence number of the 3-uniform hypergraph where edges are collinear triples.
        
        # We can use a standard Branch and Bound for Independent Set in Hypergraphs.
        # A set S is independent if it contains no edge.
        # We want to maximize |S|.
        
        # Precompute for each vertex v, the list of edges (triples) it belongs to.
        # If we include v in S, we must ensure no edge containing v is fully in S.
        # This implies that for any edge {v, u, w}, we cannot have both u and w in S if v is in S.
        
        # Let's use a recursive search.
        # State: available_mask (vertices not decided yet), current_set_mask (vertices selected).
        # We want to maximize popcount(current_set_mask).
        
        # Optimization: If we select v, then for every edge {v, u, w}, we cannot select both u and w.
        # This creates constraints.
        
        # Alternative view: This is a 3-uniform hypergraph.
        # Let's stick to the previous DFS logic but ensure it's correct.
        # The previous logic:
        # If we include v, we remove from available any u such that there exists w in current_set where {v, u, w} is an edge.
        # This is correct for ensuring that no new edge is formed with v and two existing vertices.
        # But we also need to ensure that future selections don't form edges with v.
        # The logic `new_available &= ~(1 << u)` handles this: if w is in current_set, and {v, u, w} is an edge, then u cannot be selected later (because v and w are already in set, so u would complete the edge).
        
        # However, what if we select v, and later select u, and later select w?
        # When we select v, we check against current_set.
        # When we select u, we check against current_set (which includes v). If {v, u, w} is an edge, and w is not in current_set, u is allowed.
        # When we select w, we check against current_set (which includes v and u). If {v, u, w} is an edge, then v and u are in current_set, so w is forbidden.
        # So the logic holds.
        
        triples = get_triples(lines)
        conflict_pairs = [[] for _ in range(n)]
        for t in triples:
            i, j, k = t
            conflict_pairs[i].append((j, k))
            conflict_pairs[j].append((i, k))
            conflict_pairs[k].append((i, j))
            
        degrees = [len(cp) for cp in conflict_pairs]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        best = 0
        
        def dfs(available_mask, current_mask, current_size):
            nonlocal best
            
            # Pruning
            if current_size + bin(available_mask).count('1') <= best:
                return
            
            if available_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick next vertex from order that is in available_mask
            v = -1
            for idx in order:
                if available_mask & (1 << idx):
                    v = idx
                    break
            
            if v == -1:
                return
            
            v_bit = 1 << v
            
            # Option 1: Exclude v
            dfs(available_mask & ~v_bit, current_mask, current_size)
            
            # Option 2: Include v
            can_include = True
            for u, w in conflict_pairs[v]:
                if (current_mask & (1 << u)) and (current_mask & (1 << w)):
                    can_include = False
                    break
            
            if can_include:
                new_available = available_mask & ~v_bit
                for u, w in conflict_pairs[v]:
                    if current_mask & (1 << w):
                        new_available &= ~(1 << u)
                    if current_mask & (1 << u):
                        new_available &= ~(1 << w)
                        
                dfs(new_available, current_mask | v_bit, current_size + 1)
                
        dfs((1 << n) - 1, 0, 0)
        return best

    def generate_initial():
        while True:
            points = []
            seen = set()
            while len(points) < 20:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                p = (x, y)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
            if is_valid(points):
                return points

    points = generate_initial()
    current_score = calc_mis(points)
    
    best_points = list(points)
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110.0
    
    # Pre-generate a pool of candidate points for moves
    candidate_pool = set()
    for _ in range(2000):
        candidate_pool.add((random.randint(-100, 100), random.randint(-100, 100)))
    
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        if no_improve > 2000:
            break
            
        # Strategy: Try to move a point involved in a collinear triple to break it.
        lines = get_lines(points)
        triples = get_triples(lines)
        
        move_made = False
        if triples:
            # Pick a random triple
            t = random.choice(triples)
            # Pick one of the points in the triple to move
            idx = random.choice(t)
            old_point = points[idx]
            
            # Try to find a new position for this point
            # We want to minimize the number of new triples created.
            # We can sample a few candidates and pick the best.
            
            best_new_point = None
            best_new_score = current_score + 10 # High penalty
            
            # Sample from candidate pool or nearby
            # To be faster, we just try a few random candidates
            for _ in range(5):
                new_point = random.choice(list(candidate_pool))
                if new_point == old_point:
                    continue
                    
                new_points = list(points)
                new_points[idx] = new_point
                
                if not is_valid(new_points):
                    continue
                    
                # Quick check: count triples in new configuration
                # We can approximate by checking lines through new_point
                new_lines = get_lines(new_points)
                new_triples = get_triples(new_lines)
                
                # Heuristic: prefer fewer triples
                # But we need exact MIS score for comparison
                # To save time, we can estimate first
                if len(new_triples) < len(triples):
                    new_score = calc_mis(new_points)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_point = new_point
                elif len(new_triples) == len(triples) and random.random() < 0.1:
                    # Occasionally check even if triple count is same
                    new_score = calc_mis(new_points)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_point = new_point
            
            if best_new_point is not None:
                points[idx] = best_new_point
                current_score = best_new_score
                move_made = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(points)
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # No triples, score is 20. This is optimal.
            break
            
        if not move_made:
            no_improve += 1

    return [[p[0], p[1]] for p in best_points]
