import numpy as np
import random
import time
import itertools

def construct():
    def is_valid(points):
        if len(points) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if len(set(map(tuple, points))) != 20:
            return False
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    def max_no_3_collinear(points):
        """Exact branch-and-bound for maximum subset with no 3 collinear.
        Models as: find max independent set in hypergraph where hyperedges are collinear triples.
        Uses greedy coloring on auxiliary conflict graph for upper bounds."""
        n = len(points)
        
        # Find all collinear triples
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        triples.append((i, j, k))
        
        if not triples:
            return n
        
        # Build conflict graph: edge (u,v) exists if there's a triple containing both u and v
        # This gives us a graph where independent sets correspond to no-3-collinear subsets
        # (if no edge in conflict graph within subset, then no triple fully contained)
        # Actually: if subset S has no 3 collinear, then for every triple (i,j,k), not all in S.
        # The complement: min vertex cover of hypergraph.
        # Let's use: max subset with no triple fully contained = n - min vertex cover of hypergraph.
        # Or directly: branch and bound on the hypergraph.
        
        # Better approach: branch on triples. For each triple, at least one point must be excluded.
        # State: set of candidates (bitmask), current excluded set size.
        # Upper bound: current_excluded + min_additional_exclusions_needed
        
        # Precompute for each point, which triples it's in
        point_triples = [[] for _ in range(n)]
        for idx, (a, b, c) in enumerate(triples):
            point_triples[a].append(idx)
            point_triples[b].append(idx)
            point_triples[c].append(idx)
        
        # Greedy coloring upper bound:
        # If we have candidates C, the max independent set in the "conflict hypergraph" 
        # is at most the chromatic number of a related graph.
        # Simpler upper bound: for each point in C, it's in some triples.
        # We need to exclude at least ceil(|active_triples| / 3) points? No, that's weak.
        
        # Use: upper bound on max no-3-collinear subset of C is |C| - lower_bound_on_vertex_cover(C)
        # Lower bound on vertex cover: greedy matching on triples.
        
        def greedy_vertex_cover_lower_bound(candidates_mask):
            """Lower bound on min vertex cover size for triples restricted to candidates."""
            remaining = set()
            for idx, (a, b, c) in enumerate(triples):
                if (candidates_mask >> a) & 1 and (candidates_mask >> b) & 1 and (candidates_mask >> c) & 1:
                    remaining.add(idx)
            
            if not remaining:
                return 0
            
            # Greedy: repeatedly pick a triple, add one of its points to cover, remove all triples covered
            cover_size = 0
            covered = set()
            while remaining:
                # Pick triple with most uncovered points (or just first)
                t_idx = next(iter(remaining))
                a, b, c = triples[t_idx]
                # Pick the point that covers most remaining triples
                best_point = -1
                best_count = -1
                for pt in [a, b, c]:
                    cnt = 0
                    for r_idx in remaining:
                        if r_idx in covered:
                            continue
                        ra, rb, rc = triples[r_idx]
                        if pt in (ra, rb, rc):
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_point = pt
                if best_point == -1:
                    best_point = a
                cover_size += 1
                # Remove all triples containing best_point
                to_remove = set()
                for r_idx in remaining:
                    if r_idx in covered:
                        continue
                    ra, rb, rc = triples[r_idx]
                    if best_point in (ra, rb, rc):
                        to_remove.add(r_idx)
                remaining -= to_remove
            
            return cover_size
        
        best = 0
        full_mask = (1 << n) - 1
        
        def solve(candidates_mask, excluded_count):
            nonlocal best
            
            cand_count = bin(candidates_mask).count('1')
            
            # Upper bound: even if we exclude 0 more, max size is cand_count
            # Better: max size = cand_count - lower_bound_on_additional_exclusions
            # But we already excluded 'excluded_count' points. The max no-3-collinear subset
            # from original is at most (n - excluded_count) but constrained by triples.
            # Actually: the subset we're building is from candidates. Max possible = cand_count.
            # But we need to exclude more to break triples.
            # Upper bound on final answer = (n - excluded_count) is trivial.
            # Better: upper bound = cand_count (since we can only pick from candidates)
            # But we need to break all triples within candidates.
            # Min additional exclusions from candidates = vertex_cover_lower_bound(candidates_mask)
            # So max subset size from candidates <= cand_count - vclb
            # Total answer <= excluded_count + (cand_count - vclb)? No.
            # The answer is the size of the no-3-collinear subset. We've excluded 'excluded_count' points.
            # From the remaining 'cand_count' points, we can keep at most cand_count - vclb.
            # So total kept <= (n - excluded_count) but that's n - excluded_count = cand_count.
            # The constraint is: kept <= cand_count - vclb(candidates_mask)
            # So upper bound on answer = cand_count - vclb(candidates_mask)
            
            if cand_count == 0:
                if n - excluded_count > best:
                    best = n - excluded_count
                return
            
            vclb = greedy_vertex_cover_lower_bound(candidates_mask)
            ub = cand_count - vclb
            
            if ub <= best:
                return
            
            if ub >= n - excluded_count:
                # Can't prune
                pass
            
            # Find a triple fully in candidates to branch on
            branch_idx = -1
            for idx, (a, b, c) in enumerate(triples):
                if (candidates_mask >> a) & 1 and (candidates_mask >> b) & 1 and (candidates_mask >> c) & 1:
                    branch_idx = idx
                    break
            
            if branch_idx == -1:
                # No triples in candidates, all can be kept
                total_kept = (n - excluded_count)
                if total_kept > best:
                    best = total_kept
                return
            
            a, b, c = triples[branch_idx]
            
            # Branch: exclude a, or exclude b, or exclude c
            # To improve pruning, order branches by how many other triples they cover
            # Exclude the point that's in the most triples within candidates
            
            def triple_count_in_candidates(pt):
                cnt = 0
                for t_idx in point_triples[pt]:
                    ta, tb, tc = triples[t_idx]
                    if ((candidates_mask >> ta) & 1) and ((candidates_mask >> tb) & 1) and ((candidates_mask >> tc) & 1):
                        cnt += 1
                return cnt
            
            # Sort branch points by triple count descending (exclude most constrained first)
            branch_points = sorted([a, b, c], key=lambda p: triple_count_in_candidates(p), reverse=True)
            
            for pt in branch_points:
                solve(candidates_mask & ~(1 << pt), excluded_count + 1)
        
        solve(full_mask, 0)
        return best

    # Constructive heuristic: maximize triple density
    # Strategy: use a small set of lines, place 3 points on each, with intersections
    # to create many overlapping triples.
    
    random.seed(12345)
    
    def generate_config():
        """Generate a configuration with high triple density."""
        # Use a 5x4 grid base, then perturb to break 4-collinearities
        # while keeping many 3-collinearities.
        
        # Start with points on a few lines
        # Lines: y = mx + c for small m, c
        # Place 3 points on each line, sharing intersection points
        
        points = []
        used = set()
        
        # Define lines as (slope, intercept) pairs, use rational slopes
        # Line: y = (p/q)*x + r, or ax + by = c
        # Use integer coefficient lines
        
        # Generate candidate lines with small coefficients
        candidate_lines = []
        for a in range(-3, 4):
            for b in range(-3, 4):
                if a == 0 and b == 0:
                    continue
                # Normalize: gcd(a,b)=1, a>0 or (a==0 and b>0)
                from math import gcd
                g = gcd(abs(a), abs(b))
                if g > 0:
                    a //= g
                    b //= g
                if a < 0 or (a == 0 and b < 0):
                    a, b = -a, -b
                # c range: keep points in [-100,100]
                # |ax + by| <= |a|*100 + |b|*100
                max_c = abs(a) * 100 + abs(b) * 100
                for c in range(-max_c, max_c + 1, 7):  # Step to limit count
                    candidate_lines.append((a, b, c))
        
        # Pick a subset of lines that intersect nicely
        # For each line, we can place up to 3 integer points in range
        # Choose lines and their points to maximize triple overlaps
        
        # Simpler: start with a perturbed grid
        base_points = []
        for i in range(4):
            for j in range(5):
                base_points.append([i * 15, j * 15])
        
        # Perturb to break 4-collinearities
        points = [p[:] for p in base_points]
        
        for _ in range(500):
            if is_valid(points):
                break
            # Find a 4-collinear set
            n = len(points)
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    collinear = []
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        x3, y3 = points[k]
                        if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                            collinear.append(k)
                    if len(collinear) >= 2:
                        # Move one of the collinear points
                        idx = random.choice(collinear)
                        for _ in range(20):
                            nx = points[idx][0] + random.randint(-8, 8)
                            ny = points[idx][1] + random.randint(-8, 8)
                            if -100 <= nx <= 100 and -100 <= ny <= 100:
                                if (nx, ny) not in used:
                                    used.discard(tuple(points[idx]))
                                    points[idx] = [nx, ny]
                                    used.add((nx, ny))
                                    found = True
                                    break
                        if found:
                            break
                if found:
                    break
            if not found:
                # Random move
                idx = random.randint(0, 19)
                for _ in range(20):
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    if (nx, ny) not in used:
                        used.discard(tuple(points[idx]))
                        points[idx] = [nx, ny]
                        used.add((nx, ny))
                        break
        
        return points
    
    # Generate initial config
    points = generate_config()
    if not is_valid(points):
        # Fallback: parabola
        points = [[x, x * x] for x in range(-10, 10)]
    
    current_score = max_no_3_collinear(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local search with time limit
    time_limit = time.time() + 115
    
    while time.time() < time_limit:
        # Try moving a single point to a position that creates more triples
        # or breaks fewer existing triples
        idx = random.randint(0, 19)
        old_p = points[idx][:]
        
        # Try several candidate positions
        candidates = []
        # Nearby positions
        for dx in range(-10, 11, 3):
            for dy in range(-10, 11, 3):
                nx = old_p[0] + dx
                ny = old_p[1] + dy
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    candidates.append([nx, ny])
        # Random positions
        for _ in range(20):
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            candidates.append([nx, ny])
        
        improved = False
        for new_p in candidates:
            if (new_p[0], new_p[1]) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                continue
            
            points[idx] = new_p[:]
            
            if not is_valid(points):
                points[idx] = old_p
                continue
            
            new_score = max_no_3_collinear(points)
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                improved = True
                break
            else:
                points[idx] = old_p
        
        if not improved:
            # Try 2-opt: swap two points
            i1, i2 = random.sample(range(20), 2)
            p1, p2 = points[i1][:], points[i2][:]
            points[i1], points[i2] = p2, p1
            if is_valid(points):
                new_score = max_no_3_collinear(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                else:
                    points[i1], points[i2] = p1, p2
            else:
                points[i1], points[i2] = p1, p2
    
    return best_points
