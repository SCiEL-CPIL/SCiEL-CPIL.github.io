import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
    
    def count_4_collinear(points):
        """Return True if any line has >= 4 points"""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                cnt = 2
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        cnt += 1
                        if cnt >= 4:
                            return True
        return False
    
    def max_no_3_collinear(points):
        """Exact branch and bound for largest subset with no 3 collinear"""
        n = len(points)
        
        # Precompute collinearity: coll[i] is set of j such that i,j,k collinear for some k
        # Better: for each pair (i,j), the set of k collinear with i,j
        pair_coll = {}
        for i in range(n):
            for j in range(i + 1, n):
                ks = set()
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        ks.add(k)
                pair_coll[(i,j)] = ks
                pair_coll[(j,i)] = ks
        
        best = [0]
        
        def backtrack(selected_set, candidates, depth):
            if depth + len(candidates) <= best[0]:
                return
            if not candidates:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Pick a point
            p = candidates[0]
            rest = candidates[1:]
            
            # Option 1: Skip p
            backtrack(selected_set, rest, depth)
            
            # Option 2: Take p if no triple formed
            ok = True
            for s in selected_set:
                ks = pair_coll.get((p, s), set())
                for s2 in selected_set:
                    if s2 != s and s2 in ks:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                new_selected = selected_set | {p}
                # Filter candidates
                new_cands = []
                for q in rest:
                    safe = True
                    for s1 in new_selected:
                        for s2 in new_selected:
                            if s1 != s2:
                                if q in pair_coll.get((s1, s2), set()):
                                    safe = False
                                    break
                        if not safe:
                            break
                    if safe:
                        new_cands.append(q)
                backtrack(new_selected, new_cands, depth + 1)
        
        backtrack(set(), list(range(n)), 0)
        return best[0]
    
    def heuristic_score(points):
        """Fast heuristic: count collinear triples (higher = better for minimizing no-3 subset)"""
        n = len(points)
        triples = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples += 1
        return triples
    
    def local_search(points, max_iters=3000):
        """Improve by moving individual points to increase triple count"""
        pts = [list(p) for p in points]
        best_triples = heuristic_score(pts)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try each point
            for i in range(20):
                orig = pts[i][:]
                # Try several nearby positions
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        
                        cand = [list(p) for p in pts]
                        cand[i] = [nx, ny]
                        
                        if len(set(map(tuple, cand))) < 20:
                            continue
                        if count_4_collinear(cand):
                            continue
                        
                        tr = heuristic_score(cand)
                        if tr > best_triples:
                            pts = cand
                            best_triples = tr
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return pts
    
    best_config = None
    best_exact_score = 20
    
    # Generate diverse initial configurations
    configs = []
    
    # Type 1: Perturbed grid
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        pts = []
        for i in range(5):
            for j in range(4):
                x = i * 5 + random.randint(-1, 1)
                y = j * 5 + random.randint(-1, 1)
                pts.append([x, y])
        if len(set(map(tuple, pts))) == 20 and not count_4_collinear(pts):
            configs.append(pts)
    
    # Type 2: Random in box
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        pts = []
        for _ in range(20):
            pts.append([random.randint(-15, 15), random.randint(-15, 15)])
        if len(set(map(tuple, pts))) == 20 and not count_4_collinear(pts):
            configs.append(pts)
    
    # Evaluate each config with local search
    for pts in configs:
        if time.time() > deadline - 20:
            break
        
        # Local search to improve
        improved = local_search(pts, max_iters=1000)
        
        # Compute exact score
        score = max_no_3_collinear(improved)
        
        if score < best_exact_score:
            best_exact_score = score
            best_config = improved
    
    if best_config is None:
        # Fallback
        best_config = [[x, x * x] for x in range(-10, 10)]
    
    return best_config
