import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def collinear3(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def line_signature(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0:
            return ('v', p1[0])
        if dy == 0:
            return ('h', p1[1])
        if dx > 0:
            return (dy, dx, p1[0] * dy - p1[1] * dx)
        else:
            return (-dy, -dx, p1[0] * (-dy) - p1[1] * (-dx))

    def has_four_collinear(pts):
        from collections import defaultdict
        line_map = defaultdict(list)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                sig = line_signature(pts[i], pts[j])
                line_map[sig].append((i, j))
        for sig, pairs in line_map.items():
            if len(pairs) >= 6:
                return True
            if len(pairs) == 5:
                # C(4,2)=6, so 5 pairs means 4 points
                return True
            if len(pairs) >= 4:
                # Check if any 4 points share the line
                # For a line, points are collinear if they all lie on same line
                # 5 pairs can come from 4 points (6 pairs) - no, 4 points give 6 pairs
                # 5 pairs from 4 points is impossible (4 points give 6 pairs)
                # Actually 3 points give 3 pairs, 4 points give 6 pairs
                # So 5 pairs must involve at least 4 points? No.
                # 3 points: 3 pairs. 4 points: 6 pairs.
                # So 4,5 pairs must involve exactly... let me think.
                # If we have k collinear points, we get C(k,2) pairs.
                # k=3 -> 3 pairs, k=4 -> 6 pairs.
                # So 4 or 5 pairs means the pairs might be from a line with 3 points (3 pairs)
                # plus an extra pair that's on the same line? No, if 4 points are collinear we get 6 pairs.
                # So we just need to check if any line has >= 4 points.
                # Let's just collect all points on the line.
                pts_on_line = set()
                for i, j in pairs:
                    pts_on_line.add(i)
                    pts_on_line.add(j)
                if len(pts_on_line) >= 4:
                    return True
        return False

    def count_triples(pts):
        """Count number of collinear triples."""
        from collections import defaultdict
        line_map = defaultdict(set)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                sig = line_signature(pts[i], pts[j])
                line_map[sig].add(i)
                line_map[sig].add(j)
        total = 0
        for sig, pt_set in line_map.items():
            k = len(pt_set)
            if k >= 3:
                total += k * (k - 1) * (k - 2) // 6
        return total

    def exact_score(pts):
        """
        Compute largest subset with no 3 collinear using branch and bound.
        For 20 points, this should be fast.
        """
        n = len(pts)
        # Precompute: for each pair (i,j), which other points k are collinear
        # Actually, we need: for each triple, are they collinear?
        # Better: for each point i, and each other point j, the line through them.
        # For the independence number: we want max subset S such that no 3 points in S are collinear.
        # This is the same as: in the 3-uniform hypergraph of collinear triples, find max independent set.
        
        # Precompute collinear triples
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear3(pts[i], pts[j], pts[k]):
                        triples.append((i, j, k))
        
        # For each pair (i,j), store which k's form a collinear triple
        # Actually for B&B, we track: selected set, and for each candidate, 
        # whether adding it would create a collinear triple with two already selected.
        
        # Precompute: for each i, for each j != i, the set of k such that (i,j,k) collinear
        # Better: for each pair (i,j), the set of k (k not i,j) collinear with both.
        pair_collinear = {}
        for (i, j, k) in triples:
            for pair in [(i,j), (i,k), (j,k)]:
                pair_collinear.setdefault(pair, set()).add(pair[0] if pair[1] in (i,j,k) and pair[0] not in (i,j) else pair[1])
        
        # Simpler: for each unordered pair {i,j}, store set of k collinear with both i and j
        pair_k = {}
        for (i, j, k) in triples:
            for a, b, c in [(i,j,k), (j,i,k), (i,k,j), (k,i,j), (j,k,i), (k,j,i)]:
                pass
            # Just add to the canonical pair
            for pair_set in [frozenset({i,j}), frozenset({i,k}), frozenset({j,k})]:
                pair_k.setdefault(pair_set, set()).add(i if i not in pair_set else (j if j not in pair_set else k))
        
        # Let me redo this more carefully
        pair_k = {}
        for (i, j, k) in triples:
            pair_k.setdefault(frozenset({i,j}), set()).add(k)
            pair_k.setdefault(frozenset({i,k}), set()).add(j)
            pair_k.setdefault(frozenset({j,k}), set()).add(i)
        
        # B&B: 
        # State: (available, selected_size)
        # available: set of points we can still add
        # When we add point p, we must remove from available any point q such that
        # there exists r in selected with {p,q,r} collinear.
        # I.e., for each r in selected, if p and r are a pair, we must remove all k in pair_k[frozenset({p,r})].
        
        # To speed up, we can use a different approach:
        # Greedy with backtracking, or use the fact that n=20 is small.
        
        # Actually, let's use a simpler B&B:
        # Sort points by some heuristic.
        # Use recursion with memoization on (available_tuple, )? No, available can be large.
        
        # For n=20, a well-implemented B&B should work.
        # Key insight: the "conflict" graph is sparse.
        
        # Let's use a different formulation:
        # We want to find max S subset of {0..n-1} such that no triple in S is collinear.
        # This is equivalent to: for every collinear triple (i,j,k), not all of i,j,k are in S.
        
        # Use branch and bound with the following:
        # - Maintain current selected set S and remaining candidates R
        # - Upper bound: |S| + |R|
        # - If upper bound <= best, prune
        # - Pick a candidate p from R (heuristic: most constrained)
        # - Branch 1: don't include p
        # - Branch 2: include p, then remove from R all q such that there exists r in S with (p,q,r) collinear
        
        best = [0]
        
        def bnb(S, R):
            if not R:
                if len(S) > best[0]:
                    best[0] = len(S)
                return
            if len(S) + len(R) <= best[0]:
                return
            
            # Heuristic: pick the point in R that is in the most collinear triples with S
            # or just pick the first one
            p = min(R, key=lambda x: x)
            
            # Branch 1: exclude p
            bnb(S, R - {p})
            
            # Branch 2: include p
            new_R = set(R)
            new_R.discard(p)
            valid = True
            for r in S:
                pair = frozenset({p, r})
                if pair in pair_k:
                    for k in pair_k[pair]:
                        if k in new_R:
                            new_R.discard(k)
                        if k in S:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Also need to check: is p collinear with two points in S?
                # Already checked above
                bnb(S | {p}, new_R)
        
        # Initial: S = empty, R = all points
        all_pts = set(range(n))
        bnb(frozenset(), all_pts)
        
        return best[0]

    def greedy_score(pts, n_iter=100):
        """Fast greedy to get a lower bound on the score."""
        n = len(pts)
        best_lb = 0
        for _ in range(n_iter):
            order = list(range(n))
            random.shuffle(order)
            selected = []
            for i in order:
                can_add = True
                for a in selected:
                    for b in selected:
                        if a < b and collinear3(pts[a], pts[b], pts[i]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    selected.append(i)
            if len(selected) > best_lb:
                best_lb = len(selected)
        return best_lb

    # Start with a good base configuration
    # 6 lines of 3 + 2 isolated points
    base_pts = [
        [10, 0], [11, 0], [12, 0],
        [-12, 2], [-11, 2], [-10, 2],
        [10, 4], [11, 4], [12, 4],
        [-12, 6], [-11, 6], [-10, 6],
        [10, 8], [11, 8], [12, 8],
        [-12, 10], [-11, 10], [-10, 10],
        [50, 50],
        [-50, -50]
    ]
    
    # Verify base
    assert len(set(map(tuple, base_pts))) == 20
    assert not has_four_collinear(base_pts)
    
    best_pts = base_pts[:]
    best_score = exact_score(base_pts)
    print(f"Base score: {best_score}")
    
    # Try to improve by local search
    # Strategy: try to create more collinear triples to reduce independence number
    # Key idea: overlapping lines of 3 can be more efficient than disjoint lines
    
    # Try different structured configurations
    # Configuration 1: 5 lines of 3 (15 points) + 5 isolated
    # Configuration 2: overlapping lines
    
    # Let's try a more systematic search
    # Use simulated annealing or local search to maximize triples
    
    def local_search(pts, max_iter=2000):
        current = pts[:]
        current_triples = count_triples(current)
        best_local = current[:]
        best_local_triples = current_triples
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            idx = random.randint(0, 19)
            old_x, old_y = current[idx]
            
            # Try a random nearby move
            if random.random() < 0.7:
                new_x = old_x + random.randint(-8, 8)
                new_y = old_y + random.randint(-8, 8)
            else:
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
            
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
            
            new_pts = current[:]
            new_pts[idx] = [new_x, new_y]
            
            if len(set(map(tuple, new_pts))) != 20:
                continue
            if has_four_collinear(new_pts):
                continue
            
            new_triples = count_triples(new_pts)
            
            # Accept if better or with some probability
            if new_triples >= current_triples:
                current = new_pts
                current_triples = new_triples
                if current_triples > best_local_triples:
                    best_local_triples = current_triples
                    best_local = current[:]
            else:
                # Probabilistic acceptance (simulated annealing style)
                # Simple: accept with small probability
                if random.random() < 0.01:
                    current = new_pts
                    current_triples = new_triples
        
        return best_local
    
    # Try multiple random restarts
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Start from base or a perturbed version
        if r == 0:
            start_pts = base_pts[:]
        else:
            start_pts = base_pts[:]
            for _ in range(5):
                idx = random.randint(0, 19)
                nx = start_pts[idx][0] + random.randint(-20, 20)
                ny = start_pts[idx][1] + random.randint(-20, 20)
                if abs(nx) <= 100 and abs(ny) <= 100:
                    start_pts[idx] = [nx, ny]
                    if len(set(map(tuple, start_pts))) != 20 or has_four_collinear(start_pts):
                        start_pts[idx] = base_pts[idx]
        
        # Local search
        improved = local_search(start_pts, max_iter=1000)
        
        # Evaluate exact score
        s = exact_score(improved)
        if s < best_score:
            best_score = s
            best_pts = improved[:]
            print(f"Improved to score {best_score} at restart {r}")
    
    # Also try a few random configurations with heavy local search
    for r in range(5):
        if time.time() > deadline - 10:
            break
        
        # Generate random valid configuration
        pts = []
        attempts = 0
        while len(pts) < 20 and attempts < 1000:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            candidate = [[x, y]]
            for p in pts:
                candidate.append(p)
            if len(set(map(tuple, candidate))) == len(candidate) and not has_four_collinear(candidate):
                pts.append([x, y])
            attempts += 1
        
        if len(pts) == 20:
            improved = local_search(pts, max_iter=800)
            s = exact_score(improved)
            if s < best_score:
                best_score = s
                best_pts = improved[:]
                print(f"Random improved to score {best_score}")
    
    return best_pts
