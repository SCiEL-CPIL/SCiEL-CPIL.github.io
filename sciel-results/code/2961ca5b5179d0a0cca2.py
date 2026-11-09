import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit
    
    # Strategy:
    # 1. Generate candidate sets using a "parabolic lattice" construction.
    #    Points are of the form (x, y) where x is in [0, N) and y is derived from x via a modular quadratic.
    #    Specifically, consider the set S_M = { (x, (a*x^2 + b*x + c) % M) for x in [0, N) }
    #    This set is known to have distinct distances from any point if M is sufficiently large and coprime to 2 (or other conditions).
    #    However, we need the points to be in [0, N)^2.
    #    If M >= N, the y-values are in [0, M). We can scale or map them.
    #    Actually, a simpler construction that guarantees no isosceles triangles is a "Sidon set" in 2D.
    #    A classic construction for no-3-term-arithmetic-progressions (which implies no isosceles with integer coords? No, not exactly).
    #    
    #    Let's use the property that if we take points (x, x^2) in a large enough space, they form a set with no isosceles triangles?
    #    Distance squared between (x1, x1^2) and (x2, x2^2) is (x1-x2)^2 + (x1^2-x2^2)^2 = (x1-x2)^2 * (1 + (x1+x2)^2).
    #    If we have three points x1, x2, x3, do we get isosceles?
    #    Dist(x1, x2) = Dist(x1, x3) => (x1-x2)^2(1+(x1+x2)^2) = (x1-x3)^2(1+(x1+x3)^2).
    #    This is a strong constraint.
    #    
    #    A more robust analytic construction for "no isosceles" (also called "weakly Sidon" or "distinct distances" sets) on a grid is to use a large prime modulus p > N^2 and take points (x, x^2 mod p) but scaled? No, we need integer grid [0, N)^2.
    #    
    #    Alternative: Use a greedy algorithm but with a *different* ordering strategy:
    #    Instead of random, use a "spatial" ordering (e.g., Hilbert curve, or row-major with large stride) to maximize independence.
    #    But the prompt asks for a DIFFERENT strategy family.
    #    
    #    Let's try a **Genetic Algorithm** on the *parameters* of a parametric family of sets.
    #    Family: S_{a,b} = { (x, (a*x + b) % N) for x in [0, N) }? No, lines have isosceles triangles.
    #    Family: S_{a,b} = { (x, (a*x^2 + b*x) % N) for x in [0, N) }?
    #    If we pick a, b such that the mapping x -> (x, y_x) is injective and the resulting set has no isosceles triangles.
    #    
    #    Let's stick to a high-performance **Iterated Local Search (ILS)** but with a **Bitset-based** representation for O(1) conflict checks?
    #    The current program uses sets of distances.
    #    
    #    Actually, the most "different" and potentially powerful approach for this specific "no isosceles" constraint is to realize that the constraint is local.
    #    We can use **Simulated Annealing** on the *indicator vector* of the grid.
    #    State: binary vector v of length N*N.
    #    Move: Flip a bit (add/remove point).
    #    Energy: Number of isosceles triangles. We want Energy = 0.
    #    Objective: Maximize sum(v) subject to Energy = 0.
    #    
    #    This is a different framing: instead of "add if valid", we allow invalid states and penalize them.
    #    This allows the search to escape local optima where greedy gets stuck.
    
    # Precompute all isosceles triples?
    # Number of triples is ~ (N^2)^3 = 68 billion. Too many.
    # But we can compute the energy incrementally.
    # Energy = sum over all points P of (number of pairs (Q,R) such that |P-Q| == |P-R|).
    # This is O(|S|^3) to compute from scratch, but O(|S|) to update if we maintain counts.
    
    # Let's implement Simulated Annealing on the set of points.
    # State: list of active points S.
    # We need to quickly calculate the "cost" (number of isosceles violations).
    # A violation is an isosceles triangle.
    # Let V(S) be the number of isosceles triangles in S.
    # We want to minimize V(S) and maximize |S|.
    # We can use a lexicographic objective: Primary: Minimize V(S). Secondary: Maximize |S|.
    # Or a combined score: Score = |S| - lambda * V(S).
    
    # To make SA fast, we need O(1) or O(|S|) updates for V(S).
    # When adding point P:
    # New violations involving P:
    # 1. P is apex: Pairs (Q,R) in S such that |P-Q|^2 == |P-R|^2.
    # 2. P is base: For each Q in S, pairs (R, P) such that |Q-R|^2 == |Q-P|^2? No.
    #    Isosceles with vertices {P, Q, R}:
    #    - |P-Q| == |P-R| (Apex P)
    #    - |P-Q| == |Q-R| (Apex Q)
    #    - |P-R| == |Q-R| (Apex R)
    
    # Let's maintain for each point A in S:
    # dist_count[A] = dict: distance^2 -> count of points B in S such that |A-B|^2 = d.
    # When adding P:
    # New triangles where P is a vertex:
    # For each Q in S:
    #   d_PQ = |P-Q|^2
    #   1. Apex P: Check if there is R in S such that |P-R|^2 == d_PQ.
    #      Count = dist_count[P][d_PQ] (before adding P, dist_count[P] is empty, so we build it).
    #      Actually, we need to check against existing S.
    #      Let c_P = number of R in S such that |P-R|^2 == d_PQ.
    #      These form c_P triangles (P, Q, R).
    #   2. Apex Q: Check if there is R in S such that |Q-R|^2 == d_PQ.
    #      Count = dist_count[Q].get(d_PQ, 0).
    #      These form dist_count[Q].get(d_PQ, 0) triangles (Q, P, R).
    #   3. Apex R: This is covered by "Apex Q" for the pair (P, R)?
    #      No. The triangle is (P, Q, R).
    #      If |P-R| == |Q-R|, then R is apex.
    #      This is equivalent to: For a fixed R, |Q-R| == |P-R|.
    #      This is the same condition as "Apex R".
    #      So for each pair (Q, P), we check:
    #      - Is there R in S such that |P-R| == |P-Q|? (Apex P) -> Count from P's perspective.
    #      - Is there R in S such that |Q-R| == |P-Q|? (Apex Q) -> Count from Q's perspective.
    #      - Is there R in S such that |R-P| == |R-Q|? (Apex R) -> This is not directly a count from P or Q.
    #        Wait. |R-P| == |R-Q| means R is on the perpendicular bisector.
    #        This condition is NOT captured by "distance from P" or "distance from Q" being equal to d_PQ.
    #        It requires checking for each R in S if |R-P| == |R-Q|.
    #        This makes the update O(|S|^2) if done naively.
    
    # O(|S|^2) update is too slow for SA with many steps.
    # However, if S is small (~50), O(2500) per step is fast.
    # 1 million steps * 2500 ops = 2.5 billion ops. Might be too slow in Python.
    # We need to be smarter or accept fewer steps.
    
    # Alternative: Stick to the "Valid Set" constraint (Energy=0) but use **Simulated Annealing** to allow temporary violations?
    # No, if we allow violations, we must track them.
    
    # Let's go back to the **Analytic Construction** idea but more rigorous.
    # The set of points (x, x^2 mod p) for x in [0, p) has no 3 collinear points.
    # Does it have no isosceles?
    # For p large enough, the distances are distinct.
    # If we take p = 61 (prime < 64), we get 61 points.
    # Map (x, y) in [0, 61)^2 to [0, 64)^2.
    # We can just use the points (x, x^2 % 61) for x in 0..60.
    # These are 61 points.
    # Are they valid?
    # If the modular parabola has no isosceles triangles in the integer lattice sense (Euclidean distance), we are done.
    # It is known that the graph of a quadratic function over a field of characteristic not 2 is a "Sidon set" for distances?
    # Actually, it's a "weak Sidon" set.
    # Let's test this hypothesis in the code. If it works, we have 61 points instantly.
    # Current best is ~10.6 (which is score, so maybe 10 or 11 points? Or if seed is 1, then 10 points).
    # If the seed program returns 1 point, score 10.625 means 10.625 points? No, score is integer.
    # "Score: number of points".
    # So the current program finds ~10 points.
    # If the analytic construction finds 61, that's a massive jump.
    
    # Let's implement a verifier to check if the modular parabola is valid.
    
    def verify_set(points):
        # points: list of (x, y)
        S = set(points)
        n = len(S)
        # Check all triples? O(n^3) is fine for n=61.
        # Or check distances from each point.
        for p in S:
            dists = set()
            for q in S:
                if p == q: continue
                d2 = (p[0]-q[0])**2 + (p[1]-q[1])**2
                if d2 in dists:
                    return False
                dists.add(d2)
        return True

    # Candidate 1: Modular Parabola
    # Try different primes p <= 64.
    best_points = []
    
    primes = [61, 59, 53, 47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3]
    
    for p in primes:
        # Generate points (x, x^2 % p)
        pts = [(x, (x*x) % p) for x in range(p)]
        # Scale to fit in 64x64? They already fit in p x p, and p <= 61 < 64.
        # So they are in [0, 61)^2 which is subset of [0, 64)^2.
        if verify_set(pts):
            if len(pts) > len(best_points):
                best_points = pts
        
    # If we found a large set, we can try to add more points?
    # Or try other parabolas: y = a*x^2 + b*x + c mod p.
    # For a fixed p, there are p^3 parabolas.
    # We can sample some.
    
    if len(best_points) < 20:
        # If the simple parabola didn't work or is small, try random linear transformations?
        # Or try a different analytic construction:
        # (x, x^3 mod p)?
        
        # Let's try a few random quadratic coefficients for the largest primes.
        for p in [61, 59, 53]:
            for _ in range(1000):
                if time.time() > deadline - 5: break
                a = random.randint(1, p-1)
                b = random.randint(0, p-1)
                c = random.randint(0, p-1)
                pts = [(x, (a*x*x + b*x + c) % p) for x in range(p)]
                # Quick check: are all points distinct?
                if len(set(pts)) < p: continue
                if verify_set(pts):
                    if len(pts) > len(best_points):
                        best_points = pts
    
    # If we still have a good set, try to improve it with a fast local search?
    # Given the time, if we have 60+ points, local search is risky.
    # But if we have < 20, we should use the local search from the previous strategy?
    # No, the directive says "rethinking the approach".
    # If the analytic construction works, it's a huge win.
    # If it fails, we should fallback to a robust greedy?
    # But we must output a complete program.
    
    # Let's add a fallback greedy if the analytic set is small.
    if len(best_points) < 15:
        # Fallback: Fast Greedy with numpy (similar to current but optimized)
        N_grid = 64
        xs = np.arange(N_grid)
        X, Y = np.meshgrid(xs, xs, indexing='ij')
        xi = (X.flatten() // N_grid).astype(np.int32)
        yi = (X.flatten() % N_grid).astype(np.int32)
        
        # Precompute distances
        dx = xi[:, None] - xi[None, :]
        dy = yi[:, None] - yi[None, :]
        d2_matrix = (dx*dx + dy*dy).astype(np.int32)
        
        S = []
        dist_from = {}
        
        order = list(range(N_grid*N_grid))
        random.shuffle(order)
        
        for p_idx in order:
            if time.time() > deadline - 5: break
            if len(S) == 0:
                S.append(p_idx)
                dist_from[p_idx] = set()
                continue
            
            # Check uniqueness of distances from p_idx to S
            d_vals = d2_matrix[p_idx, S]
            if len(set(d_vals)) != len(d_vals):
                continue
            
            # Check if distance to any a in S is already in dist_from[a]
            valid = True
            for a in S:
                d_ap = d2_matrix[a, p_idx]
                if d_ap in dist_from[a]:
                    valid = False
                    break
            if not valid:
                continue
            
            S.append(p_idx)
            dist_from[p_idx] = set()
            for a in S[:-1]:
                d_ap = d2_matrix[a, p_idx]
                dist_from[a].add(d_ap)
                dist_from[p_idx].add(d_ap)
        
        # Convert to tuples
        fallback_pts = [(p // N_grid, p % N_grid) for p in S]
        if len(fallback_pts) > len(best_points):
            best_points = fallback_pts

    return best_points

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} points")
