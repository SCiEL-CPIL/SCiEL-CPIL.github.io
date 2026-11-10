import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin (evaluator kills at 630s)
    
    # Strategy:
    # 1. Start with a set of points.
    # 2. Iteratively try to add more points from a candidate pool.
    # 3. If stuck, try to remove some points and re-add to break symmetry/deadlocks.
    # 
    # We will use a greedy approach with randomization and "simulated annealing" like cooling
    # or just random restarts with local search (k-exchange).
    # Given the time limit, a simple greedy with many restarts is a good baseline.
    # To improve, we can use a "maximal independent set" heuristic.
    
    # Candidate generation
    # We want points with small norms to allow packing more.
    # Norm 1: 22 points (e_i, -e_i)
    # Norm 2: 220 points (e_i +/- e_j)
    # Norm 4: 5280 points (4 non-zeros +/-1)
    # Norm 5: C(11,5)*2^5 = 462*32 = 14784 points
    # Norm 8: C(11,8)*2^8 = 165*256 = 42240 points (too many?)
    
    # Let's stick to Norm 2 and Norm 4 initially, as they are dense enough.
    # Actually, the current program uses Norm 2 as base and adds Norm 4.
    # Let's try a different base: Start with empty set, add Norm 1, then Norm 2, then Norm 4.
    
    # However, the constraint is min_dist_sq >= max_norm_sq.
    # If we include Norm 4 points, max_norm_sq = 4.
    # Then all pairs must have dist_sq >= 4.
    # For Norm 2 points u, v: dist_sq = 4 - 2<u,v>. Need 4 - 2<u,v> >= 4 => <u,v> <= 0.
    # So Norm 2 points must be orthogonal!
    # Max orthogonal set in Z^11 with norm 2?
    # e1+e2, e3+e4, ... we can have floor(11/2) = 5 such vectors.
    # Plus maybe one more if we use e1-e2? No, (e1+e2) . (e1-e2) = 0.
    # So we can pack more Norm 2 points if they are orthogonal.
    
    # Let's reconsider the structure.
    # If we only use Norm 2 points:
    # max_norm_sq = 2.
    # Condition: dist_sq >= 2.
    # dist_sq(u,v) = 4 - 2<u,v> >= 2 => <u,v> <= 1.
    # This is the kissing number problem for norm 2 vectors in Z^11.
    # The current program finds 220 Norm 2 points?
    # Wait, the current program starts with ALL 220 Norm 2 points.
    # Are they mutually compatible with max_norm_sq=2?
    # Check: u = e1+e2, v = e1-e2. <u,v> = 0. dist_sq = 4. OK.
    # u = e1+e2, v = e1+e3. <u,v> = 1. dist_sq = 2. OK.
    # u = e1+e2, v = -e1-e2. <u,v> = -2. dist_sq = 8. OK.
    # Is there any pair with <u,v> > 1?
    # Max dot product for distinct norm 2 vectors:
    # If they share 1 index and signs match: <u,v>=1.
    # If they share 2 indices and signs match: u=v (not distinct) or u=-v (<u,v>=-2).
    # So max <u,v> = 1.
    # Thus ALL 220 Norm 2 vectors are mutually compatible with max_norm_sq=2.
    # So k=220 is a valid solution using only Norm 2.
    
    # Can we add Norm 4 vectors?
    # If we add Norm 4, max_norm_sq becomes 4.
    # Then we need dist_sq >= 4 for ALL pairs.
    # This is a much stricter constraint.
    # The current program tries to find a subset of Norm 4 vectors compatible with the 220 Norm 2 vectors.
    # It requires <u, v> <= 1 for all u in Base (Norm 2), v in New (Norm 4).
    # And <v, w> <= 2 for all v, w in New.
    
    # Let's verify the score.
    # Current score is 110. This implies the current program returns 110 points?
    # Or maybe the seed score is 110 and we are at 110?
    # The prompt says "CURRENT PROGRAM (score 110.0000)".
    # The inspirations have scores 147+.
    # So we need to find a better packing.
    
    # Idea: What if we drop some Norm 2 vectors to allow more Norm 4 vectors?
    # Or use a different mix?
    
    # Let's try a greedy construction from scratch.
    # 1. Generate all candidates with norm_sq <= 4.
    # 2. Sort by norm_sq (ascending).
    # 3. Greedily add points if they satisfy the distance constraint w.r.t current set.
    #    Constraint: dist_sq(p, q) >= max(norm_sq(p), norm_sq(q)).
    #    Note: The global constraint is min_dist_sq >= max_norm_sq_global.
    #    This is tricky because adding a high-norm point increases the threshold for ALL pairs.
    #    
    # Alternative:
    # Fix a target max_norm_sq = M.
    # Find the largest set of points with norm_sq <= M such that all pairwise dist_sq >= M.
    # Then the answer is the max over M.
    
    # Possible M values: 1, 2, 4, 5, 8, ...
    # If M=1: Only norm 1 points. Max orthogonal set in Z^11 is 11 (e_i). 
    #            But we can have -e_i too? dist(e_i, -e_i) = 4 >= 1. 
    #            Actually, if M=1, we need dist_sq >= 1.
    #            All non-zero integer vectors have dist_sq >= 1? No, e1, e2 dist=2.
    #            But max_norm_sq must be 1. So all points must have norm_sq <= 1.
    #            Points: +/-e_i. 22 points.
    #            Check: dist(e_i, e_j) = 2 >= 1. dist(e_i, -e_j) = 2 >= 1. dist(e_i, -e_i) = 4 >= 1.
    #            So k=22 is valid for M=1.
            
    # If M=2: Points with norm_sq <= 2.
    #            Norm 1: 22 points.
    #            Norm 2: 220 points.
    #            Total candidates: 242.
    #            Constraint: dist_sq >= 2.
    #            We know all 220 Norm 2 points work together.
    #            Can we add Norm 1 points?
    #            u (norm 1), v (norm 2). dist_sq = 1 + 2 - 2<u,v> = 3 - 2<u,v>.
    #            Need 3 - 2<u,v> >= 2 => 1 - 2<u,v> >= 0 => <u,v> <= 0.5 => <u,v> <= 0.
    #            So Norm 1 point e_i must be orthogonal to all Norm 2 points in the set.
    #            If we include all 220 Norm 2 points, is there any e_i orthogonal to all?
    #            No, e_i has dot product 1 with e_i+e_j.
    #            So we can't add any Norm 1 points if we keep all Norm 2 points.
    #            Can we drop some Norm 2 points to add Norm 1 points?
    #            Norm 1 points are "cheaper" in terms of count? No, we want max count.
    #            220 is already quite high.
            
    # If M=4: Points with norm_sq <= 4.
    #            Norm 1: 22
    #            Norm 2: 220
    #            Norm 4: 5280
    #            Constraint: dist_sq >= 4.
    #            This is the regime the current program explores.
    
    # Let's analyze the M=4 case more closely.
    # We need a set S such that for all u, v in S, ||u-v||^2 >= 4.
    # This is equivalent to a code with minimum distance squared 4.
    # The points are in Z^11.
    
    # Let's try a greedy algorithm specifically for M=4.
    # 1. Start with empty set.
    # 2. Candidate pool: All vectors with norm_sq <= 4.
    # 3. Sort candidates by norm_sq, then randomly.
    # 4. For each candidate v:
    #       If for all u in S, ||u-v||^2 >= 4, add v to S.
    # 5. Repeat with random restarts.
    
    # To speed up, we can use a spatial index or just vectorized distance checks.
    # With ~5500 candidates, a full pairwise check is 5500^2 ~ 30M ops.
    # In Python, this is too slow per restart.
    # We need to be smarter.
    
    # Observation:
    # The set of all 220 Norm 2 vectors is a valid set for M=2.
    # For M=4, it is NOT valid because some pairs have dist_sq = 2.
    # Specifically, u=e1+e2, v=e1+e3. dist_sq = 2.
    # So we must remove conflicting pairs from the Norm 2 set.
    
    # Let's build a conflict graph for M=4.
    # Nodes: All candidates with norm_sq <= 4.
    # Edge (u,v) if ||u-v||^2 < 4.
    # We want Max Independent Set.
    
    # Since the graph is huge (5500 nodes), MIS is hard.
    # But maybe we can decompose it?
    
    # Let's try a different approach:
    # Use the 220 Norm 2 points as a starting point.
    # They form a "cluster" where many pairs are too close (dist 2).
    # We need to select a subset of them that are mutually dist >= 4.
    # This is equivalent to selecting a subset of Norm 2 points with <u,v> <= 0.
    # (Since dist_sq = 4 - 2<u,v> >= 4 => <u,v> <= 0).
    # So we need a set of mutually orthogonal Norm 2 vectors.
    # Max size of mutually orthogonal Norm 2 vectors in Z^11?
    # Each vector uses 2 dimensions.
    # We can have at most floor(11/2) = 5 such vectors.
    # Example: e1+e2, e3+e4, e5+e6, e7+e8, e9+e10.
    # The 11th dimension is unused.
    # So we can only keep 5 Norm 2 points if we want them to be mutually orthogonal.
    # This is a huge drop from 220.
    
    # Wait, do we NEED them to be orthogonal?
    # Yes, if they are both in the set and M=4.
    # So for M=4, we can keep at most 5 Norm 2 points.
    # This suggests M=4 might yield a smaller total set than M=2 (which yields 220).
    
    # Let's check M=2 again.
    # Set of 220 Norm 2 points.
    # Max norm sq = 2.
    # Min dist sq = 2.
    # Condition: min_dist_sq >= max_norm_sq => 2 >= 2. Valid.
    # So k=220 is valid.
    
    # Can we do better than 220?
    # What if we use M=5?
    # Candidates with norm_sq <= 5.
    # Norm 5 vectors: 5 non-zeros +/-1.
    # Count: C(11,5)*2^5 = 14784.
    # Condition: dist_sq >= 5.
    # For two Norm 2 points u, v: dist_sq = 4 - 2<u,v> >= 5 => -2<u,v> >= 1 => <u,v> <= -0.5 => <u,v> <= -1.
    # So Norm 2 points must have dot product <= -1.
    # This is even more restrictive than M=4 (which required <= 0).
    # So we can keep even fewer Norm 2 points.
    
    # It seems M=2 with 220 points is a strong candidate.
    # Can we add any points to the 220 Norm 2 set?
    # If we add a point with norm_sq > 2, say norm_sq = 5.
    # Then max_norm_sq becomes 5.
    # We need ALL pairs to have dist_sq >= 5.
    # The existing Norm 2 pairs have dist_sq >= 2. Some have dist_sq = 2.
    # These pairs would violate the new constraint.
    # So we would have to remove all Norm 2 pairs with dist_sq < 5.
    # This is likely to reduce the count significantly.
    
    # What if we don't start with all 220?
    # What if we find a large set of points all with norm_sq = 1?
    # Max orthogonal set in Z^11 is 11.
    # But we can have non-orthogonal if dist_sq >= 1.
    # All +/-e_i work. k=22.
    
    # What if we mix Norm 1 and Norm 2?
    # If we include Norm 2, max_norm_sq >= 2.
    # If we include only Norm 1 and Norm 2, max_norm_sq = 2.
    # Constraint: dist_sq >= 2.
    # We established that all 220 Norm 2 points are mutually compatible.
    # Can we add Norm 1 points?
    # e_i must be orthogonal to all selected Norm 2 points.
    # If we select a subset of Norm 2 points that avoids e_i, we can add e_i.
    # But we want to maximize total count.
    # 220 is large. Adding 11 Norm 1 points requires removing many Norm 2 points?
    # For each e_i, we must remove all Norm 2 points involving index i.
    # Number of Norm 2 points involving index i:
    # Pairs (i, j) for j!=i. 10 pairs.
    # Signs: 4 per pair.
    # Total 40 Norm 2 points involve index i.
    # So to add e_i, we must remove 40 Norm 2 points.
    # Net change: +1 - 40 = -39. Bad.
    
    # So it seems 220 is a local optimum for M=2.
    
    # Is there a way to get > 220?
    # Maybe using Norm 4 points exclusively?
    # If we use only Norm 4 points, max_norm_sq = 4.
    # Constraint: dist_sq >= 4.
    # This is equivalent to <u,v> <= 2 for all u,v.
    # We want a large subset of the 5280 Norm 4 vectors with pairwise dot product <= 2.
    
    # Let's estimate the size of such a set.
    # Random dot product between two Norm 4 vectors:
    # They share k indices.
    # If k=0, dot=0.
    # If k=1, dot = +/-1.
    # If k=2, dot = 2, 0, -2.
    # If k=3, dot = 3, 1, -1, -3.
    # If k=4, dot = 4, 2, 0, -2, -4.
    # We need dot <= 2.
    # So we cannot have dot = 3 or 4.
    # Dot = 4 happens if they are identical (not allowed) or... wait.
    # If they share 4 indices and all signs match, dot = 4.
    # If they share 3 indices and signs match, dot = 3.
    
    # So we must avoid pairs with high overlap and matching signs.
    
    # This looks like a code construction problem.
    # Let's try a greedy algorithm on the 5280 Norm 4 vectors.
    
    def generate_norm4_vectors():
        vecs = []
        for combo in itertools.combinations(range(11), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(11, dtype=np.int64)
                v[list(combo)] = signs
                vecs.append(v)
        return np.array(vecs)
    
    V4 = generate_norm4_vectors()
    N = len(V4)
    
    # Greedy MIS
    # We want to pick as many as possible such that dot(u,v) <= 2.
    
    # Precompute dot products? 5280x5280 is ~28M entries. 28M * 8 bytes = 224 MB.
    # This is feasible.
    
    # But computing it takes time.
    # V4 @ V4.T
    
    # Then we need to find a large independent set in the graph where edges are dot > 2.
    
    # Let's try a simple greedy:
    # 1. Compute D = V4 @ V4.T
    # 2. Create a boolean matrix Conflict = D > 2
    # 3. Greedy selection:
    #    - Start with all vertices active.
    #    - Pick a random active vertex v.
    #    - Add v to solution.
    #    - Remove v and all its neighbors (conflicts) from active set.
    #    - Repeat.
    
    # This is O(N * AvgDegree).
    
    # Let's implement this.
    
    def greedy_mis(V, threshold, deadline):
        N = len(V)
        D = V @ V.T
        Conflict = D > threshold
        
        active = np.ones(N, dtype=bool)
        solution = []
        
        # To speed up, we can use a list of active vertices and remove from it.
        # But boolean mask is easier with numpy.
        
        # Random restarts
        best_size = 0
        best_sol = []
        
        while time.time() < deadline:
            # Shuffle active? No, just pick random order.
            order = np.random.permutation(N)
            active = np.ones(N, dtype=bool)
            sol = []
            
            for v in order:
                if not active[v]:
                    continue
                sol.append(v)
                # Deactivate v and neighbors
                # active[Conflict[v]] = False
                # active[v] = False
                # This is vectorized, but creates a new array or modifies in place?
                # active[Conflict[v]] = False modifies in place.
                active[Conflict[v]] = False
                active[v] = False
                
            if len(sol) > best_size:
                best_size = len(sol)
                best_sol = sol.copy()
                
        return V[best_sol]
    
    # This might be slow in Python if N=5280.
    # Each iteration of the inner loop is O(N) due to vectorized deactivation?
    # No, active[Conflict[v]] is a boolean indexing.
    # Conflict[v] is a boolean array of length N.
    # So it's O(N) per pick.
    # Total O(N^2) per restart.
    # 5280^2 ~ 28M ops.
    # In Python, 28M ops might take ~1-2 seconds?
    # So we can do ~300 restarts in 600s?
    # Maybe.
    
    # Let's try to optimize.
    # Instead of vectorized deactivation, we can maintain a set of active vertices?
    # No, numpy is faster for boolean ops.
    
    # Let's run this greedy on Norm 4 vectors.
    
    start_time = time.time()
    deadline = start_time + 580
    
    V4 = generate_norm4_vectors()
    
    # Check if we can beat 220.
    # If greedy finds > 220, we return it.
    # Else, we return the 220 Norm 2 points.
    
    # Let's also consider a hybrid:
    # Maybe a subset of Norm 2 and Norm 4?
    # But the constraint is global.
    
    # Let's just run the greedy on Norm 4 and compare with 220.
    
    best_points = np.array([])
    best_k = 0
    
    # Baseline: 220 Norm 2 points
    candidates_2 = []
    for combo in itertools.combinations(range(11), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(11, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    base_2 = np.array(candidates_2, dtype=np.int64)
    best_points = base_2
    best_k = len(base_2)
    
    # Try Greedy on Norm 4
    # We need to be careful with time.
    # Let's limit the number of restarts or time per restart.
    
    N4 = len(V4)
    D4 = V4 @ V4.T
    Conflict4 = D4 > 2
    
    # Greedy
    for _ in range(100): # Max 100 restarts
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(N4)
        active = np.ones(N4, dtype=bool)
        sol = []
        
        for v in order:
            if not active[v]:
                continue
            sol.append(v)
            active[Conflict4[v]] = False
            active[v] = False
            
        if len(sol) > best_k:
            best_k = len(sol)
            best_points = V4[sol]
            
    return best_points

if __name__ == "__main__":
    points = kissing_number11()
    print(f"Found {len(points)} points")

if __name__ == "__main__":
    # Quick self-test
    points = kissing_number11()
    print(f"Found {len(points)} points")
    # Verify
    if len(points) > 0:
        norms = np.sum(points**2, axis=1)
        max_norm = np.max(norms)
        min_dist_sq = float('inf')
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        print(f"Max norm sq: {max_norm}, Min dist sq: {min_dist_sq}")
        if min_dist_sq >= max_norm:
            print("Valid!")
        else:
            print("INVALID!")
