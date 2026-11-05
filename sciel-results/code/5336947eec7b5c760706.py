import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    N = 20
    best_score = N
    best_points = None
    current_points = None
    
    # Helper: Check if a set of points has any 4 collinear
    def has_four_collinear(pts):
        n = len(pts)
        # Precompute pairs? No, just check all 4-subsets
        # 20 choose 4 = 4845. Very fast.
        for combo in combinations(range(n), 4):
            p1, p2, p3, p4 = [pts[i] for i in combo]
            # Check collinearity
            # (x2-x1)(y3-y1) == (y2-y1)(x3-x1)
            # and (x2-x1)(y4-y1) == (y2-y1)(x4-x1)
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            # Slope 12 vs 13
            if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
                continue
            # Slope 12 vs 14
            if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
                continue
            return True
        return False

    # Helper: Get all collinear triples
    def get_triples(pts):
        triples = []
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples.append((i, j, k))
        return triples

    # Helper: Calculate exact max 3-free subset size
    # Uses Branch and Bound
    def max_3_free_size(pts):
        n = len(pts)
        triples = get_triples(pts)
        # Map point index to list of triples it belongs to
        point_to_triples = [[] for _ in range(n)]
        for t in triples:
            for idx in t:
                point_to_triples[idx].append(t)
        
        # State: (current_set, next_candidate_index)
        # We want to maximize the size of the set such that no triple is fully contained.
        # This is the Independence Number of the 3-uniform hypergraph.
        
        best = 0
        # Sort points by degree (number of triples) descending to prune earlier?
        # Or just iterate.
        
        # Recursive B&B
        # To speed up, we can use a greedy upper bound.
        # Upper bound: current_size + remaining_points
        # Better upper bound: current_size + max_independent_set_of_remaining_hypergraph
        # For now, simple B&B with strong pruning.
        
        # We represent the current set as a bitmask or list.
        # Since N=20, bitmask is efficient.
        
        # Precompute conflict masks for pairs?
        # If we pick i and j, we cannot pick any k such that (i,j,k) is a triple.
        # Let's precompute `conflict_mask[i][j]` = bitmask of points k that form a triple with i and j.
        conflict_mask = [[0] * n for _ in range(n)]
        for t in triples:
            i, j, k = t
            conflict_mask[i][j] |= (1 << k)
            conflict_mask[j][i] |= (1 << k)
            conflict_mask[i][k] |= (1 << j)
            conflict_mask[k][i] |= (1 << j)
            conflict_mask[j][k] |= (1 << i)
            conflict_mask[k][j] |= (1 << i)
            
        # Also, if we pick i, we don't immediately ban j, but if we pick i and j, we ban k.
        
        def solve(mask, current_size, last_picked):
            nonlocal best
            # mask: bitmask of available points
            # current_size: size of current set
            # last_picked: index of last picked point, or -1
            
            if mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Upper bound: current_size + bit_count(mask)
            if current_size + bin(mask).count('1') <= best:
                return
            
            # Heuristic: Pick a point with highest degree in the remaining subgraph?
            # Or just pick the first available point.
            # To improve pruning, pick a point that is in many triples with other available points.
            
            # Let's try picking the point with the smallest index first, but maybe branching on high-degree points first is better.
            # For simplicity, iterate through available points.
            
            # Optimization: If there are no triples left in the available mask, we can take all remaining.
            # Check if any triple is fully contained in mask.
            has_triple = False
            for t in triples:
                # Check if all 3 points in t are in mask
                if (mask & (1 << t[0])) and (mask & (1 << t[1])) and (mask & (1 << t[2])):
                    has_triple = True
                    break
            
            if not has_triple:
                # All remaining points are safe to add
                new_size = current_size + bin(mask).count('1')
                if new_size > best:
                    best = new_size
                return
            
            # Branching
            # Choose a point to branch on.
            # Let's pick a point involved in a triple.
            # Find any triple in mask
            branch_triple = None
            for t in triples:
                if (mask & (1 << t[0])) and (mask & (1 << t[1])) and (mask & (1 << t[2])):
                    branch_triple = t
                    break
            
            if not branch_triple:
                 # Should be handled above
                 new_size = current_size + bin(mask).count('1')
                 if new_size > best:
                     best = new_size
                 return
            
            i, j, k = branch_triple
            
            # Option 1: Do not pick i
            solve(mask & ~(1 << i), current_size, -1)
            
            # Option 2: Pick i
            # If we pick i, we must not pick any pair (j,k) that forms a triple with i?
            # No, we just add i to the set.
            # But if we add i, we cannot have j and k both in the final set if (i,j,k) is a triple.
            # This constraint is handled by the "has_triple" check in the next recursion?
            # No, the "has_triple" check checks if a triple is *fully contained* in the *available* mask.
            # This logic is flawed for hypergraph independent set.
            
            # Correct B&B for Hypergraph Independent Set:
            # State: Set of chosen points S. Set of remaining points R.
            # Constraint: No triple from H is subset of S.
            # If we add v to S, we must ensure no triple in H is subset of S U {v}.
            # This means if there is a triple {v, u, w}, we cannot have both u and w in S (which is already true if S is valid) 
            # AND we cannot pick both u and w in the future if we pick v?
            # Actually, if we pick v, then u and w become "conflicting" with each other in the context of v.
            # Specifically, if we pick v, then we cannot pick a set that contains both u and w.
            # This is complex.
            
            # Simpler B&B for N=20:
            # Just enumerate subsets? 2^20 is 1M.
            # Check if subset is 3-free.
            # This is too slow if done naively for every step.
            
            # Let's use a different approach for scoring:
            # The score is the minimum vertex cover of the hypergraph.
            # We can use a greedy algorithm to find a vertex cover.
            # The size of the greedy cover is an upper bound on the min cover.
            # So Score <= N - GreedyCover.
            # We want to minimize Score, so we want to maximize GreedyCover.
            # This is a proxy.
            
            # Let's implement a fast greedy max 3-free set finder.
            # It's an upper bound on the true max 3-free set? No, it's a valid set, so it's a LOWER bound on the max size.
            # Wait.
            # Max 3-free set size = $\alpha$.
            # Greedy finds a set $S$ of size $|S|$. So $\alpha \ge |S|$.
            # We want to MINIMIZE $\alpha$.
            # So we want to make the greedy result small?
            # No, if greedy finds a small set, $\alpha$ could still be large.
            # We need an UPPER bound on $\alpha$ to prune?
            # Or we just calculate the exact $\alpha$ for the best candidates.
            
            # Given N=20, let's just calculate exact $\alpha$ using a simple recursive search with memoization or just simple B&B that is correct.
            
            # Let's restart the scoring function logic.
            pass

    # Revised Scoring: Exact Max 3-Free Subset
    def calculate_score(pts):
        n = len(pts)
        triples = get_triples(pts)
        
        # If no triples, score is N
        if not triples:
            return n
            
        # B&B for Max Independent Set in 3-uniform Hypergraph
        # We use a bitmask for the set of available points.
        # We want to find the max size subset of available points that contains no triple.
        
        # Precompute for each pair (i,j) the mask of k such that (i,j,k) is a triple.
        # pair_conflict[i][j]
        pair_conflict = [[0]*n for _ in range(n)]
        for t in triples:
            i,j,k = t
            pair_conflict[i][j] |= (1<<k)
            pair_conflict[j][i] |= (1<<k)
            pair_conflict[i][k] |= (1<<j)
            pair_conflict[k][i] |= (1<<j)
            pair_conflict[j][k] |= (1<<i)
            pair_conflict[k][j] |= (1<<i)
            
        best = 0
        # We will recursively build a set.
        # State: mask (available points), current_size
        # We don't need to track "chosen" explicitly if we just remove them from mask?
        # No, because conflicts depend on pairs of chosen points.
        # So we need to know which points are chosen to determine which future points are banned.
        # This state space is too large for simple B&B without clever pruning.
        
        # Alternative: Convert to graph?
        # If we pick i, we don't ban j.
        # If we pick i and j, we ban k if (i,j,k) is a triple.
        
        # Let's use a simple greedy heuristic for the score during search, 
        # and only compute exact score for the top K candidates or when a new "greedy best" is found.
        
        # Greedy Max 3-Free Set:
        # 1. Start with all points available.
        # 2. Repeatedly pick a point with the "least" impact? 
        #    Or pick a point, remove it and all points that would form a triple with it and another picked point?
        #    This is tricky.
        
        # Let's just use a random sampling or a simple local search for the score itself?
        # No, the score is an exact value required by the evaluator? 
        # "The score is the largest subset with no three collinear (exact branch and bound)."
        # The evaluator calculates it. We just need to return the points.
        # We don't need to calculate the score ourselves for the return value, 
        # but we need it to guide our search.
        
        # So, for search guidance, we can use a fast approximation.
        # Approximation: 
        #   1. Greedily construct a 3-free set.
        #   2. The size of this set is a lower bound on the true score.
        #   3. We want to MINIMIZE the true score.
        #   4. If our greedy lower bound is small, the true score is likely small.
        #   5. If our greedy lower bound is large, the true score is definitely large (bad).
        
        # So, minimize the greedy lower bound.
        
        def greedy_lower_bound(pts):
            n = len(pts)
            available = list(range(n))
            chosen = []
            # To make it deterministic and robust, pick points in a specific order?
            # Or random order?
            # Let's try: Pick a point, remove it. 
            # If we pick i, we don't remove others immediately.
            # But if we pick i and j, we must remove k if (i,j,k) is triple.
            
            # Algorithm:
            # While available:
            #   Pick a point p from available.
            #   Add p to chosen.
            #   Remove p from available.
            #   For all other points q in available:
            #       For all other points r in available:
            #           If (p,q,r) is a triple, we can't have both q and r in chosen.
            #           This doesn't mean we remove them immediately.
            
            # This greedy is hard to implement correctly for hypergraphs.
            
            # Let's use a different proxy: Maximize the number of triples.
            # More triples -> likely smaller independence number.
            
            return len(triples) # Just use triple count as proxy for now?
            # Risky. Two configs with same triple count might have different scores.
            
        # Let's use a better proxy: 
        # The size of the minimum vertex cover found by a greedy algorithm.
        # Min VC size = N - Max Independent Set.
        # Greedy VC:
        #   While there are uncovered triples:
        #       Pick a triple with the most uncovered edges?
        #       Pick a vertex in it that covers the most triples.
        #       Add to cover.
        #       Mark all triples containing it as covered.
        
        def greedy_vc_size(pts):
            n = len(pts)
            triples = get_triples(pts)
            if not triples: return 0
            
            # Map vertex to list of triples
            v_to_t = [[] for _ in range(n)]
            for i, t in enumerate(triples):
                for v in t:
                    v_to_t[v].append(i)
                    
            uncovered = set(range(len(triples)))
            cover = 0
            
            while uncovered:
                # Find vertex that covers most uncovered triples
                best_v = -1
                best_count = 0
                for v in range(n):
                    count = 0
                    for t_idx in v_to_t[v]:
                        if t_idx in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                        
                if best_count == 0:
                    break
                    
                cover += 1
                # Remove all triples containing best_v from uncovered
                for t_idx in v_to_t[best_v]:
                    if t_idx in uncovered:
                        uncovered.remove(t_idx)
                        
            return cover

        # Score estimate = N - Greedy VC Size
        # We want to MINIMIZE Score, so MAXIMIZE Greedy VC Size.
        
        # Let's use this as the objective function for local search.
        return n - greedy_vc_size(pts)

    # Generate initial valid config
    def generate_valid_config():
        pts = []
        for _ in range(N):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_pt = [x, y]
                # Check duplicates
                if new_pt in pts:
                    continue
                # Check 4 collinear
                temp_pts = pts + [new_pt]
                if not has_four_collinear(temp_pts):
                    pts.append(new_pt)
                    break
        return pts

    current_points = generate_valid_config()
    current_score = calculate_score(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    # Local Search / Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < deadline:
        # Pick a random point to move
        i = random.randint(0, N-1)
        old_pt = current_points[i]
        
        # Generate new point
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            new_pt = [x, y]
            if new_pt in current_points:
                continue
            # Check if moving i to new_pt creates 4 collinear
            # Remove i, add new_pt
            temp_pts = current_points[:i] + current_points[i+1:] + [new_pt]
            if not has_four_collinear(temp_pts):
                break
        
        # Calculate new score
        new_score = calculate_score(temp_pts)
        
        # Accept if better, or with probability exp(-(new_score - current_score)/T) if worse
        if new_score < current_score:
            current_points = temp_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            delta = new_score - current_score
            if delta == 0:
                # Accept to explore
                if random.random() < 0.5:
                    current_points = temp_pts
                    current_score = new_score
            else:
                # delta > 0
                prob = pow(2.718, -delta / T)
                if random.random() < prob:
                    current_points = temp_pts
                    current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart temperature?
            # Or just keep going
            pass
            
    return best_points
# EVOLVE-BLOCK-END
