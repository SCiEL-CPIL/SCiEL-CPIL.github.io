import random
import time
import itertools
from collections import defaultdict

def construct():
    """
    Constructs 20 distinct integer points with |x|, |y| <= 100 and no four collinear.
    Minimizes the size of the largest subset with no three collinear.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Grid of candidate points
    # Using a smaller grid to speed up line checks
    grid_range = 15 
    candidates = [(x, y) for x in range(-grid_range, grid_range + 1) for y in range(-grid_range, grid_range + 1)]
    
    # Helper to check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1)

    # Helper to check if a set of points has no 4 collinear
    def is_valid_no_4(points):
        # Group points by lines
        # For each pair, determine line, add to line map
        # If any line has > 3 points, invalid
        # Optimization: Only check lines formed by pairs in the set
        # But we need to ensure NO line in the plane has 4.
        # Since points are in the set, any line with 4 points must contain at least one pair from the set.
        line_counts = defaultdict(int)
        # To avoid double counting and ensure we catch all 4s:
        # For each pair (i, j), calculate line key.
        # Increment count.
        # If count becomes 4, invalid.
        # However, a line with 4 points A,B,C,D has pairs (A,B), (A,C)...
        # If we just count pairs, a line with k points has k*(k-1)/2 pairs.
        # We can't just count pairs.
        
        # Better: For each point i, for each pair j,k in points:
        # This is O(N^3). N=20, 8000 ops. Very fast.
        n = len(points)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for k in range(n):
                    if k == i or k == j: continue
                    if are_collinear(points[i], points[j], points[k]):
                        # Check if there is a 4th point
                        # We can just count collinear points for each line
                        pass
        
        # Simpler validity check:
        # For each subset of 4 points, check if collinear.
        # C(20, 4) = 4845. Very fast.
        for combo in itertools.combinations(points, 4):
            if are_collinear(combo[0], combo[1], combo[2]) and are_collinear(combo[0], combo[1], combo[3]):
                return False
        return True

    def get_triples(points):
        """Get all collinear triples in the set."""
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples

    def calc_score(points):
        """
        Calculate the size of the largest subset with no three collinear.
        This is the independence number of the hypergraph of triples.
        """
        triples = get_triples(points)
        n = len(points)
        if not triples:
            return n
            
        # Build adjacency/bitmask for B&B
        # For each vertex, store a bitmask of vertices it conflicts with?
        # Conflict if they are in a triple together?
        # No, independent set in hypergraph: no edge is fully contained.
        # This is NOT the same as graph independent set.
        # However, we can use a greedy heuristic for fast evaluation, 
        # and B&B only for final candidates or small sets.
        
        # Greedy Heuristic:
        # Sort vertices by degree (number of triples they are in).
        # Iteratively pick the vertex with lowest degree, remove it and all triples containing it.
        # This gives a lower bound on alpha? No, it gives an upper bound on the max independent set?
        # Actually, picking low degree vertices is a heuristic for finding a LARGE independent set.
        
        # Let's use a simple B&B for n=20.
        # State: (current_set, remaining_candidates)
        # But 2^20 is 1M. Feasible.
        
        # Precompute for each vertex, the set of triples it belongs to
        vertex_triples = defaultdict(list)
        for t in triples:
            for v in t:
                vertex_triples[v].append(t)
                
        best_alpha = 0
        # Sort vertices by degree descending for better pruning?
        # Or just use a recursive search.
        
        # To speed up, we can use a bitmask representation.
        # 20 points -> 20 bits.
        
        # Precompute a list of triples as bitmasks
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
            
        # B&B
        # current_mask: points selected so far
        # available_mask: points that can still be selected
        # If we select a point, we must ensure no triple is formed.
        # This is hard to enforce incrementally in a simple mask.
        
        # Alternative: Enumerate all subsets? 2^20 = 1,048,576.
        # For each subset, check if it contains any triple.
        # If not, it's an independent set. Find max size.
        # This is O(2^N * E). E is number of triples.
        # Max triples for 20 points no 4 collinear:
        # Each pair is in at most 1 triple? No, a pair is in a line. If line has 3, it's 1 triple.
        # If line has 4, invalid.
        # So each pair is in at most 1 triple.
        # Max pairs C(20,2) = 190. So max triples 190/3 ~ 63.
        # 1M * 63 checks is ~60M ops. Might be slow in Python if done naively.
        
        # Optimized check:
        # Precompute for each triple a bitmask.
        # For a given subset mask, check if (subset_mask & triple_mask) == triple_mask for any triple.
        
        # We can iterate through subsets in order of size (descending) to find the max.
        # Or just iterate all and keep max.
        
        # Let's try to optimize the inner loop.
        # If we have many triples, the check is fast if we break early.
        
        # Given the time limit, we might want a heuristic score for search, 
        # and exact score only for the final best few.
        
        # Heuristic Score:
        # Count the number of points that are NOT in any triple? No.
        # A good heuristic for independence number is the size of the set minus the number of "forced" exclusions?
        
        # Let's use a simple greedy independent set construction for the score estimation.
        # Greedy:
        # 1. Start with all points available.
        # 2. While available is not empty:
        #    a. Pick a point p with the minimum number of triples it forms with other available points.
        #    b. Add p to independent set.
        #    c. Remove p and all points that form a triple with p and another point in the independent set? 
        #       No, that's for graph.
        #       For hypergraph: If we pick p, we cannot pick any pair (a,b) such that (p,a,b) is a triple.
        #       This is complex.
        
        # Let's stick to exact B&B but optimized.
        # Since N=20, we can use Python's `itertools` or bit manipulation.
        
        # Actually, a simpler exact method:
        # The complement of the independent set is a vertex cover of the hypergraph.
        # Min vertex cover in 3-uniform hypergraph is NP-hard.
        
        # Let's use a random greedy for the score in the search loop, 
        # and exact check for the final answer.
        
        # Random Greedy Independent Set:
        # Shuffle points.
        # Iterate, if adding point doesn't create a triple with already added points, add it.
        # Run 100 times, take max.
        
        def greedy_independent_set(pts, triples_set):
            # triples_set: set of frozensets of indices
            best_size = 0
            for _ in range(50): # 50 restarts
                random.shuffle(pts)
                selected = []
                selected_set = set()
                for p in pts:
                    idx = p[2] # assuming p is (x,y,idx)
                    conflict = False
                    # Check if any triple contains idx and two points in selected_set
                    for t in triples_set:
                        if idx in t:
                            others = [x for x in t if x != idx]
                            if all(o in selected_set for o in others):
                                conflict = True
                                break
                    if not conflict:
                        selected.append(p)
                        selected_set.add(idx)
                best_size = max(best_size, len(selected))
            return best_size

        # Prepare data for greedy
        pts_with_idx = [(p[0], p[1], i) for i, p in enumerate(points)]
        triples_set = set(frozenset(t) for t in triples)
        
        return greedy_independent_set(pts_with_idx, triples_set)

    def exact_independent_set(points):
        """Exact B&B for max independent set size."""
        triples = get_triples(points)
        n = len(points)
        if not triples:
            return n
            
        # Bitmask approach
        # triple_masks: list of int
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
            
        # Check if a mask contains any triple
        def has_triple(mask):
            for tm in triple_masks:
                if (mask & tm) == tm:
                    return True
            return False
            
        best = 0
        # Iterate all subsets? 2^20 is 1M.
        # We can prune if popcount(mask) + popcount(~mask) <= best.
        # But checking all 1M masks might be slow in Python if has_triple is slow.
        
        # Optimization: 
        # Generate subsets in decreasing order of size?
        # Or just iterate 0 to 2^N - 1.
        
        # Let's try a recursive B&B on vertices.
        # Order vertices by degree.
        deg = [0]*n
        for t in triples:
            for v in t:
                deg[v] += 1
        order = sorted(range(n), key=lambda i: deg[i], reverse=True)
        
        # Map original indices to order indices? 
        # It's easier to just use the original indices and a visited/available mask.
        
        def search(current_mask, avail_mask, current_size):
            nonlocal best
            if current_size + bin(avail_mask).count('1') <= best:
                return
            
            if avail_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick a vertex from avail_mask
            # Heuristic: pick vertex with highest degree in the induced subgraph?
            # For simplicity, pick the first available vertex in 'order'
            v = None
            for u in order:
                if avail_mask & (1 << u):
                    v = u
                    break
            
            if v is None:
                return
                
            # Option 1: Exclude v
            search(current_mask, avail_mask & ~(1 << v), current_size)
            
            # Option 2: Include v
            # If we include v, we must exclude all pairs (a,b) that form a triple with v.
            # So, new_avail = avail_mask with v removed, AND any point a such that there exists b in current_mask 
            # such that (v,a,b) is a triple?
            # No, if we include v, we cannot include ANY pair (a,b) from the remaining set 
            # such that (v,a,b) is a triple.
            # This means if (v,a,b) is a triple, we cannot select both a and b.
            # This is a constraint on the remaining set, not just removal.
            # This makes the state space larger (needs to track forbidden pairs).
            
            # Given the complexity, the 2^N enumeration with bitmasks is safer.
            # 1M iterations * fast bit check.
            # Let's implement the 2^N check.
            
        # 2^N check
        for mask in range(1 << n):
            # Prune: if bits set in mask <= best, skip
            if bin(mask).count('1') <= best:
                continue
            if has_triple(mask):
                continue
            best = bin(mask).count('1')
            
        return best

    # Main Search
    best_points = None
    best_score = 20
    
    # Initial random valid set
    def generate_valid_set():
        pts = []
        # Greedily add points
        # To ensure speed, we just pick random and check validity of the whole set at the end?
        # No, we need no 4 collinear.
        # Let's pick random 20, check validity. If invalid, retry.
        # Probability of no 4 collinear in random grid is high for sparse grids.
        while True:
            candidate = random.sample(candidates, 20)
            if is_valid_no_4(candidate):
                return candidate

    current_set = generate_valid_set()
    current_score = calc_score(current_set)
    if current_score < best_score:
        best_score = current_score
        best_points = current_set.copy()
        
    print(f"Initial score: {current_score}", flush=True)
    
    # Local Search
    # Swap one point with another
    # Candidate pool: all grid points
    
    while time.time() < deadline:
        # Pick a random point to remove
        idx_to_remove = random.randint(0, 19)
        removed_point = current_set[idx_to_remove]
        
        # Try to replace with a random candidate
        # To be more directed, pick a candidate that forms triples with remaining points
        remaining = [p for i, p in enumerate(current_set) if i != idx_to_remove]
        
        # Score candidates
        best_candidate = None
        best_candidate_gain = 0
        
        # Sample some candidates
        for _ in range(50):
            cand = random.choice(candidates)
            if cand in current_set:
                continue
                
            new_set = remaining + [cand]
            if not is_valid_no_4(new_set):
                continue
                
            # Quick score estimate
            # Count triples
            triples = get_triples(new_set)
            # More triples usually means lower independence number
            # But not always.
            # Let's just use the greedy score for comparison
            score = calc_score(new_set)
            
            if score < current_score:
                best_candidate = cand
                best_candidate_gain = current_score - score
                current_score = score # Update for comparison? No, keep original
                break # Found improvement, take it
        
        if best_candidate:
            new_set = [cand if i == idx_to_remove else p for i, p in enumerate(current_set)]
            # Actually, we need to replace the point
            new_set = remaining + [best_candidate]
            current_set = new_set
            current_score = calc_score(current_set)
            
            if current_score < best_score:
                best_score = current_score
                best_points = current_set.copy()
                print(f"New best score: {best_score}", flush=True)
                
    # Final Exact Verification
    if best_points:
        final_score = exact_independent_set(best_points)
        print(f"Final exact score: {final_score}", flush=True)
        # Return the points as lists of lists
        return [[p[0], p[1]] for p in best_points]
    else:
        # Fallback to seed
        return [[x, x * x] for x in range(-10, 10)]

# EVOLVE-BLOCK-END
