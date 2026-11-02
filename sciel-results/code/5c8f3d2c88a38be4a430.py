"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import itertools
from functools import lru_cache

def _generate_weight_vectors(max_weight=4):
    """Generate all nonzero integer vectors in R^D with l1-norm <= max_weight and l2-norm >= 1.
    We focus on vectors with small coordinates to maximize density."""
    vectors = []
    # We want vectors with integer components.
    # To ensure min distance >= max norm, we typically look at configurations where norms are equal or similar.
    # The seed program used weight-2 vectors (two 1s, rest 0). Norm^2 = 2.
    # Let's try to find a dense set of vectors with norm^2 = 4 (e.g., four 1s, or one 2 and two 1s? No, 2^2+1^2+1^2=6. 
    # Vectors with norm^2=4: 
    #   (2, 0, ...) -> 1 vector per position (11*2=22)
    #   (1, 1, 0, ...) -> C(11,2)*4 = 55*4 = 220 vectors
    #   (1, -1, 0, ...) -> same set as above essentially if we allow signs.
    # Actually, the condition is min pairwise distance >= max norm.
    # If all vectors have norm R, then distance >= R means dot product <= R^2/2.
    # For norm^2 = 4, R=2. Distance >= 2 => dot product <= 2.
    # Let's consider the set of all vectors with components in {-1, 0, 1} with exactly two non-zero entries.
    # This is the seed. Count = C(11,2) * 4 = 220.
    
    # Can we add more?
    # What if we use vectors with norm^2 = 4 but different structures?
    # E.g., (2, 0, ..., 0). 
    # Distance between (2,0,..) and (1,1,0,..): 
    # If disjoint: dist^2 = 4+1+1=6 >= 4. OK.
    # If share index: e.g., (2,0,..) and (1,1,0,..) where first is 2 and second has 1 at same spot.
    # v1 = (2, 0, ...), v2 = (1, 1, 0, ...). 
    # dist^2 = (2-1)^2 + (0-1)^2 = 1+1=2 < 4. FAIL.
    # So we cannot mix (2,0..) with (1,1..) if they overlap.
    # If we take all (1,1,0..) vectors, can we add any (2,0..)?
    # No, because for any (2,0..), there are (1,1..) vectors overlapping it.
    # So we must choose a subset.
    
    # Let's stick to the weight-2 (two non-zeros) family first, but maybe optimize signs?
    # The seed takes ALL sign combinations.
    # v = (1, 1, 0..), w = (1, -1, 0..) (same positions).
    # dist^2 = 0^2 + 2^2 = 4. Norm^2 = 2.
    # Condition: dist >= norm => 4 >= 2. True.
    # v = (1, 1, 0..), w = (-1, -1, 0..).
    # dist^2 = 2^2 + 2^2 = 8. True.
    # v = (1, 1, 0..), u = (1, 0, 1, 0..) (disjoint).
    # dist^2 = 0 + 1 + 1 + 0 = 2. Norm^2 = 2.
    # Condition: dist >= norm => 2 >= 2. True.
    # So the seed configuration is valid.
    
    # Can we add weight-3 vectors?
    # Norm^2 = 3.
    # Distance to weight-2 vector (norm^2=2):
    # If disjoint: dist^2 = 3+2=5 >= 4 (max norm squared is 4 if we keep max norm as 2? 
    # Wait, if we add a vector of norm sqrt(3), max norm becomes sqrt(3) ~ 1.732.
    # The condition is: min pairwise distance >= MAX NORM.
    # If we mix norms, the threshold is the global max norm.
    # If we introduce a vector of norm R_max, all pairwise distances must be >= R_max.
    # The seed has R_max = sqrt(2).
    # If we add a vector of norm sqrt(3), R_max = sqrt(3).
    # Then distance between two seed vectors (norm sqrt(2)) must be >= sqrt(3).
    # Min distance in seed is sqrt(2) (for disjoint support).
    # sqrt(2) < sqrt(3). So we cannot simply add higher norm vectors to the existing set.
    # We would need to prune the seed set to ensure all distances >= sqrt(3).
    # Or use a uniform norm.
    
    # Let's try to find a large set of vectors with uniform norm^2 = 4.
    # Candidates:
    # 1. Weight 2 with values 1,1 or 1,-1. (220 vectors). Norm^2=2.
    #    If we scale them by sqrt(2)? No, must be integer.
    #    So we can't have norm^2=4 with weight 2 integers unless we use 2,0.
    # 2. Weight 1 with value 2. (22 vectors). Norm^2=4.
    # 3. Weight 2 with values 1,1? Norm^2=2.
    # 4. Weight 4 with values 1,1,1,1? Norm^2=4.
    
    # Let's explore Weight 4 vectors (four 1s, rest 0).
    # Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280.
    # Norm^2 = 4.
    # Distance between two such vectors u, v:
    # dist^2 = ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v.
    # We need dist >= 2 (since norm=2).
    # dist^2 >= 4 => 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
    # u.v is the dot product.
    # u, v are vectors with 4 entries of 1 or -1.
    # Let k be the number of positions where u and v have the same sign (and are non-zero).
    # Let m be the number of positions where u and v have opposite signs (and are non-zero).
    # u.v = k - m.
    # The number of common non-zero positions is k+m.
    # Since each has 4 non-zeros, k+m <= 4.
    # We need k - m <= 2.
    # Since k+m <= 4, max k-m is 4 (if m=0, k=4).
    # If u and v are identical, k=4, m=0, u.v=4. 4 > 2. Conflict.
    # If u and v share 3 positions:
    #   e.g., u=1110, v=1110 (on those positions).
    #   If signs match: k=3, m=0, u.v=3. 3>2. Conflict.
    #   If one sign differs: k=2, m=1, u.v=1. 1<=2. OK.
    # If u and v share 2 positions:
    #   Max u.v = 2. OK.
    # If u and v share 1 position:
    #   Max u.v = 1. OK.
    # If u and v share 0 positions:
    #   u.v = 0. OK.
    
    # So, two weight-4 vectors conflict if:
    # 1. They are identical.
    # 2. They share 3 or more positions AND their dot product > 2.
    #    Share 4: u.v = 4 (identical) or -4 (opposite) or 0? 
    #    If share 4 positions, they have the same support.
    #    u.v = sum of products of signs.
    #    Let signs be s1..s4. u.v = s1t1 + ... + s4t4.
    #    Max is 4 (all same), min -4 (all diff).
    #    Values: 4, 2, 0, -2, -4.
    #    We need u.v <= 2.
    #    So if support is same, we can have u.v = 2, 0, -2, -4.
    #    u.v = 4 is the only conflict (identical).
    #    Wait, if u.v = 2, dist^2 = 8-4=4. dist=2. OK.
    #    So on the same support, we can pick a subset of sign patterns with pairwise dot product <= 2.
    #    This is equivalent to finding a large code in {1,-1}^4 with min distance?
    #    No, max dot product.
    #    Max dot product 2 means min Hamming distance?
    #    u.v = 4 - 2 * (Hamming Distance).
    #    We need 4 - 2d <= 2 => 2d >= 2 => d >= 1.
    #    So any two distinct sign patterns on the same support are valid!
    #    Because if they are distinct, d>=1, so u.v <= 2.
    #    
    # 3. They share 3 positions.
    #    Support intersection size 3.
    #    u.v = (sum of 3 products) + 0.
    #    Let the 3 products be p1, p2, p3 in {1, -1}.
    #    Sum can be 3, 1, -1, -3.
    #    We need sum <= 2.
    #    So sum=3 is a conflict.
    #    Sum=3 means p1=p2=p3=1. i.e., u and v agree on all 3 shared positions.
    #    
    # 4. They share 2 positions.
    #    u.v = sum of 2 products. Max 2. OK.
    #    
    # 5. They share 1 or 0. OK.
    
    # Summary of conflicts for weight-4 vectors:
    # - Identical vectors.
    # - Vectors sharing exactly 3 positions with the same sign on all 3.
    # - Vectors sharing 4 positions: only identical is conflict? 
    #   Wait, if they share 4 positions, they are on the same support.
    #   As derived, any distinct pair is OK.
    #   So only identical vectors conflict on same support.
    
    # So the problem reduces to:
    # Select a subset of the 5280 weight-4 vectors such that:
    # 1. No two are identical. (Trivial, we pick a set)
    # 2. For any two vectors with support intersection size 3, they do NOT have the same sign on those 3 positions.
    
    # This looks like a graph coloring or independent set problem.
    # Let's try a greedy approach or local search to maximize the count.
    # Given the time limit, a simple greedy with random restarts might work.
    # Or we can use the structure.
    
    # Let's generate all weight-4 vectors.
    vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(tuple(v))
            
    # Now we need to find the maximum independent set in the conflict graph.
    # Conflicts:
    # 1. Same vector.
    # 2. Share 3 indices, same signs on those 3.
    
    # Let's build the conflict graph or check on the fly.
    # With 5280 vectors, an O(N^2) check is ~27M operations. Doable in Python?
    # 27M * constant. Might be slow in pure Python loops.
    # Let's use NumPy to speed up conflict detection.
    
    V = np.array(vectors, dtype=np.int8)
    N = len(V)
    
    # Compute pairwise dot products?
    # Dot product matrix D = V @ V.T
    # Conflict if D[i][j] > 2?
    # Let's verify.
    # If share 4: D=4 (identical), 2, 0, -2, -4.
    #   D=4 is conflict.
    #   D=2 is OK.
    # If share 3: D=3, 1, -1, -3.
    #   D=3 is conflict.
    #   D=1 is OK.
    # If share 2: D=2, 0, -2.
    #   D=2 is OK.
    # If share 1: D=1, -1. OK.
    # If share 0: D=0. OK.
    
    # So the condition is simply: D[i][j] <= 2 for all i != j.
    # And we want to select the largest subset satisfying this.
    # This is exactly the Maximum Independent Set on the graph where edge exists if D[i][j] > 2.
    
    # Compute D = V @ V.T
    # This is 5280x5280. 27M floats. ~200MB. OK.
    # Time: 5280^3 / 3 ~ 5e10 ops? No, matrix multiply is faster.
    # V is (5280, 11). V @ V.T is (5280, 11) @ (11, 5280).
    # This is very fast in NumPy.
    
    D_mat = V @ V.T
    
    # Create adjacency matrix for conflicts
    # Conflict if D_mat > 2
    # Note: Diagonal is 4 (norm^2). We ignore diagonal or treat self as non-conflict for IS?
    # In IS, we pick nodes. We don't pick a node twice.
    # So we just need to ensure for i != j, if D[i][j] > 2, we can't pick both.
    
    # Let's use a simple greedy algorithm with random ordering.
    # Or Tomita BB-MCS as suggested?
    # Tomita is for Max Clique. Max Independent Set is Max Clique in complement graph.
    # Complement graph has edge if D[i][j] <= 2.
    # This graph is dense.
    # Actually, conflict graph (edges if D>2) is sparse?
    # Let's check density.
    # Each vector conflicts with:
    # - Itself (1)
    # - Vectors sharing 3 indices with same sign.
    #   For a fixed vector, how many conflict?
    #   Pick 3 of the 4 indices. 4 choices.
    #   The sign on these 3 is fixed.
    #   The 4th index of the new vector can be any of the remaining 7 indices.
    #   The sign on the 4th index of the new vector can be +/- 1.
    #   So for each choice of 3 indices (4 ways), there are 7 * 2 = 14 vectors.
    #   Total conflicts from share-3: 4 * 14 = 56.
    #   Are there overlaps?
    #   If we pick different sets of 3, the supports are different (differ by 1 index).
    #   So they are distinct vectors.
    #   What about share-4?
    #   Only identical is conflict.
    #   So each node has degree ~ 56 (plus self).
    #   Total edges ~ 5280 * 56 / 2 ~ 147k.
    #   Very sparse graph!
    
    # So we are looking for MIS in a sparse graph with 5280 nodes and ~150k edges.
    # Greedy coloring bound + branch and bound should work well.
    # Or simply greedy MIS.
    # Greedy:
    # 1. Order nodes by degree (ascending or descending).
    # 2. Iterate, if node not in conflict with already selected, select it.
    
    # Let's try a simple greedy first to get a baseline.
    # Then maybe local search.
    
    # To be safe and fast, let's implement a greedy MIS.
    # Degree calculation:
    # deg[i] = sum(D_mat[i] > 2) - 1 (exclude self)
    
    # Actually, we can just do:
    # selected = []
    # available = set(range(N))
    # while available:
    #   pick node with min degree in induced subgraph? (Expensive)
    #   Or just random/greedy.
    
    # Given 150s limit, we can afford a decent B&B if implemented in C++ but we are in Python.
    # Python B&B on 5280 nodes might be slow if not careful.
    # However, the graph is very structured.
    
    # Let's try a simple greedy with multiple random restarts.
    # Time limit: 150s.
    
    best_size = 0
    best_set = []
    
    deadline = time.time() + 140
    
    # Precompute adjacency lists for faster conflict checking
    # adj[i] = set of j such that D[i][j] > 2
    # Building this set for 5280 nodes:
    # 5280 * 5280 checks = 27M.
    # In Python loop, 27M iterations is ~1-2 seconds. Acceptable.
    
    adj = [set() for _ in range(N)]
    for i in range(N):
        row = D_mat[i]
        # Find indices where row > 2
        # Using NumPy to find indices
        idxs = np.where(row > 2)[0]
        for j in idxs:
            if j != i:
                adj[i].add(j)
                
    # Now we have adjacency lists.
    # Greedy MIS:
    # 1. Get degrees.
    # 2. Sort nodes by degree.
    # 3. Iterate.
    
    # To improve, we can use a "conflict-free" greedy:
    # Maintain a set of available nodes.
    # Pick a node, add to solution, remove it and its neighbors from available.
    
    # Random restarts:
    # Shuffle node order, then greedy.
    
    # Better: Degree-based greedy.
    # Let's try one run with static degree ordering first.
    
    degrees = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    selected = []
    in_set = [False] * N
    
    for i in order:
        if in_set[i]:
            continue
        # Check if i conflicts with any already selected?
        # No, if we remove neighbors when we select, we don't need to check.
        # But we are iterating in a fixed order.
        # We need to check if i is still "available".
        # Let's maintain a boolean array 'available'.
        pass
        
    # Let's implement the removal-based greedy.
    # This is O(N + E).
    
    def greedy_mis(order):
        available = [True] * N
        selected = []
        for i in order:
            if available[i]:
                selected.append(i)
                available[i] = False
                for j in adj[i]:
                    available[j] = False
        return selected
        
    # Try static degree order
    sel = greedy_mis(order)
    best_size = len(sel)
    best_set = sel
    
    # Try random restarts
    # We have ~130s left.
    # Each greedy run is O(N+E) ~ 150k ops. Very fast (<1ms).
    # We can do thousands of restarts.
    
    import random
    random.seed(42)
    
    # Also try different orderings:
    # 1. Degree ascending
    # 2. Degree descending
    # 3. Random
    # 4. Weighted by something else?
    
    # Let's just do random shuffles.
    start_time = time.time()
    iterations = 0
    
    while time.time() < deadline:
        # Random shuffle
        random_order = list(range(N))
        random.shuffle(random_order)
        sel = greedy_mis(random_order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel
            
        # Also try degree-based with random tie-breaking?
        # Or just random is enough for now.
        
        iterations += 1
        if iterations % 1000 == 0:
            # Check time
            pass

    # Convert best_set indices to vectors
    if best_set:
        result = V[best_set]
    else:
        # Fallback to seed-like behavior if something went wrong
        # But best_size should be > 0
        result = V[:1]
        
    return result.astype(np.int64)

def construct():
    return _construct_impl()

def _construct_impl():
    # Move logic here to keep construct clean?
    # No, just put logic in construct or call helper.
    # The prompt asks to keep function names.
    # I'll put the logic in construct.
    pass

# Let's refactor to put logic in construct directly.

def construct():
    import time
    import itertools
    import numpy as np
    import random
    
    D = 11
    deadline = time.time() + 140
    
    # Generate weight-4 vectors
    vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(tuple(v))
            
    V = np.array(vectors, dtype=np.int8)
    N = len(V)
    
    # Compute conflict graph
    D_mat = V @ V.T
    adj = [set() for _ in range(N)]
    for i in range(N):
        idxs = np.where(D_mat[i] > 2)[0]
        for j in idxs:
            if j != i:
                adj[i].add(j)
                
    degrees = [len(adj[i]) for i in range(N)]
    
    def greedy_mis(order):
        available = [True] * N
        selected = []
        for i in order:
            if available[i]:
                selected.append(i)
                available[i] = False
                for j in adj[i]:
                    available[j] = False
        return selected
        
    # Initial greedy with degree order
    order = sorted(range(N), key=lambda x: degrees[x])
    best_set = greedy_mis(order)
    best_size = len(best_set)
    
    # Random restarts
    random.seed(42)
    while time.time() < deadline:
        random_order = list(range(N))
        random.shuffle(random_order)
        sel = greedy_mis(random_order)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = sel
            
    result = V[best_set]
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
