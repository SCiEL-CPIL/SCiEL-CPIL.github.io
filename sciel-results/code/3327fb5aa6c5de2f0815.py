import time
import numpy as np
import random
from itertools import combinations
import sys

D = 11

# EVOLVE-BLOCK-START

def _generate_supports(max_weight=3):
    """
    Generate all possible supports (sets of indices) for integer vectors 
    with small L1 norm (weight <= max_weight).
    We represent a vector by its support (indices where non-zero) and 
    we handle signs separately or assume we can pick signs to avoid conflicts.
    
    Actually, for the MIS approach on supports:
    Two vectors conflict if they are "too close".
    If we fix the supports, we can choose signs to maximize distance.
    However, the constraint is min pairwise distance >= max norm.
    
    Let's refine the model:
    We want to select a set of vectors V.
    Constraint: For all u, v in V, ||u - v|| >= max_{w in V} ||w||.
    
    If we restrict to vectors with the SAME norm, say R, then the constraint becomes:
    ||u - v|| >= R for all u != v.
    Squaring: ||u||^2 + ||v||^2 - 2 u·v >= R^2
    2R^2 - 2 u·v >= R^2  =>  R^2 >= 2 u·v  =>  u·v <= R^2 / 2.
    
    If we use vectors with integer coordinates, norms are sqrt(integer).
    The "max norm" in the set determines the threshold.
    To maximize count, we should probably stick to a single shell (same norm) 
    or a mix where the largest norm dictates the constraint for all pairs.
    
    Strategy: 
    1. Consider vectors of a specific norm R.
    2. Build a conflict graph where nodes are vectors.
    3. Edge exists if ||u - v|| < R.
    4. Find MIS.
    
    But the number of integer vectors of norm R in D=11 can be large.
    For R^2 = 1 (weight 1): 22 vectors.
    For R^2 = 2 (weight 2): 2 * C(11,2) * 2^2 = 2 * 55 * 4 = 440? No.
    Vectors with two 1s and rest 0s: C(11,2) * 2^2 = 55 * 4 = 220.
    Vectors with one 1 and one -1? No, norm is still sqrt(2).
    Wait, vector (1,1,0..) has norm sqrt(2). (1,-1,0..) has norm sqrt(2).
    Total vectors of norm sqrt(2) in Z^11:
    Choose 2 indices: C(11,2) = 55.
    Signs: 2^2 = 4.
    Total = 220.
    
    For R^2 = 3 (weight 3):
    Case 1: Three 1s. C(11,3) * 2^3 = 165 * 8 = 1320.
    Case 2: One 1, two 0s? No, sum of squares = 3.
    Possible integer squares summing to 3: 1+1+1.
    So only weight 3 vectors with entries +/-1.
    Total = 1320.
    
    For R^2 = 4:
    2^2: One entry 2, rest 0. 11 * 2 = 22.
    1^2+1^2+1^2+1^2: Four 1s. C(11,4) * 2^4 = 330 * 16 = 5280.
    
    The seed program uses vectors of norm 1 (22) and norm sqrt(2) (220).
    Wait, the seed `_generate_seed_vectors` creates vectors with two non-zero entries of 1 or -1.
    Norm is sqrt(2).
    It does NOT include unit vectors (norm 1) in the seed generation, but `_valid` checks them.
    Actually, `_generate_seed_vectors` returns vectors of norm sqrt(2).
    If we add unit vectors (norm 1), the max norm becomes sqrt(2).
    Constraint: distance between unit vector e_i and sqrt(2) vector v_jk >= sqrt(2).
    If jk doesn't include i, dist^2 = 1 + 2 = 3 >= 2. OK.
    If jk includes i, say v=(1,1,0..), e=(1,0,0..). dist^2 = 0^2 + 1^2 + ... = 1.
    1 < 2. Conflict!
    So we cannot mix norm 1 and norm sqrt(2) easily if they share indices.
    
    To get a high score, we should likely focus on a single shell (same norm) 
    to simplify the constraint to u·v <= R^2/2.
    
    Let's try R^2 = 2 (norm sqrt(2)).
    Vectors: 220 candidates.
    Constraint: ||u-v||^2 >= 2.
    ||u||^2 + ||v||^2 - 2u·v >= 2 => 2 + 2 - 2u·v >= 2 => 2 >= 2u·v => u·v <= 1.
    
    So we need to find the largest set of vectors from the 220 candidates 
    such that for any two, their dot product is <= 1.
    
    This is an Independent Set problem in a graph where nodes are the 220 vectors,
    and an edge connects u, v if u·v > 1 (i.e., u·v >= 2).
    Since u, v are integer vectors with entries in {-1, 0, 1} and weight 2,
    u·v is an integer.
    Max possible dot product:
    If u = (1, 1, 0..), v = (1, 1, 0..), dot = 2. Conflict.
    If u = (1, 1, 0..), v = (1, -1, 0..), dot = 0. OK.
    If u = (1, 1, 0..), v = (1, 0, 1..), dot = 1. OK.
    If u = (1, 1, 0..), v = (0, 0, 1..)? No, v must have weight 2.
    
    So conflicts occur when:
    1. u = v (dot 2). But we select distinct vectors.
    2. u = -v? Dot = -2. OK.
    3. u and v share 2 indices with same signs. E.g., (1,1) and (1,1).
    4. u and v share 2 indices with opposite signs? (1,1) and (-1,-1) -> dot -2.
    5. u and v share 1 index? Dot is +/-1. OK.
    6. u and v share 0 indices? Dot 0. OK.
    
    So the ONLY conflicts are between vectors that are identical?
    Wait.
    u = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    v = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) -> Identical. Not in set.
    
    What about u = (1, 1, 0..) and w = (1, 1, 0..)? They are the same vector.
    Are there distinct vectors with dot product 2?
    u = (1, 1, 0..)
    v = (1, 1, 0..)
    If u != v, can they have dot 2?
    Max dot product for distinct weight-2 vectors:
    If they share both indices, they must be identical to have dot 2.
    If they share one index, dot is +/-1.
    If they share none, dot is 0.
    
    Therefore, for R^2 = 2, ALL distinct vectors are compatible!
    u·v <= 1 is always true for distinct u, v of weight 2.
    Let's verify.
    u = (1, 1, 0..)
    v = (1, -1, 0..) -> dot 0.
    v = (-1, 1, 0..) -> dot 0.
    v = (-1, -1, 0..) -> dot -2.
    v = (1, 0, 1..) -> dot 1.
    
    So we can take ALL 220 vectors of norm sqrt(2)?
    Let's check the constraint: min pairwise distance >= max norm.
    Max norm = sqrt(2).
    Min pairwise distance between distinct vectors:
    Closest pair: (1,1,0..) and (1,0,1..).
    Diff: (0, 1, -1, 0..). Norm squared = 2. Norm = sqrt(2).
    So distance = sqrt(2).
    Condition: sqrt(2) >= sqrt(2). True.
    
    So we can take all 220 vectors of weight 2?
    Score = 220.
    The seed program score was 1.0, which corresponds to 55*4 = 220 vectors?
    Let's check the seed code.
    `for i in range(D): for j in range(i+1, D): for si in (1,-1): for sj in (1,-1):`
    This generates exactly the 220 vectors.
    So the seed program already finds 220.
    Score 1.0 means the baseline is 220.
    
    Can we do better?
    Try R^2 = 1 (norm 1).
    Vectors: 22 (e_i, -e_i).
    Constraint: dist >= 1.
    Min dist between e_i and e_j (i!=j) is sqrt(2) >= 1.
    Min dist between e_i and -e_i is 2 >= 1.
    So we can take all 22.
    But 22 < 220.
    
    Try R^2 = 3 (norm sqrt(3)).
    Vectors: 1320 candidates.
    Constraint: ||u-v||^2 >= 3.
    3 + 3 - 2u·v >= 3 => 3 >= 2u·v => u·v <= 1.5 => u·v <= 1 (integer).
    
    So we need u·v <= 1 for all distinct u, v.
    Conflicts if u·v >= 2.
    When is u·v >= 2 for weight-3 vectors?
    u = (1,1,1, 0..)
    v = (1,1,1, 0..) -> dot 3. Conflict (identical).
    v = (1,1,-1, 0..) -> dot 1. OK.
    v = (1,1,0, 1,0..) -> dot 2. Conflict!
    
    So we have conflicts. We need to find the MIS of the graph of 1320 nodes.
    This is a much larger graph.
    
    Can we mix shells?
    If we use R^2=2 vectors, max norm is sqrt(2).
    If we add R^2=1 vectors, max norm becomes sqrt(2).
    We already established that R^2=1 and R^2=2 vectors conflict if they share indices.
    But we can select a subset of R^2=1 vectors that don't conflict with the R^2=2 set.
    However, the R^2=2 set is "dense".
    
    Let's focus on R^2=3.
    Number of candidates: 1320.
    We need to select max subset with pairwise dot <= 1.
    
    Alternative: R^2=4.
    Vectors of norm 2.
    Types:
    A: One entry 2. (11 * 2 = 22 vectors).
    B: Four entries 1. (C(11,4) * 2^4 = 330 * 16 = 5280 vectors).
    Total 5302.
    Constraint: ||u-v||^2 >= 4.
    4 + 4 - 2u·v >= 4 => 4 >= 2u·v => u·v <= 2.
    
    This graph is huge (5302 nodes).
    
    The best strategy is likely to solve the MIS for R^2=3 (1320 nodes) or R^2=2 (220 nodes, trivial).
    Since R^2=2 gives 220, and R^2=3 has 1320 candidates, even a small fraction could beat 220.
    For example, if we can pick 1/3 of them, that's 440.
    
    Let's implement a greedy MIS solver for the R^2=3 case.
    Nodes: 1320 vectors of weight 3 (entries +/-1).
    Edge: u·v >= 2.
    
    We can optimize the dot product calculation.
    Represent vector by its support (3 indices) and signs.
    u = (s1, s2, s3) at (i, j, k).
    v = (t1, t2, t3) at (p, q, r).
    u·v = sum of s_a * t_b for common indices.
    
    We will:
    1. Generate all 1320 vectors.
    2. Compute pairwise dot products.
    3. Build conflict graph (adjacency matrix or list).
    4. Run a greedy algorithm to find a large independent set.
       - Order nodes by degree (ascending).
       - Pick node, remove it and its neighbors.
    
    To speed up, we can use bitsets or numpy.
    1320x1320 matrix is ~1.7M entries. Fits in memory easily.
    
    Let's try this.
    
    Also, consider R^2=1 and R^2=2 mixed?
    If we take all 220 R^2=2 vectors, can we add any R^2=1 vectors?
    Max norm = sqrt(2).
    R^2=1 vector e_i.
    Conflict with R^2=2 vector v_jk if dist(e_i, v_jk) < sqrt(2).
    dist^2 = 1 + 2 - 2 e_i·v_jk = 3 - 2 e_i·v_jk.
    Need 3 - 2 e_i·v_jk >= 2 => 1 >= 2 e_i·v_jk.
    e_i·v_jk is 1 if i in support and sign matches, -1 if sign opposite, 0 if not in support.
    If e_i·v_jk = 1, dist^2 = 1 < 2. Conflict.
    If e_i·v_jk = -1, dist^2 = 5 >= 2. OK.
    If e_i·v_jk = 0, dist^2 = 3 >= 2. OK.
    
    So e_i conflicts with v_jk if i is in support of v_jk AND sign of v_jk at i is +1 (same as e_i).
    To add e_i, we must remove all v_jk in our current set that have +1 at index i.
    Since we have all 220 R^2=2 vectors, for any i, there are C(10,1) * 2 * 2 = 40 vectors with +1 at i?
    Support includes i. Other index j (10 choices). Sign at i is +1. Sign at j is +/-1 (2 choices).
    So 10 * 2 = 20 vectors conflict with e_i.
    We would remove 20 vectors to add 1. Net loss.
    So mixing is not beneficial for this specific setup.
    
    Focus on R^2=3.
    
    """
    vectors = []
    supports = []
    signs = []
    
    # Generate all weight 3 vectors
    for combo in combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[combo[0]] = s1
                    v[combo[1]] = s2
                    v[combo[2]] = s3
                    vectors.append(v)
                    supports.append(combo)
                    signs.append((s1, s2, s3))
                    
    n = len(vectors)
    # n = 1320
    
    # Compute dot products
    # V is (n, D)
    V = np.array(vectors, dtype=np.int8)
    Dots = V @ V.T
    
    # Conflict if Dots[i,j] >= 2
    # Note: Dots[i,i] = 3. We ignore self.
    # We want to find Independent Set.
    
    # Greedy MIS
    # 1. Compute degrees
    # 2. Sort by degree
    # 3. Iterate
    
    # To make it faster, we can use a bitset representation for the graph.
    # But 1320 bits is ~166 bytes. We can use Python integers or numpy arrays.
    
    # Let's use a list of sets for neighbors.
    # Or better, a numpy boolean matrix.
    # Conflict matrix C: C[i,j] = True if Dots[i,j] >= 2.
    C = Dots >= 2
    np.fill_diagonal(C, False) # Self is not a conflict in the graph sense, we don't pick same node twice
    
    # Degree
    degrees = C.sum(axis=1)
    
    # Order by degree
    order = np.argsort(degrees)
    
    selected = []
    available = np.ones(n, dtype=bool)
    
    # Greedy: pick node with min degree among available
    # This is O(N^2) if we re-scan.
    # N=1320, N^2 = 1.7M. Very fast.
    
    remaining = list(order)
    # We need to be careful with removal.
    
    # Better: 
    # while remaining:
    #   pick u with min degree in induced subgraph
    #   add u to set
    #   remove u and neighbors from remaining
    
    # To get min degree in induced subgraph, we can just check degrees of remaining nodes.
    # But degrees change as we remove neighbors.
    
    # Simple approach:
    # 1. Start with all nodes.
    # 2. Pick node u with smallest current degree.
    # 3. Add u.
    # 4. Remove u and all neighbors of u from the pool.
    
    pool = set(range(n))
    # We need to track current degrees.
    # Initial degrees are known.
    # When we remove a node v, we must decrement the degree of its neighbors.
    
    # This is standard greedy MIS.
    
    # Optimization: 
    # If we just pick all nodes with degree 0 first?
    # Then update.
    
    # Let's implement the standard greedy.
    
    # For speed, use numpy for neighbor removal.
    # C is boolean matrix.
    
    selected_indices = []
    
    # We'll maintain a boolean array `in_pool`
    in_pool = np.ones(n, dtype=bool)
    
    # We need to efficiently find the node with min degree in the pool.
    # And efficiently remove neighbors.
    
    # Since N is small (1320), we can just loop.
    
    while in_pool.any():
        # Find min degree among in_pool
        # degrees is updated? 
        # Let's maintain a list of current degrees.
        # Actually, we can just recompute or update.
        
        # To avoid O(N^2) inside the loop, we can use a heap.
        # But N is small.
        
        # Find index of min degree in pool
        # Mask degrees
        masked_degrees = np.where(in_pool, degrees, np.inf)
        u = np.argmin(masked_degrees)
        
        if not in_pool[u]:
            break
            
        selected_indices.append(u)
        
        # Remove u and neighbors
        # Neighbors are C[u] == True
        to_remove = C[u] | (np.arange(n) == u)
        in_pool[to_remove] = False
        
        # Update degrees of remaining nodes?
        # For the next iteration, we need correct degrees.
        # Degree of a node v in the pool is sum(C[v, w] for w in pool).
        # This is expensive to update one by one.
        # But we only need the MIN degree.
        # We can just recompute the masked degrees? 
        # Or update degrees:
        # For each v in pool, if C[v, u] was True, degree[v] -= 1.
        # Also, neighbors of u are removed, so they don't affect degrees of remaining nodes 
        # except for the edges between them? No, edges to removed nodes are gone.
        
        # Let's update degrees properly.
        # For all v in pool:
        #   if C[v, u]: degrees[v] -= 1
        
        # We can do this with numpy.
        # neighbors_of_u = C[u]
        # But we only care about v that are still in pool.
        # Actually, C is symmetric.
        # The neighbors of u that are in pool are C[u] & in_pool (before removal).
        # Let's capture them before removing.
        
        # Wait, I already set in_pool[to_remove] = False.
        # I need to update degrees of the REMAINING nodes.
        # The remaining nodes are those not in to_remove.
        # Their degree decreases by 1 if they were connected to u.
        
        # Let's do it before setting in_pool to False?
        # No, C[u] is fixed.
        
        # Let's restart the loop logic to be safe.
        
        pass # Placeholder, will rewrite in construct

def _generate_supports_and_vectors():
    vectors = []
    for combo in combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[combo[0]] = s1
                    v[combo[1]] = s2
                    v[combo[2]] = s3
                    vectors.append(v)
    return np.array(vectors, dtype=np.int8)

def _greedy_mis(V):
    n = len(V)
    Dots = V @ V.T
    C = Dots >= 2
    np.fill_diagonal(C, False)
    
    degrees = C.sum(axis=1).copy()
    in_pool = np.ones(n, dtype=bool)
    
    selected = []
    
    # Use a heap for min degree?
    # Or just scan. N=1320. Scan is 1320 * 1320 = 1.7M ops. Fast.
    
    while in_pool.any():
        # Find min degree in pool
        masked_degrees = np.where(in_pool, degrees, np.inf)
        u = np.argmin(masked_degrees)
        
        if not in_pool[u]:
            break
            
        selected.append(u)
        
        # Identify neighbors to remove
        # Neighbors of u are C[u]
        # We remove u and its neighbors
        to_remove_mask = C[u].copy()
        to_remove_mask[u] = True
        
        # Update degrees of remaining nodes
        # A node v remains if not to_remove_mask[v]
        # If C[v, u] is True, degree[v] decreases by 1
        # Note: C is symmetric, so C[v, u] == C[u, v]
        # So we can use C[:, u] to find which nodes were connected to u
        
        connected_to_u = C[:, u]
        
        # Update degrees for all nodes
        # But only those that remain?
        # If a node is removed, its degree doesn't matter.
        # If a node remains, and it was connected to u, its degree drops by 1.
        
        # degrees[connected_to_u] -= 1
        # But we must be careful not to update removed nodes? 
        # It doesn't matter if we update them, they are masked out.
        
        degrees[connected_to_u] -= 1
        
        # Remove from pool
        in_pool[to_remove_mask] = False
        
    return selected

def construct():
    deadline = time.time() + 110
    
    # 1. Try R^2 = 2 (Baseline)
    # Generate all weight 2 vectors
    vecs_w2 = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vecs_w2.append(v)
    best_vectors = np.array(vecs_w2, dtype=np.int64)
    best_score = len(best_vectors)
    
    # 2. Try R^2 = 3
    if time.time() < deadline - 10:
        vecs_w3 = _generate_supports_and_vectors()
        # Solve MIS
        selected_indices = _greedy_mis(vecs_w3)
        candidate_vectors = vecs_w3[selected_indices]
        
        if len(candidate_vectors) > best_score:
            best_vectors = candidate_vectors
            best_score = len(candidate_vectors)
            
    # 3. Try R^2 = 4 (Maybe)
    # Weight 4 vectors (four 1s)
    # C(11,4) * 16 = 5280.
    # Constraint: u·v <= 2.
    # This graph is larger.
    # Let's see if we have time.
    
    if time.time() < deadline - 30:
        vecs_w4 = []
        for combo in combinations(range(D), 4):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    for s3 in (1, -1):
                        for s4 in (1, -1):
                            v = [0] * D
                            v[combo[0]] = s1
                            v[combo[1]] = s2
                            v[combo[2]] = s3
                            v[combo[3]] = s4
                            vecs_w4.append(v)
        
        # This might be slow to generate and solve.
        # Let's just generate and check time.
        if time.time() < deadline - 20:
             vecs_w4_arr = np.array(vecs_w4, dtype=np.int8)
             selected_indices = _greedy_mis(vecs_w4_arr)
             candidate_vectors = vecs_w4_arr[selected_indices]
             if len(candidate_vectors) > best_score:
                 best_vectors = candidate_vectors
                 best_score = len(candidate_vectors)

    return np.array(best_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
