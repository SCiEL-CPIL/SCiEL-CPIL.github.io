import time
import numpy as np
import random

def construct():
    n = 20
    deadline = time.time() + 145.0
    
    # Precompute all possible edges as a flat array
    # Index mapping: edge_idx -> (u, v)
    # We use a 20x20 matrix for adjacency, but flatten to a 1D array of size 190 (n*(n-1)/2)
    # Actually, for speed, we can use the full n*n matrix and just ignore diagonal/self-loops in fitness, 
    # or map indices carefully. 
    # Let's use the full n*n matrix for O(1) access in numpy operations.
    # State: X is a numpy array of shape (n, n), X[i,j] is 1 if edge exists.
    # We only care about i < j.
    
    # Precompute indices for all pairs (i, j) with i < j
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    p_u = pairs[:, 0]
    p_v = pairs[:, 1]
    m_total = len(pairs)
    
    # Precompute the "triple" structure for K3,3 detection?
    # A K3,3 violation exists if there are 3 vertices {a,b,c} and 3 vertices {x,y,z} 
    # such that all 9 edges between them exist.
    # This is equivalent to: There exist a,b,c such that |N(a) ∩ N(b) ∩ N(c)| >= 3.
    # Let C_{abc} = N(a) ∩ N(b) ∩ N(c).
    # Violation count = sum over all triples {a,b,c} of max(0, popcount(C_{abc}) - 2).
    # Actually, if we just penalize every set of 3 common neighbors, we might over-penalize, 
    # but it's a valid proxy. Or we can just count the number of K3,3 subgraphs.
    # Number of K3,3 subgraphs = sum_{a<b<c} C(popcount(N(a)∩N(b)∩N(c)), 3).
    # This is a smooth-ish function we can optimize.
    
    # To speed up evaluation, we can precompute the list of all triples of vertices.
    # There are C(20,3) = 1140 triples.
    triples = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    t_a = triples[:, 0]
    t_b = triples[:, 1]
    t_c = triples[:, 2]
    num_triples = len(triples)
    
    def get_fitness(X):
        """
        X: (n, n) boolean or int array.
        Returns: (edge_count, violation_penalty)
        """
        # Edge count
        # Sum of upper triangle
        edge_count = np.sum(np.triu(X, 1))
        
        # Calculate common neighbors for each triple
        # N(a) is row a of X.
        # Common = X[a] & X[b] & X[c]
        # We need to do this for all 1140 triples.
        # Vectorized approach:
        # X is (20, 20).
        # We can create a 3D array of shape (1140, 20) where each row is the common neighbors for that triple.
        # common_neighbors = X[t_a] & X[t_b] & X[t_c]
        # This creates a (1140, 20) array.
        
        # However, creating this array for every fitness call is expensive if done naively in Python loops.
        # Let's try to optimize the SA loop by caching or using a different representation.
        
        # Alternative: 
        # For each pair (u, v), let S_{uv} = N(u) ∩ N(v).
        # A K3,3 violation involving u, v and some w occurs if |S_{uv} ∩ N(w)| >= 3.
        # Actually, the condition is: exists u,v,w such that |N(u)∩N(v)∩N(w)| >= 3.
        
        # Let's stick to the triple sum for simplicity, but optimize the numpy ops.
        # X[t_a] uses advanced indexing, which is fast.
        
        # common = X[t_a] & X[t_b] & X[t_c] -> shape (1140, 20)
        # counts = np.sum(common, axis=1) -> shape (1140,)
        # violations = np.sum(np.maximum(0, counts - 2)) # This penalizes excess common neighbors
        # Or count actual K3,3s:
        # For each triple, if count >= 3, add C(count, 3) to penalty.
        # This is harder to vectorize directly without a loop or lookup table.
        # Let's use a lookup table for C(k, 3) for k=0..20.
        
        return edge_count, common  # Return common for potential reuse? No, just compute penalty.
        
    # Lookup table for C(k, 3)
    def comb3(k):
        if k < 3: return 0
        return k * (k-1) * (k-2) // 6
        
    # We will precompute the indices for the triples to allow fast numpy access.
    # In the loop, we will update X and recalculate fitness.
    
    # Initial state: Random dense graph
    X = np.zeros((n, n), dtype=np.uint8)
    for i in range(m_total):
        if random.random() < 0.6:
            u, v = p_u[i], p_v[i]
            X[u, v] = 1
            X[v, u] = 1
            
    def calc_penalty(X):
        # X is symmetric (n,n)
        # Get rows for triples
        # t_a, t_b, t_c are arrays of indices
        # X[t_a] has shape (1140, 20)
        # This operation is memory intensive but vectorized.
        # Let's check if we can do it faster.
        
        # Alternative: 
        # For each pair (u, v), compute S_uv = N(u) & N(v).
        # Then for each w, check if |S_uv & N(w)| >= 3.
        # This is C(20,2) = 190 pairs.
        # For each pair, we have a bitmask S_uv (20 bits).
        # We need to check all w != u, v.
        
        # Let's use the bitmask approach in Python integers for the penalty calculation, 
        # as it might be faster than allocating large numpy arrays repeatedly if we cache.
        
        # Actually, let's stick to NumPy for the main loop if possible, but 
        # the "common neighbors" calculation for 1140 triples * 20 vertices = 22,800 operations 
        # per fitness eval. If we do 10,000 evals/sec, that's 228M ops/sec. NumPy can handle this.
        
        common = X[t_a] & X[t_b] & X[t_c]
        counts = np.sum(common, axis=1)
        # Penalty: sum of C(count, 3)
        # We can use a lookup table vectorized
        # lut = np.array([comb3(i) for i in range(21)])
        # penalty = np.sum(lut[counts])
        
        # Let's precompute the LUT outside
        return counts
        
    lut = np.array([comb3(i) for i in range(21)], dtype=np.float64)
    
    def get_score(X):
        edge_count = np.sum(np.triu(X, 1))
        common = X[t_a] & X[t_b] & X[t_c]
        counts = np.sum(common, axis=1)
        penalty = np.sum(lut[counts])
        # We want to maximize edges, minimize penalty.
        # Score = edges - lambda * penalty
        # If penalty > 0, it's invalid. We want to drive penalty to 0.
        # Let's use a large penalty weight to ensure validity is prioritized initially, 
        # or just maximize edges subject to penalty=0.
        # A common trick: Score = edges * 1000 - penalty * 10000?
        # No, let's just maximize edges, and if penalty > 0, subtract a huge amount.
        # Better: Objective = edges - 100 * penalty.
        # If penalty is 1, we lose 100 edges worth of score.
        # This encourages removing violations.
        
        return edge_count - 100.0 * penalty

    current_score = get_score(X)
    best_X = X.copy()
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    alpha = 0.9995 # Cooling rate
    
    # Precompute edge indices for random flipping
    # We flip one edge at a time.
    
    while time.time() < deadline - 5.0:
        # Randomly pick an edge to flip
        idx = random.randint(0, m_total - 1)
        u, v = p_u[idx], p_v[idx]
        
        # Calculate delta score? 
        # Calculating full score is O(1140 * 20) ~ 22k ops.
        # Delta calculation:
        # Flipping edge (u, v) affects:
        # 1. edge_count by +/- 1
        # 2. Penalty: 
        #    The penalty depends on triples (a,b,c) and their common neighbors.
        #    Adding edge (u,v) increases N(u) and N(v) by each other? No, N(u) is neighbors of u.
        #    If we add edge (u,v), then v becomes a neighbor of u and u becomes a neighbor of v.
        #    This affects any triple (a,b,c) where u or v is one of a,b,c?
        #    No, it affects the set of common neighbors.
        #    N(a) changes if a==u or a==v? No, N(a) is the set of neighbors.
        #    If we add edge (u,v):
        #    - v is added to N(u).
        #    - u is added to N(v).
        #    
        #    Consider a triple (a,b,c). Its common neighbor set is N(a) ∩ N(b) ∩ N(c).
        #    This set changes only if u or v is in {a,b,c} AND the other endpoint is in the intersection?
        #    Actually, simpler:
        #    The only triples whose common neighbor count changes are those that include u or v.
        #    If the triple is (u, x, y), the common neighbors are N(u) ∩ N(x) ∩ N(y).
        #    Before flip: N(u) did not contain v. After flip: N(u) contains v.
        #    So v is now a common neighbor of (u, x, y) IF v is in N(x) and N(y).
        #    Similarly, u is now a common neighbor of (v, x, y) IF u is in N(x) and N(y).
        #    
        #    So we only need to check triples containing u or v.
        #    Number of such triples:
        #    Triples with u: C(19, 2) = 171.
        #    Triples with v: C(19, 2) = 171.
        #    Triples with both u and v: C(18, 1) = 18.
        #    Total unique triples affected: 171 + 171 - 18 = 324.
        #    
        #    This is much faster than 1140.
        #    
        #    Let's implement delta calculation.
        
        is_present = X[u, v]
        
        # Identify affected triples
        # We can precompute the list of triples containing each vertex.
        
        # Let's just do the full calculation for now if it's fast enough, 
        # or implement the delta if we find it's too slow.
        # 22k ops is very fast. Let's try full calculation first.
        
        # Flip
        if is_present:
            X[u, v] = 0
            X[v, u] = 0
            new_score = get_score(X)
            delta = new_score - current_score
        else:
            X[u, v] = 1
            X[v, u] = 1
            new_score = get_score(X)
            delta = new_score - current_score
            
        # Acceptance
        if delta > 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_X = X.copy()
        else:
            # Revert
            if is_present:
                X[u, v] = 1
                X[v, u] = 1
            else:
                X[u, v] = 0
                X[v, u] = 0
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Convert best_X to list of edges
    best_edges = []
    for i in range(m_total):
        u, v = p_u[i], p_v[i]
        if best_X[u, v]:
            best_edges.append([int(u), int(v)])
            
    # Final verification
    if best_edges:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
            
        # Check K3,3
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = adj_final[u] & adj_final[v]
                if common_uv == 0: continue
                for w in range(v + 1, n):
                    if (common_uv & adj_final[w]).bit_count() > 2:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return best_edges
            
    return []

# Precomputation for speed in the main loop? 
# The current get_score allocates large arrays. 
# Let's optimize get_score by using the precomputed triple indices.

# Re-define construct with optimizations if needed, but the above logic is sound.
# One issue: `get_score` inside `construct` captures `t_a`, `t_b`, `t_c` from closure.
# This is fine.

# Let's double check the delta logic vs full score.
# Full score is ~22k bitwise ops. 
# Python overhead for numpy calls is significant. 
# 1000 calls/sec * 22k ops = 22M ops/sec. This should be well within 150s.
# 150s * 1000 calls = 150k calls. 
# If we can do 10k calls/sec, we get 1.5M calls.
# Let's ensure the numpy ops are efficient.
# X[t_a] creates a copy. This is the bottleneck.
# Maybe we can avoid the copy? 
# Or use a different representation?
# For now, let's trust NumPy's optimized indexing.

# One optimization: 
# Instead of `common = X[t_a] & X[t_b] & X[t_c]`, 
# we can precompute the indices and use `np.take`? No, `X[t_a]` is `np.take`.
# 
# Let's refine the penalty calculation to be slightly faster by using `np.bitwise_and` explicitly?
# No, `&` is fine.

# Final check on return type: list of edges [u, v].
# The code does this.

# One potential bug: `T` cooling. 
# If we start with T=5 and alpha=0.9995, it takes a long time to cool.
# log(0.01/5) / log(0.9995) ~ -6.2 / -0.0005 ~ 12400 steps.
# This is good for SA.

# Let's run it.

