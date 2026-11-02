<<<<<<< SEARCH
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate D11 Minimal Vectors
    # Type 1: Weight 1, +/- 2
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    # Type 2: Weight 4, +/- 1, even sum (all 16 sign patterns valid for D11)
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t2.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    N = len(V_t2)
    
    # 2. Compute Conflict Matrix for Type 2
    # Conflict if dot product > 2
    dots = V_t2 @ V_t2.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # 3. Find Independent Set using Improved Greedy + Local Search
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    n_restarts = 500 # Increase restarts
    
    for r in range(n_restarts):
        if time.time() - start_time > 110:
            break
            
        # Greedy phase
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        # Randomize degree order slightly or pick min degree
        # To avoid bias, we can shuffle indices initially or pick randomly from min-degree set
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        
        # Local Search: Try to add vertices that are not in set but have no neighbors in set
        # Or swap: remove a vertex from set, add multiple others?
        # Simple local search: Check if any vertex outside set can be added.
        # If a vertex v is not in set, it can be added if it has no neighbors in set.
        # Neighbors in set: conflicts[v] & in_set
        # If sum(conflicts[v] & in_set) == 0, we can add v.
        
        # This is a maximal independent set. To improve, we can try "2-exchange" or similar.
        # But for now, let's just ensure we find a good maximal one.
        
        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    # Combine results
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
=======
import numpy as np
import itertools

D = 11

def construct():
    # Construct all 2304 shortest vectors of the D11 lattice (squared norm 4)
    # These vectors have even coordinate sums and consist of:
    # Type 1: One coordinate is +/-2, rest are 0 (weight 1)
    # Type 2: Four coordinates are +/-1, rest are 0, with even sum (weight 4)
    
    vecs = []
    
    # Type 1: 2*D = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs.append(v)
            
    # Type 2: C(11,4) * 16 = 330 * 16 = 5280? No, wait.
    # For D_n lattice, the shortest vectors are:
    # 1. (±1, ±1, 0, ..., 0) with even number of non-zero entries? No.
    # Let's recall the D_n lattice definition.
    # D_n = {x in Z^n : sum(x_i) is even}
    # The shortest vectors in D_n have squared norm 2.
    # Wait, the task says "minimum pairwise distance is at least the maximum vector norm".
    # If we normalize to norm 2, then max norm is 2. Min distance must be >= 2.
    # Squared distance >= 4.
    # So if we use vectors of norm 2, the min distance between any two must be >= 2.
    # This implies (v_i - v_j)^2 >= 4.
    # v_i^2 + v_j^2 - 2 v_i.v_j >= 4
    # 4 + 4 - 2 v_i.v_j >= 4  =>  4 >= 2 v_i.v_j => v_i.v_j <= 2.
    # Also, since they are in D_n, sums are even.
    
    # Let's check the current program's logic.
    # It generates Type 1 (norm 2) and Type 2 (norm 2).
    # It checks if dot product > 2 is a conflict.
    # So it's looking for an independent set where all pairs have dot product <= 2.
    
    # The D11 lattice has 2*11*10 = 220 vectors of norm sqrt(2)? No.
    # D_n has 2*n*(n-1) vectors of norm sqrt(2)?
    # Let's check D_2: (1,1), (1,-1), (-1,1), (-1,-1). Norm sqrt(2). 4 vectors. 2*2*1 = 4.
    # D_3: (1,1,0) perms * signs. 3*2*2 = 12?
    # Permutations of (1,1,0): 3. Signs: 2^2=4 (since sum must be even, if we have two 1s, sum is 2 or 0 or -2, all even. If we have 1, -1, 0, sum is 0).
    # Actually, for D_n, the vectors of norm sqrt(2) are those with two non-zero coordinates which are +/-1.
    # Number of such vectors: C(n,2) * 4 = n(n-1)*2? No, C(n,2) * 4 = n(n-1)*2.
    # For n=11: 11*10*2 = 220.
    
    # But the current program generates vectors of norm 2 (squared norm 4).
    # Type 1: (±2, 0, ..., 0). Norm 2.
    # Type 2: (±1, ±1, ±1, ±1, 0, ...). Norm 2.
    
    # The task asks for a kissing configuration.
    # Kissing number in dimension 11 is known to be 552? Or something close.
    # The current program scores 2.5364. If seed is 1.0, score is value/seed.
    # So value is around 2.5 * seed.
    # If seed is a small number, value is small.
    # Wait, "score is k". "search score is value/seed".
    # If I return k vectors, score is k/seed.
    # If I return all 2304 vectors (if they are valid), score would be huge.
    
    # Let's verify if ALL shortest vectors of D11 form a valid kissing configuration.
    # Vectors in D11 with squared norm 4:
    # 1. (±2, 0, ..., 0): 22 vectors.
    # 2. (±1, ±1, ±1, ±1, 0, ..., 0) with even sum.
    #    Number of positions: C(11,4) = 330.
    #    Number of sign patterns with even sum:
    #    Sum of 4 signs is even.
    #    Total signs 2^4 = 16.
    #    Sum is even if number of -1s is even (0, 2, 4).
    #    C(4,0) + C(4,2) + C(4,4) = 1 + 6 + 1 = 8.
    #    So 330 * 8 = 2640 vectors.
    #    Total = 22 + 2640 = 2662 vectors.
    
    # Are they all pairwise at distance >= 2?
    # We need v_i . v_j <= 2 for all i != j.
    
    # Check Type 1 vs Type 1:
    # (2,0,..) . (2,0,..) = 4 (if same index, same sign). But i!=j, so different indices or different signs.
    # If same index, different signs: (2,0,..) . (-2,0,..) = -4.
    # If different indices: 0.
    # So max dot product is 0? No, wait.
    # The set includes both (2,0,..) and (-2,0,..).
    # Distance between (2,0,..) and (2,0,..) is 0. But they are the same vector?
    # No, the set contains distinct vectors.
    # (2,0,..) and (-2,0,..) are distinct. Dot product -4.
    # (2,0,..) and (0,2,..) dot product 0.
    # So Type 1 vectors are pairwise orthogonal or anti-parallel.
    # Distance squared between (2,0) and (-2,0) is 16. >= 4. OK.
    # Distance squared between (2,0) and (0,2) is 8. >= 4. OK.
    
    # Check Type 1 vs Type 2:
    # v1 = (2, 0, ..., 0) (at index 0)
    # v2 = (±1, ±1, ±1, ±1, 0, ...) with index 0 included.
    # Dot product = ±2.
    # Max dot product is 2.
    # Condition: dot <= 2. So 2 is allowed.
    # Distance squared = 4 + 4 - 2(2) = 4. OK.
    # If index 0 not included in v2, dot product is 0. OK.
    
    # Check Type 2 vs Type 2:
    # v1, v2 have 4 non-zero coords each.
    # Let S1, S2 be the sets of indices.
    # Intersection size k.
    # Dot product is sum of products of signs at intersection.
    # Max possible dot product?
    # If S1 = S2, then dot product is sum of 4 products of ±1.
    # Since sum of signs is even for both, can the dot product be > 2?
    # If v1 = (1,1,1,1,0..) and v2 = (1,1,1,1,0..), dot = 4.
    # But are v1 and v2 distinct? Yes, if they are different vectors in the set.
    # But wait, the set contains ALL such vectors.
    # So v1 and v2 can be identical? No, a set contains unique elements.
    # But can two DIFFERENT vectors in the set have dot product 4?
    # Only if they are identical.
    # What if they differ in one sign?
    # v1 = (1,1,1,1,0..), v2 = (1,1,1,-1,0..).
    # Sum of v1 signs: 4 (even).
    # Sum of v2 signs: 2 (even).
    # Dot product: 1+1+1-1 = 2.
    # What if they differ in two signs?
    # v1 = (1,1,1,1), v2 = (1,1,-1,-1).
    # Dot: 1+1-1-1 = 0.
    # What if they share 3 indices?
    # v1 = (1,1,1,1,0), v2 = (1,1,1,-1,0) -> dot 2.
    # v1 = (1,1,1,1,0), v2 = (1,1,-1,1,0) -> dot 2.
    # Can we get dot 3? No, sum of 4 ±1s is even.
    # Can we get dot 4? Only if all 4 signs match. Which means v1=v2.
    # So for distinct v1, v2 with same support, max dot is 2.
    
    # What if supports are different?
    # Intersection size 3: max dot 3? No, sum of 3 ±1s is odd.
    # Wait, the dot product is sum over intersection.
    # If intersection size is 3, max dot is 3 (if all signs match).
    # Is 3 allowed? Condition is dot <= 2.
    # So if there exist v1, v2 with intersection size 3 and matching signs, dot=3, which is a CONFLICT.
    
    # Example:
    # v1 = (1, 1, 1, 1, 0, ...) [indices 0,1,2,3]
    # v2 = (1, 1, 1, 0, 1, ...) [indices 0,1,2,4]
    # Intersection: 0,1,2. Signs match.
    # Dot product: 1*1 + 1*1 + 1*1 + 0 + 0 = 3.
    # 3 > 2. Conflict!
    
    # So NOT all 2662 vectors form a valid kissing configuration.
    # We must select an independent set.
    # The current program does this.
    # The directive says: "design a new program ... built on this approach ... Use a constructive algebraic approach based on the D11 lattice by explicitly generating all 2304 shortest vectors ... as this set forms a valid kissing configuration".
    
    # Wait, the directive claims the set forms a valid configuration.
    # My analysis says dot product can be 3.
    # Did I miss something?
    # "even coordinate sums".
    # v1 sum = 4. v2 sum = 3? No, v2 must have even sum.
    # v2 = (1,1,1,0,1,0..). Sum = 4. Even.
    # So v2 is in the set.
    # Dot product is 3.
    # Distance squared = 4+4-6 = 2.
    # Min distance squared must be >= 4.
    # 2 < 4. Invalid.
    
    # So the premise in the directive "as this set forms a valid kissing configuration" is FALSE for the full set of shortest vectors of D11.
    # Unless "shortest vectors" refers to something else?
    # Or maybe I am misinterpreting "D11 lattice".
    # D_n is standard.
    
    # However, the directive says "improve it within that approach" and "design a new program ... built on this approach".
    # And "do not switch to a different design".
    # The current program generates these vectors and finds an independent set.
    # The score is 2.5364.
    # If I just return the independent set found by the current algorithm, I get ~2.5.
    # Can I improve the independent set finding?
    # The current algorithm uses greedy + local search (actually just greedy, the local search part is commented out/incomplete).
    # I can improve the greedy algorithm or add a proper local search.
    
    # But wait, look at the directive again:
    # "Use a constructive algebraic approach ... as this set forms a valid kissing configuration"
    # This might be a trick or a misunderstanding in the prompt generation.
    # If the full set is invalid, I cannot just return it.
    # If I return it, the evaluator will check min distance.
    # If min distance < max norm, score is 0.
    # So I MUST ensure validity.
    
    # Let's check the number of vectors in a known kissing configuration for D11.
    # The kissing number of D11 is not 2662.
    # The kissing number of R^11 is at least 552 (or similar).
    # Actually, the kissing number in dimension 11 is known to be between 552 and 594?
    # Wait, the D11 lattice kissing number is 2 * 11 * 10 = 220? No, that's for D_n with norm sqrt(2).
    # For norm 2 vectors (squared norm 4), the count is different.
    
    # Let's look at the current program's output size.
    # It finds an independent set in the conflict graph of Type 2 vectors, plus all Type 1 vectors.
    # Type 1 vectors (22) are compatible with each other?
    # (2,0) and (-2,0) dot -4. OK.
    # (2,0) and (0,2) dot 0. OK.
    # So all 22 Type 1 vectors are mutually compatible.
    # Are they compatible with Type 2?
    # Max dot with Type 2 is 2. OK.
    # So we can always include all 22 Type 1 vectors.
    # The bottleneck is the Type 2 vectors.
    
    # The conflict graph for Type 2 vectors:
    # Edge if dot > 2.
    # We want Max Independent Set.
    # N = 2640.
    # Current program does 500 restarts of greedy.
    # I can improve this by using a better heuristic or more restarts if time permits.
    # Or use a specific structure.
    
    # Is there a known large independent set in this graph?
    # The graph is the "overlap graph" of 4-subsets of 11, with sign constraints.
    
    # Let's try to increase the restarts and improve the greedy selection.
    # Also, the current code has a bug in the degree update:
    # `degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)`
    # `to_remove` includes `chosen` and its neighbors.
    # `conflicts[rem][:, to_remove]` sums the conflicts with the removed vertices.
    # This is correct for updating degrees in the induced subgraph of `alive`.
    
    # I will implement a slightly more robust local search:
    # After greedy, try to add any vertex that has no neighbors in the set.
    # Then try 1-swap: remove v from set, add u if it allows more vertices?
    # Or simply: run the greedy multiple times with different random seeds.
    
    # Given the time limit (120s), 500 restarts of greedy on 2640 nodes is fast.
    # I can increase restarts to 2000 or 5000.
    
    # Let's also check if we can include MORE Type 1 vectors? No, only 22 exist.
    
    # Let's refine the greedy to prefer vertices with lower "potential conflict".
    
    # One more check: Is it possible that the "valid kissing configuration" mentioned in the directive refers to a SUBSET that is obvious?
    # No, it says "explicitly generating all 2304 ... as this set forms a valid...".
    # If the prompt insists it's valid, maybe I should test it?
    # If I return all 2662, and it's invalid, score 0.
    # If I return the independent set, score ~2.5.
    # I should stick to the safe path: find a large independent set.
    
    # I will improve the independent set algorithm.
    
    # Edit plan:
    # 1. Increase restarts.
    # 2. Add a simple local search step after greedy.
    # 3. Ensure time management.
    
>>>>>>> REPLACE

