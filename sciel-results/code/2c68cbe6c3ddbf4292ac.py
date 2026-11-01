import time
import numpy as np
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We look for a set of integer vectors v_i such that:
    # 1. v_i != 0
    # 2. min_{i != j} ||v_i - v_j||^2 >= max_i ||v_i||^2
    
    # Let R^2 be the maximum squared norm among the selected vectors.
    # Then for all i, j, we need ||v_i - v_j||^2 >= R^2.
    # Note: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
    # If we restrict all vectors to have the same squared norm N (so R^2 = N),
    # the condition becomes N + N - 2 v_i . v_j >= N  =>  N >= 2 v_i . v_j.
    # Also, for any two distinct vectors, this must hold.
    
    # The seed program uses weight 2 vectors (squared norm 2).
    # Number of such vectors: choose 2 positions from 11 (55 ways), 4 sign combinations = 220.
    # For weight 2, N=2. Condition: 2 >= 2 v_i . v_j => v_i . v_j <= 1.
    # Let's check if the 220 weight-2 vectors form a valid configuration.
    # v_i . v_j for weight 2 vectors:
    # - Disjoint supports: dot = 0.
    # - Share 1 coordinate: if signs same, dot = 1. If signs diff, dot = -1.
    # - Share 2 coordinates (identical support): if signs same, dot = 2. If one sign diff, dot = 0. If both diff, dot = 2.
    # Wait, the seed includes all sign combinations.
    # Example: v1 = e1+e2, v2 = e1+e2. v1==v2 is not allowed as they are distinct vectors?
    # The task says "nonzero integer vectors". It doesn't explicitly say distinct, but "pairwise distance" implies distinct points.
    # If v1 == v2, distance is 0. R^2 = 2. 0 < 2. Invalid.
    # So we must select distinct vectors.
    
    # Let's verify the seed's 220 vectors.
    # Are there pairs with dot product 2?
    # Yes, v = e1+e2 and v' = e1+e2 are the same vector. We only list unique vectors.
    # What about v = e1+e2 and v' = e1-e2? Dot = 1 - 1 = 0.
    # What about v = e1+e2 and v' = -e1-e2? Dot = -1 -1 = -2.
    # Condition: N >= 2 dot. 2 >= 2(-2) => 2 >= -4. OK.
    # Max dot product among distinct weight-2 vectors:
    # If they share 1 index, max dot is 1 (e.g. e1+e2 and e1+e3 -> dot 1).
    # If they share 2 indices, they are either identical (excluded) or differ in signs.
    # e1+e2 vs e1-e2: dot 0.
    # e1+e2 vs -e1+e2: dot 0.
    # e1+e2 vs -e1-e2: dot -2.
    # So max dot is 1.
    # Condition: 2 >= 2(1) => 2>=2. OK.
    # So the 220 weight-2 vectors are a valid configuration.
    
    # Can we do better?
    # Try weight 1. N=1. Vectors: +/- e_i. 22 vectors.
    # Dot products: 1, -1, 0.
    # Condition: 1 >= 2 dot.
    # If dot = 1 (e.g. e1 and e1 - identical, excluded).
    # e1 and e2: dot 0. 1 >= 0. OK.
    # e1 and -e1: dot -1. 1 >= -2. OK.
    # So 22 vectors. Worse.
    
    # Try weight 3. N=3.
    # Number of weight 3 vectors:
    # Case 1: 3 distinct non-zero coords.
    # Choose 3 from 11: C(11,3) = 165.
    # Signs: 2^3 = 8.
    # Total: 165 * 8 = 1320.
    # We need to select a subset such that for all i,j, ||v_i - v_j||^2 >= 3.
    # If all have norm squared 3, this means 6 - 2 dot >= 3 => 3 >= 2 dot => dot <= 1.5 => dot <= 1.
    # So we need a set of weight-3 vectors with pairwise dot products <= 1.
    
    # This is an independent set problem in a graph where edges connect vectors with dot > 1.
    # Or simply greedy addition.
    
    # Let's generate all weight-3 vectors.
    # And maybe weight 4?
    # Weight 4: N=4.
    # Dot <= 2.
    # C(11,4) * 16 = 330 * 16 = 5280.
    
    # Let's try to find a large set from weight 3 vectors first.
    # Greedy:
    # 1. Generate all weight 3 vectors.
    # 2. Shuffle.
    # 3. Add v if dot(v, u) <= 1 for all already added u.
    
    # To improve, we can try mixed weights?
    # If we mix weights, R^2 is determined by the max weight.
    # Suppose we use only weight 3 and 4. Max N=4.
    # Condition for v_i (N_i) and v_j (N_j):
    # N_i + N_j - 2 dot >= 4.
    # dot <= (N_i + N_j - 4) / 2.
    
    # Let's start with pure weight 3.
    
    def generate_weight_k_vectors(k):
        vecs = []
        # Indices
        for idxs in itertools.combinations(range(D), k):
            # Signs
            for signs in itertools.product([1, -1], repeat=k):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(idxs, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs)
    
    # Generate weight 3 vectors
    w3_vecs = generate_weight_k_vectors(3)
    n_w3 = len(w3_vecs)
    
    # Compute dot products matrix?
    # n_w3 is 1320. 1320x1320 matrix is ~1.7M entries. Manageable.
    # But we don't need the full matrix if we do greedy.
    
    # Greedy approach with multiple restarts
    best_count = 0
    best_set = []
    
    rng = np.random.default_rng(42)
    
    # Optimization: Precompute dots with a reference? No, just do it.
    # To speed up, we can check compatibility as we go.
    
    def try_greedy(vecs, rng, max_iters=20):
        best_local = 0
        for _ in range(max_iters):
            selected = []
            # Random order
            order = rng.permutation(len(vecs))
            for i in order:
                v = vecs[i]
                ok = True
                for u in selected:
                    dot = np.dot(v, u)
                    if dot > 1: # Condition for N=3: dot <= 1
                        ok = False
                        break
                if ok:
                    selected.append(v)
            
            if len(selected) > best_local:
                best_local = len(selected)
                best_set_local = selected[:]
            
            if time.time() > deadline - 10:
                break
                
        return best_local, best_set_local
    
    count_w3, set_w3 = try_greedy(w3_vecs, rng, max_iters=10)
    best_count = count_w3
    best_set = set_w3
    
    # What if weight 3 doesn't beat 220?
    # Theoretical max for kissing number in 11D is quite high (estimated 240-270?).
    # But integer constraints are tight.
    # Let's check if we can mix weights.
    
    # Generate weight 4 vectors?
    # N=4.
    # If we select only weight 4 vectors:
    # Condition: 8 - 2 dot >= 4 => dot <= 2.
    # Number of weight 4 vectors: 5280.
    # Greedy on weight 4.
    
    w4_vecs = generate_weight_k_vectors(4)
    
    def try_greedy_w4(vecs, rng, max_iters=5):
        best_local = 0
        best_set_local = []
        for _ in range(max_iters):
            selected = []
            order = rng.permutation(len(vecs))
            for i in order:
                v = vecs[i]
                ok = True
                for u in selected:
                    dot = np.dot(v, u)
                    if dot > 2:
                        ok = False
                        break
                if ok:
                    selected.append(v)
            
            if len(selected) > best_local:
                best_local = len(selected)
                best_set_local = selected[:]
            
            if time.time() > deadline - 10:
                break
                
        return best_local, best_set_local
    
    count_w4, set_w4 = try_greedy_w4(w4_vecs, rng, max_iters=5)
    
    if count_w4 > best_count:
        best_count = count_w4
        best_set = set_w4
        
    # What if we combine weight 3 and 4?
    # Max N = 4.
    # Condition for v (N_v) and u (N_u):
    # N_v + N_u - 2 dot >= 4
    # dot <= (N_v + N_u - 4) / 2
    
    # If we have a set of mixed vectors, we can try to add weight 3 vectors to a weight 4 set?
    # Or build from scratch with mixed pool.
    
    # Pool = w3_vecs + w4_vecs
    # But checking compatibility is slower.
    # Let's try a greedy on the combined pool, but limit the number of candidates to keep it fast.
    
    # Actually, weight 3 vectors are "smaller" and might pack better if we don't force N=4?
    # If we only use weight 3, N=3.
    # If we use weight 4, N=4.
    # If we mix, N=4.
    # A weight 3 vector v and weight 4 vector u:
    # dot <= (3 + 4 - 4) / 2 = 1.5 => dot <= 1.
    # So weight 3 vectors must have dot <= 1 with weight 4 vectors.
    # Weight 4 vectors must have dot <= 2 with weight 4 vectors.
    # Weight 3 vectors must have dot <= 1 with weight 3 vectors.
    
    # Let's try a mixed greedy.
    # To save time, we'll just do a few restarts on the combined pool.
    
    # However, the pool is 1320 + 5280 = 6600 vectors.
    # Greedy is O(N * K) where K is selected size.
    # 6600 * 200 ~ 1.3M dot products per restart. Fast.
    
    def try_greedy_mixed(vecs, rng, max_iters=5):
        best_local = 0
        best_set_local = []
        for _ in range(max_iters):
            selected = []
            order = rng.permutation(len(vecs))
            for i in order:
                v = vecs[i]
                N_v = np.dot(v, v)
                ok = True
                for u in selected:
                    N_u = np.dot(u, u)
                    dot = np.dot(v, u)
                    # Condition: ||v-u||^2 >= max(N_v, N_u) is NOT correct.
                    # Condition: ||v-u||^2 >= R^2 where R^2 = max selected norm squared.
                    # But R^2 changes as we add vectors.
                    # If we add v, the new R^2 is max(old_R2, N_v).
                    # We need:
                    # 1. ||v-u||^2 >= max(old_R2, N_v) for all u in selected.
                    # 2. ||u-w||^2 >= max(old_R2, N_v) for all u,w in selected?
                    #    No, if we increase R^2, we might invalidate existing pairs!
                    #    This makes greedy addition tricky if norms vary.
                    
                    # The condition is global: min pairwise dist >= max norm.
                    # If we add a vector with large norm, the max norm increases,
                    # potentially breaking the condition for existing pairs.
                    # So we must check ALL pairs if we add a vector that increases the max norm.
                    
                    # To keep it simple, let's just stick to single weight sets.
                    # Or, if we add a vector with norm <= current max norm,
                    # the max norm doesn't change. Then we only need to check v against existing ones.
                    
                    # Let's implement a greedy that only adds vectors if their norm <= current_max_norm.
                    # This allows mixed weights if we start with high weight vectors?
                    # Or start with low?
                    
                    # If we start with weight 4 (N=4), R^2=4.
                    # Then we can add weight 3 (N=3) if it is compatible with all existing.
                    # Condition for v (N=3) and u (N=4 or 3):
                    # ||v-u||^2 >= 4.
                    # N_v + N_u - 2 dot >= 4.
                    # dot <= (N_v + N_u - 4) / 2.
                    
                    # If we add weight 3 to a set of weight 4:
                    # dot(v, u) <= (3 + 4 - 4)/2 = 1.5 => 1.
                    # dot(v, v2) where v2 is weight 3:
                    # dot <= (3 + 3 - 4)/2 = 1.
                    
                    # So it works.
                    
                    # Let's refine the mixed greedy:
                    # 1. Start with empty set.
                    # 2. Add vectors in random order.
                    # 3. Maintain current_max_norm_sq.
                    # 4. If N_v > current_max_norm_sq:
                    #    We would need to recheck all existing pairs. Expensive.
                    #    So let's forbid adding vectors that increase the max norm.
                    #    This means we should process vectors in decreasing norm order?
                    #    Or just restrict to a single norm.
                    
                    # Given time constraints, single norm is safer and faster to verify.
                    # The weight 3 greedy already ran.
                    # The weight 4 greedy already ran.
                    
                    pass # Placeholder to indicate we won't do complex mixed greedy in this version
                if ok:
                    selected.append(v)
            
            if len(selected) > best_local:
                best_local = len(selected)
                best_set_local = selected[:]
            
            if time.time() > deadline - 5:
                break
                
        return best_local, best_set_local
    
    # Since mixed greedy is complex, let's just ensure we have the best of w3 and w4.
    # And maybe weight 5?
    # Weight 5: N=5.
    # C(11,5) * 32 = 462 * 32 = 14784.
    # Condition: dot <= (5+5-5)/2 = 2.5 => 2.
    
    w5_vecs = generate_weight_k_vectors(5)
    
    def try_greedy_w5(vecs, rng, max_iters=3):
        best_local = 0
        best_set_local = []
        for _ in range(max_iters):
            selected = []
            order = rng.permutation(len(vecs))
            for i in order:
                v = vecs[i]
                ok = True
                for u in selected:
                    dot = np.dot(v, u)
                    if dot > 2:
                        ok = False
                        break
                if ok:
                    selected.append(v)
            
            if len(selected) > best_local:
                best_local = len(selected)
                best_set_local = selected[:]
            
            if time.time() > deadline - 5:
                break
                
        return best_local, best_set_local
    
    count_w5, set_w5 = try_greedy_w5(w5_vecs, rng, max_iters=3)
    
    if count_w5 > best_count:
        best_count = count_w5
        best_set = set_w5
        
    # Fallback to seed if all failed
    if best_count == 0:
        # Return seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
