import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Build the full candidate vector set
    # Type A: ±2e_i (22 vectors, norm 2)
    # Type D: ±1 on 4 coords, 0 elsewhere (330*16 = 5280 vectors, norm 2)
    # Type M: mixed norm vectors - ±2 on 2 coords, ±1 on 1 coord (norm sqrt(5))
    #   Actually, let's think about what's compatible.
    #   For min pairwise distance >= max norm:
    #   If we include vectors of different norms, the constraint is:
    #   ||v_i - v_j|| >= max(||v_i||, ||v_j||) for all i != j
    #   
    #   Let's focus on the proven structure: all vectors of norm 2.
    #   Type A: ±2e_i, norm 2
    #   Type D: unit vectors on 4 coords, norm 2
    #   
    #   Distance between two Type A: ||2e_i - 2e_j|| = 2*sqrt(2) >= 2 ✓
    #   Distance between Type A (2e_i) and Type D (support S, signs):
    #     If i in S: ||2e_i - v||^2 = (2 - s_i)^2 + sum_{j in S, j!=i} 1 = 1 + 3 = 4, dist = 2 ✓
    #     If i not in S: ||2e_i - v||^2 = 4 + 4 = 8, dist = 2*sqrt(2) >= 2 ✓
    #   Distance between two Type D (supports S1, S2, signs):
    #     Need ||v1 - v2|| >= 2, i.e., ||v1-v2||^2 >= 4
    #     ||v1-v2||^2 = sum over all coords of (v1[c]-v2[c])^2
    #     On coords in S1∩S2: (±1-±1)^2 = 0 or 4
    #     On coords in S1\S2: (±1-0)^2 = 1
    #     On coords in S2\S1: (0-±1)^2 = 1
    #     On coords in neither: 0
    #     So ||v1-v2||^2 = 4*|{c in S1∩S2: signs differ}| + |S1\S2| + |S2\S1|
    #     = 4*|{c in S1∩S2: signs differ}| + 2*(4 - |S1∩S2|)
    #     Since |S1| = |S2| = 4, |S1\S2| = 4 - |S1∩S2|, same for S2\S1.
    #     So ||v1-v2||^2 = 4*k + 2*(4 - |S1∩S2|) where k = #sign differences in intersection
    #     We need this >= 4.
    #     If |S1∩S2| <= 2: 2*(4-|S1∩S2|) >= 4, so always OK regardless of signs.
    #     If |S1∩S2| = 3: 2*(4-3) = 2, need 4k + 2 >= 4, so k >= 1 (at least 1 sign diff)
    #     If |S1∩S2| = 4: 2*(4-4) = 0, need 4k >= 4, so k >= 1 (vectors differ in at least 1 sign)
    #     But if S1 = S2 and same signs, they're the same vector (distance 0).
    #     If S1 = S2 and different signs, k >= 1, distance^2 >= 4. ✓
    
    # So the conflict structure:
    # - Two Type D vectors conflict if their supports intersect in >= 3 coords AND 
    #   they have the same sign on all 3+ shared coords (i.e., k=0 when |intersection|>=3)
    #   Actually: if |intersection| = 3, need k >= 1. If |intersection| = 4, need k >= 1.
    #   Conflict when |intersection| >= 3 AND k = 0 (all signs agree on intersection).
    # - Type A never conflicts with Type D.
    # - Two Type A never conflict.
    
    # This means: for each support S (4-subset), we can pick at most one sign pattern
    # from each "equivalence class" where two patterns agree on all shared coords with
    # any other selected support.
    
    # Actually, let me reconsider. The key insight from the current program:
    # Two supports S1, S2 with |S1∩S2| >= 3 conflict if there exist sign assignments
    # to S1 and S2 that agree on the intersection. Since we're choosing WHICH sign patterns
    # to include, the constraint is more subtle.
    
    # Simplification: if we restrict to at most ONE sign pattern per support, then
    # two supports with |intersection| >= 3 always conflict (because we can always find
    # sign patterns that agree, but we need ALL pairs to be compatible).
    # Wait no - we're selecting specific vectors, not supports.
    
    # Let me think differently. The current program selects a set of supports (no two 
    # sharing >= 3 coords), then includes ALL 16 sign patterns for each selected support.
    # This works because if |S1∩S2| <= 2, then for ANY sign patterns on S1 and S2,
    # ||v1-v2||^2 >= 2*(4-2) = 4. ✓
    
    # Can we do better by allowing some supports with |intersection| = 3 but restricting
    # which sign patterns we include?
    # If S1∩S2 has size 3, we need for every v1 in our S1-patterns and v2 in our S2-patterns:
    #   4*k + 2 >= 4, i.e., k >= 1.
    # So no pair (v1, v2) can agree on all 3 shared coords.
    # This means: the set of sign patterns on S1 restricted to the 3 shared coords, and
    # the set on S2 restricted to the 3 shared coords, must be "orthogonal" in the sense
    # that no pattern from S1 matches any pattern from S2 on those 3 coords.
    # Each support has 2^4 = 16 patterns, which project to 2^3 = 8 patterns on the 3 shared coords.
    # Each projection value appears exactly 2 times (the 4th coord is free).
    # So if we include all 16 patterns from S1, the projection covers all 8 values, each twice.
    # Then S2 cannot include ANY pattern whose projection matches any of those 8 = all of them.
    # So if S1 has all 16, S2 has 0.
    # If S1 has 8 patterns (one per projection value), S2 still can't have any.
    # If S1 has 4 patterns covering 4 distinct projection values, S2 can have patterns
    # whose projection values are among the other 4.
    # This gets complicated. Let's stick with the proven structure but try to improve
    # the independent set search.
    
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)

    # Conflict: two supports conflict if |intersection| >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
                adj[i].add(j)
                adj[j].add(i)

    adj_arrays = [np.array(sorted(adj[i]), dtype=np.int64) for i in range(N_S)]
    adj_counts = np.array([len(adj[i]) for i in range(N_S)], dtype=np.int64)

    # Degree ordering for greedy
    degree_order = np.argsort(adj_counts)  # lowest degree first

    rng = np.random.RandomState(42)

    def compute_blocked(x):
        blocked = np.zeros(N_S, dtype=bool)
        in_set = np.where(x)[0]
        for v in in_set:
            neighbors = adj_arrays[v]
            if len(neighbors) > 0:
                blocked[neighbors] = True
        return blocked

    def greedy_fill(x, blocked):
        changed = True
        while changed:
            changed = False
            candidates = np.where(~x & ~blocked)[0]
            for v in candidates:
                if not x[v] and not blocked[v]:
                    x[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
                    changed = True
        return x, blocked

    def greedy_by_order(order):
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                neighbors = adj_arrays[v]
                if len(neighbors) > 0:
                    blocked[neighbors] = True
        return x

    def local_search(x, max_iter=30000, time_limit=None):
        current_x = x.copy()
        blocked = compute_blocked(current_x)
        current_x, blocked = greedy_fill(current_x, blocked)
        current_score = int(np.sum(current_x))

        best_x = current_x.copy()
        best_score = current_score

        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break

            r = rng.random()

            if r < 0.30:
                # Swap: remove one, add one
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                temp_x = current_x.copy()
                temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) > 0:
                    v = candidates[rng.randint(len(candidates))]
                    current_x[u] = False
                    current_x[v] = True
                    blocked = compute_blocked(current_x)
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()

            elif r < 0.55:
                # Remove 1-2, fill
                in_set = np.where(current_x)[0]
                if len(in_set) < 1:
                    continue
                n_remove = 1 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

            elif r < 0.80:
                # 2-swap: remove 2, add 2
                in_set = np.where(current_x)[0]
                if len(in_set) < 2:
                    continue
                u1_idx, u2_idx = rng.choice(len(in_set), size=2, replace=False)
                u1, u2 = in_set[u1_idx], in_set[u2_idx]
                temp_x = current_x.copy()
                temp_x[u1] = False
                temp_x[u2] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 2:
                    v1_idx, v2_idx = rng.choice(len(candidates), size=2, replace=False)
                    v1, v2 = candidates[v1_idx], candidates[v2_idx]
                    if not conflict[v1, v2]:
                        current_x[u1] = False
                        current_x[u2] = False
                        current_x[v1] = True
                        current_x[v2] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()
            else:
                # Remove 3-5, fill
                in_set = np.where(current_x)[0]
                if len(in_set) < 3:
                    continue
                n_remove = 3 + rng.randint(0, 3)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

        return best_x

    # Branch and bound with greedy coloring bound
    def branch_and_bound(order, time_limit):
        """
        B&B for maximum independent set.
        order: array of vertices in processing order.
        Uses greedy coloring on the remaining subgraph for upper bound.
        """
        n = len(order)
        # Map from position in order to vertex
        # We work with the induced subgraph on the remaining vertices
        
        best_set = set()
        best_size = 0
        
        # Greedy coloring to get upper bound
        def color_bound(vertices):
            """
            Greedy coloring of the subgraph induced by vertices.
            Returns number of colors used (upper bound on MIS size).
            """
            if len(vertices) == 0:
                return 0
            if len(vertices) == 1:
                return 1
            
            # Order vertices by degree within the subgraph (descending)
            v_list = list(vertices)
            degs = []
            for v in v_list:
                d = 0
                for w in v_list:
                    if w != v and conflict[v, w]:
                        d += 1
                degs.append(d)
            
            order_idx = np.argsort(-degs)
            colored = np.zeros(len(v_list), dtype=int)
            color = 0
            for idx in order_idx:
                v = v_list[idx]
                # Find smallest available color
                used = set()
                for w_idx in range(len(v_list)):
                    if w_idx != idx and conflict[v, v_list[w_idx]] and colored[w_idx] > 0:
                        used.add(colored[w_idx])
                c = 1
                while c in used:
                    c += 1
                colored[idx] = c
                color = max(color, c)
            return color
        
        # Recursive B&B
        # Use a stack-based approach to avoid deep recursion
        # State: (remaining_vertices_set, current_set)
        # But this is exponential. Let's use a simpler heuristic B&B.
        
        # Actually, for 330 nodes, full B&B is infeasible.
        # Let's use a greedy + local search hybrid instead.
        # The B&B here is just for initial solution quality.
        
        # Greedy with multiple orderings
        results = []
        
        # Ordering 1: degree
        x1 = greedy_by_order(degree_order)
        results.append(x1)
        
        # Ordering 2: random
        x2 = greedy_by_order(rng.permutation(N_S))
        results.append(x2)
        
        # Ordering 3: reverse degree
        x3 = greedy_by_order(degree_order[::-1])
        results.append(x3)
        
        # Ordering 4: degree + random jitter
        for _ in range(10):
            jittered = adj_counts.astype(float) + rng.random(N_S) * 2
            order = np.argsort(jittered)
            x = greedy_by_order(order)
            results.append(x)
        
        # Take the best
        best_x = max(results, key=lambda x: int(np.sum(x)))
        return best_x

    # Run multiple rounds of local search with diverse initializations
    best_support_set = set()
    best_size = 0

    # Initial solutions from various heuristics
    initial_solutions = []
    
    # Degree-ordered greedy
    x_deg = greedy_by_order(degree_order)
    initial_solutions.append(x_deg)
    
    # Reverse degree
    x_rev = greedy_by_order(degree_order[::-1])
    initial_solutions.append(x_rev)
    
    # Random orderings
    for _ in range(20):
        x_rand = greedy_by_order(rng.permutation(N_S))
        initial_solutions.append(x_rand)
    
    # Degree + jitter
    for _ in range(20):
        jittered = adj_counts.astype(float) + rng.random(N_S) * 3
        order = np.argsort(jittered)
        x_jit = greedy_by_order(order)
        initial_solutions.append(x_jit)
    
    # Take top 15 initial solutions
    initial_solutions.sort(key=lambda x: int(np.sum(x)), reverse=True)
    initial_solutions = initial_solutions[:15]
    
    for x0 in initial_solutions:
        if time.time() > deadline - 5.0:
            break
        
        size0 = int(np.sum(x0))
        if size0 > best_size:
            best_size = size0
            best_support_set = set(np.where(x0)[0].tolist())
        
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
        
        # Allocate time proportionally
        time_per = min(remaining_time / (len(initial_solutions) + 1), 12.0)
        
        x_best = local_search(x0, max_iter=20000, time_limit=time.time() + time_per)
        size = int(np.sum(x_best))
        
        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x_best)[0].tolist())
    
    # Additional local search rounds from the best solution
    if best_size > 0 and time.time() < deadline - 3.0:
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb the best solution
            x_pert = np.zeros(N_S, dtype=bool)
            best_nodes = list(best_support_set)
            for v in best_nodes:
                x_pert[v] = True
            
            # Remove a few random nodes
            n_remove = rng.randint(2, 6)
            if len(best_nodes) > n_remove:
                to_remove = rng.choice(len(best_nodes), size=n_remove, replace=False)
                for idx in to_remove:
                    x_pert[best_nodes[idx]] = False
            
            blocked = compute_blocked(x_pert)
            x_pert, blocked = greedy_fill(x_pert, blocked)
            
            x_best = local_search(x_pert, max_iter=10000, time_limit=deadline - 1.0)
            size = int(np.sum(x_best))
            
            if size > best_size:
                best_size = size
                best_support_set = set(np.where(x_best)[0].tolist())
    
    # Final polish
    if time.time() < deadline - 1.0 and best_support_set:
        polish_x = np.zeros(N_S, dtype=bool)
        for v in best_support_set:
            polish_x[v] = True
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            blocked = compute_blocked(polish_x)
            candidates = np.where(~polish_x & ~blocked)[0]
            for v in candidates:
                if not polish_x[v] and not blocked[v]:
                    polish_x[v] = True
                    changed = True
        final_size = int(np.sum(polish_x))
        if final_size > best_size:
            best_size = final_size
            best_support_set = set(np.where(polish_x)[0].tolist())

    # Build the vector matrix
    vectors = []
    
    # Type A: ±2e_i
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)
    
    # Type D: all 16 sign patterns for each selected support
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    return np.array(vectors, dtype=np.int8)
