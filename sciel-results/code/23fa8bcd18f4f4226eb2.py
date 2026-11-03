import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all 4-coordinate vectors with ±1 entries
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280
    all_vectors = []
    support_to_vectors = {}
    
    for support in itertools.combinations(range(D), 4):
        support_to_vectors[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors.append(v)
            support_to_vectors[support].append(len(all_vectors) - 1)
    
    N = len(all_vectors)
    vectors_arr = np.array(all_vectors, dtype=np.int8)
    
    # Build conflict graph: two vectors conflict if their dot product > 2
    # (since for 4-sparse ±1 vectors, need dot <= 2 for distance >= norm)
    # Actually: distance^2 = 8 - 2*(dot), need distance^2 >= 4, so 8 - 2*dot >= 4, dot <= 2
    
    # Precompute: for each vector, which other vectors conflict with it
    # Use the fact that conflicts depend on support overlap and sign agreement
    
    # Two vectors conflict iff:
    # - They share >= 3 coordinates AND
    # - On the shared coordinates, the number of matching signs > number of mismatching signs
    #   (i.e., dot product on shared coords >= 3)
    
    # For 4-sparse vectors:
    # - Share 4 coords (same support): conflict if dot >= 3, i.e., dot = 4 (all match)
    # - Share 3 coords: conflict if dot on those 3 >= 3, i.e., all 3 match
    # - Share <= 2 coords: never conflict
    
    # Build adjacency using support grouping
    adj = [set() for _ in range(N)]
    
    # For each pair of supports that share >= 3 coordinates, check all vector pairs
    supports_list = list(support_to_vectors.keys())
    N_sup = len(supports_list)
    
    # Precompute which support pairs share >= 3
    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            s1 = set(supports_list[i])
            s2 = set(supports_list[j])
            inter = s1 & s2
            if len(inter) >= 3:
                # For each vector in support i and each in support j, check conflict
                for vi in support_to_vectors[supports_list[i]]:
                    for vj in support_to_vectors[supports_list[j]]:
                        # Compute dot product on intersection
                        v1 = vectors_arr[vi]
                        v2 = vectors_arr[vj]
                        dot = int(np.dot(v1, v2))
                        if dot > 2:
                            adj[vi].add(vj)
                            adj[vj].add(vi)
        
        # Also check within same support (share 4 coords)
    for support in supports_list:
        vecs_in_support = support_to_vectors[support]
        for a in range(len(vecs_in_support)):
            for b in range(a + 1, len(vecs_in_support)):
                vi = vecs_in_support[a]
                vj = vecs_in_support[b]
                dot = int(np.dot(vectors_arr[vi], vectors_arr[vj]))
                if dot > 2:
                    adj[vi].add(vj)
                    adj[vj].add(vi)
    
    # Find large independent set using simulated annealing
    rng = np.random.RandomState(42)
    
    best_set = set()
    best_size = 0
    
    # Initial solution: greedy
    def greedy_init():
        avail = set(range(N))
        s = set()
        while avail:
            # Pick node with min degree in avail
            min_deg = float('inf')
            min_node = None
            for v in avail:
                deg = len(adj[v] & avail)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            avail -= adj[min_node]
            avail.discard(min_node)
        return s
    
    init_set = greedy_init()
    best_size = len(init_set)
    best_set = set(init_set)
    
    # Simulated annealing
    current_set = set(init_set)
    current_size = len(current_set)
    
    in_set = np.zeros(N, dtype=bool)
    for v in current_set:
        in_set[v] = True
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline - 5.0:
        steps += 1
        
        # Random move: add, remove, or swap
        move_type = rng.randint(3)
        
        if move_type == 0:  # Try to add
            # Find nodes not in set and not blocked
            blocked = np.zeros(N, dtype=bool)
            for v in current_set:
                for nb in adj[v]:
                    blocked[nb] = True
            
            candidates = np.where(~in_set & ~blocked)[0]
            if len(candidates) == 0:
                continue
            v_add = candidates[rng.randint(len(candidates))]
            
            new_size = current_size + 1
            delta = 1  # Always beneficial
            
            # Accept if beneficial or with probability exp(-delta/T) for harmful
            if delta > 0 or rng.random() < np.exp(-delta / T):
                current_set.add(v_add)
                in_set[v_add] = True
                current_size = new_size
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
        
        elif move_type == 1:  # Try to remove
            if len(current_set) == 0:
                continue
            v_rem = list(current_set)[rng.randint(len(current_set))]
            
            new_size = current_size - 1
            delta = -1  # Harmful
            
            if rng.random() < np.exp(-delta / T):
                current_set.remove(v_rem)
                in_set[v_rem] = False
                current_size = new_size
        
        else:  # Swap: remove one, add one
            if len(current_set) == 0:
                continue
            
            v_rem = list(current_set)[rng.randint(len(current_set))]
            current_set.remove(v_rem)
            in_set[v_rem] = False
            
            # Recompute blocked
            blocked = np.zeros(N, dtype=bool)
            for v in current_set:
                for nb in adj[v]:
                    blocked[nb] = True
            
            candidates = np.where(~in_set & ~blocked)[0]
            if len(candidates) > 0:
                v_add = candidates[rng.randint(len(candidates))]
                current_set.add(v_add)
                in_set[v_add] = True
                # Size unchanged, delta = 0
                # Always accept for exploration
            else:
                # Put it back
                current_set.add(v_rem)
                in_set[v_rem] = True
            
            T *= cooling
        
        T = max(T_min, T * cooling)
    
    # Final: try to add more greedily
    blocked = np.zeros(N, dtype=bool)
    for v in best_set:
        for nb in adj[v]:
            blocked[nb] = True
    
    in_best = np.zeros(N, dtype=bool)
    for v in best_set:
        in_best[v] = True
    
    candidates = np.where(~in_best & ~blocked)[0]
    for v in candidates:
        best_set.add(v)
        for nb in adj[v]:
            blocked[nb] = True
        in_best[v] = True
    
    final_vectors = vectors_arr[list(best_set)]
    
    return final_vectors

