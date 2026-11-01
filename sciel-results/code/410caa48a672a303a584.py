"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Generate W4 supports
    all_w4_supports = list(itertools.combinations(range(D), 4))
    n_w4 = len(all_w4_supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(all_w4_supports)}
    
    # 2. Build Conflict Graph for W4 supports
    # Conflict if intersection >= 3
    # We want Independent Set, so edges connect compatible supports?
    # No, standard MIS: nodes are supports, edges connect INCOMPATIBLE pairs.
    # We want to select nodes with NO edges between them.
    
    # Precompute intersections for speed
    # 330 nodes is small, O(N^2) is fine
    conflicts = [set() for _ in range(n_w4)]
    for i in range(n_w4):
        si = set(all_w4_supports[i])
        for j in range(i + 1, n_w4):
            sj = set(all_w4_supports[j])
            if len(si & sj) >= 3:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # 3. Solve Max Independent Set on this graph
    # Use a greedy + local search approach, enhanced
    rng = np.random.default_rng(42)
    
    best_set = set()
    best_size = 0
    
    def get_degree(i, current_set):
        return len([x for x in conflicts[i] if x in current_set])
    
    def greedy_init():
        s = set()
        alive = set(range(n_w4))
        while alive:
            # Pick node with min degree in induced subgraph of alive
            min_deg = float('inf')
            min_node = None
            for i in alive:
                deg = len([x for x in conflicts[i] if x in alive])
                if deg < min_deg:
                    min_deg = deg
                    min_node = i
            if min_node is None:
                break
            s.add(min_node)
            alive.discard(min_node)
            for n in list(conflicts[min_node]):
                if n in alive:
                    alive.discard(n)
        return s

    def local_search(s, time_limit=10.0):
        current_set = set(s)
        current_size = len(current_set)
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Move: Remove 1 node from set, Add 1 node not in set
            # Or Add 1 node, Remove its conflicts
            
            # Try adding a node not in set
            candidates = [i for i in range(n_w4) if i not in current_set]
            if not candidates:
                break
            
            # Score candidates by number of conflicts in current_set
            scored = []
            for i in candidates:
                k = len([x for x in conflicts[i] if x in current_set])
                scored.append((k, i))
            
            scored.sort()
            
            # Pick from top few
            top_k = min(10, len(scored))
            choice = rng.integers(top_k)
            k, i = scored[choice]
            
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                # Remove conflicts
                to_remove = [x for x in conflicts[i] if x in current_set]
                for x in to_remove:
                    current_set.remove(x)
                current_set.add(i)
                current_size += delta
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return current_set, current_size

    # Run multiple restarts
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 20 and n_restarts < max_restarts:
        n_restarts += 1
        init_set = greedy_init()
        if len(init_set) > best_size:
            best_size = len(init_set)
            best_set = set(init_set)
            
        remaining = deadline - 20 - time.time()
        if remaining < 2:
            break
        sa_time = min(remaining * 0.5, 10.0)
        if sa_time < 1:
            sa_time = 1.0
            
        final_set, final_size = local_search(init_set, sa_time)
        if final_size > best_size:
            best_size = final_size
            best_set = set(final_set)

    # 4. Determine valid W2 supports
    # W2 support T is valid if T is NOT a subset of any S in best_set
    # And we need to select a matching from valid W2 supports
    
    all_w2_supports = list(itertools.combinations(range(D), 2))
    
    forbidden_w2 = set()
    for s in best_set:
        S = all_w4_supports[s]
        for t in itertools.combinations(S, 2):
            forbidden_w2.add(t)
            
    valid_w2_supports = [t for t in all_w2_supports if t not in forbidden_w2]
    
    # Find Max Matching in valid_w2_supports
    # Greedy matching
    matched_nodes = set()
    selected_w2 = []
    
    # Sort by degree in the "conflict" sense?
    # Two W2 supports conflict if they share a node.
    # We want a set of W2 supports that are pairwise disjoint.
    
    # Simple greedy: pick one, remove its nodes, repeat
    # To be better, sort by "constraint" (number of other supports sharing nodes)
    
    w2_degrees = {}
    for t in valid_w2_supports:
        deg = 0
        for u in valid_w2_supports:
            if t != u and (t[0] in u or t[1] in u):
                deg += 1
        w2_degrees[t] = deg
        
    sorted_w2 = sorted(valid_w2_supports, key=lambda x: w2_degrees[x])
    
    for t in sorted_w2:
        if t[0] in matched_nodes or t[1] in matched_nodes:
            continue
        selected_w2.append(t)
        matched_nodes.add(t[0])
        matched_nodes.add(t[1])
        
    # 5. Construct Vectors
    vectors = []
    
    # Add W4 vectors
    for s in best_set:
        support = all_w4_supports[s]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(support, signs):
                v[pos] = sign
            vectors.append(v)
            
    # Add W2 vectors
    for t in selected_w2:
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, sign in zip(t, signs):
                v[pos] = sign
            vectors.append(v)
            
    # Add W1 vectors (All 22)
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            vectors.append(v)
            
    result = np.array(vectors, dtype=np.int64)
    
    # 6. Verification
    if len(result) > 0:
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check a sample of pairs
        if len(result) > 50:
            valid = True
            for _ in range(1000):
                i, j = rng.integers(0, len(result), size=2)
                if i == j: continue
                dist_sq = np.sum((result[i] - result[j])**2)
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            if not valid:
                # Fallback to just W1
                return np.array([[2 if i==p else 0 for i in range(D)] for p in range(D)] + 
                                [[-2 if i==p else 0 for i in range(D)] for p in range(D)], dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
