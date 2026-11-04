import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is 3
    # We represent the graph using bitmasks of neighbors for each node
    # adj_mask[i] is a bitmask of all nodes j such that i and j conflict
    
    adj_mask = [0] * N_S
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            # Check intersection size
            # masks[i] and masks[j] are 11-bit integers
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # 3. Maximum Independent Set using Branch and Bound
    # We use a recursive search with pruning.
    # To make it fast, we use bitmasks for the set of available candidates.
    
    # Order vertices by degree (descending) to improve pruning? 
    # Or use a static order. 
    # Let's try a simple recursive B&B.
    
    best_size = 0
    best_set_mask = 0  # Bitmask of the best independent set found
    
    # Greedy initialization to get a good lower bound
    def greedy_init():
        remaining = (1 << N_S) - 1
        s_mask = 0
        # Greedy: pick node with min degree in remaining? 
        # For simplicity, pick first available, remove it and its neighbors.
        # This is fast but maybe weak. Let's do a few random restarts of greedy.
        for _ in range(100):
            r_mask = 0
            rem = remaining
            # Randomize order by shuffling bits? Hard with big ints.
            # Just pick random available node.
            while rem:
                # Pick a random set bit
                # Get lowest set bit
                lsb = rem & -rem
                # To pick random, we can shift and pick?
                # Simpler: iterate through bits and pick one randomly from available?
                # For speed, just pick the lowest set bit (deterministic greedy) 
                # but vary by rotating the mask?
                # Let's just do deterministic greedy for now, then B&B.
                v_bit = lsb
                r_mask |= v_bit
                # Remove v and its neighbors from rem
                # neighbors of v are in adj_mask[v_index]
                v_idx = v_bit.bit_length() - 1
                rem &= ~(adj_mask[v_idx] | v_bit)
            if r_mask.bit_count() > best_size:
                return r_mask
        return 0

    # Initial lower bound
    init_set = greedy_init()
    best_size = init_set.bit_count()
    best_set_mask = init_set
    
    # Branch and Bound
    # State: (candidates_mask, current_set_mask, current_size)
    # Pruning: if current_size + popcount(candidates_mask) <= best_size, return
    
    # To speed up, we can use a coloring bound.
    # But for 330 nodes, simple B&B with good ordering might suffice if we have a strong lower bound.
    
    # Let's use a recursive function with a time check.
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline - 1.0:
            return
            
        # Pruning
        # Upper bound: current_size + number of candidates
        # A tighter bound: greedy coloring of the candidate graph
        # For now, use popcount
        if current_size + candidates.bit_count() <= best_size:
            return
            
        # If no candidates, update best
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_set
            return
            
        # Select a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates?
        # Or just the first one.
        # Let's pick the vertex with the highest degree in the induced subgraph of candidates.
        # This is expensive to compute dynamically.
        # Alternative: use a static order.
        # Let's try picking the lowest set bit for speed, but this might be slow.
        # Better: pick a vertex that conflicts with many candidates.
        
        # To keep it simple and fast, we pick the first available vertex.
        v_bit = candidates & -candidates
        v_idx = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: candidates that are not v and not neighbors of v
        new_candidates = candidates & ~(adj_mask[v_idx] | v_bit)
        branch_and_bound(new_candidates, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude v
        # New candidates: candidates without v
        branch_and_bound(candidates & ~v_bit, current_set, current_size)

    # The simple B&B above might be too slow without a good coloring bound.
    # Let's implement a greedy coloring bound.
    
    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the size of MIS in the graph induced by candidates.
        Uses greedy coloring.
        """
        # We need to color the graph.
        # Colors are assigned such that adjacent nodes have different colors.
        # The number of colors is an upper bound on the size of the independent set?
        # No, the size of the largest color class is a lower bound on the chromatic number?
        # Wait. The chromatic number $\chi(G) \ge \omega(G)$ (clique number).
        # And $\alpha(G) \le |V| / \chi(G)$? No.
        # Actually, if we color the graph with $k$ colors, the maximum independent set size is at most the sum of the sizes of the largest color class? No.
        # The standard bound is: $\alpha(G) \le \sum_{i=1}^k |C_i|$? No.
        # The bound is: $\alpha(G) \le$ number of colors used in a greedy coloring? 
        # Yes, if we greedily color the graph, the number of colors used is an upper bound on the clique number, but for MIS?
        # Actually, the size of the MIS is at most the number of colors used in *any* valid coloring? No.
        # Example: Triangle. 3 colors. MIS size 1. 1 <= 3.
        # Example: Path 3 nodes. 2 colors. MIS size 2. 2 <= 2.
        # It is a known result that the size of the maximum independent set is at most the number of colors used in a greedy coloring of the graph?
        # No, that's not right. 
        # The bound is: $\alpha(G) \le$ the number of colors in a *minimal* coloring?
        # Let's use a different bound: The size of the MIS is at most the number of vertices, obviously.
        # A better bound: Greedily pick a vertex, add to bound, remove it and neighbors. This gives a *lower* bound (an independent set).
        # For an *upper* bound, we can use: $\alpha(G) \le \sum_{v \in V} \frac{1}{d(v)+1}$? No.
        
        # Let's stick to the simple bound: current_size + candidates.bit_count().
        # To improve performance, we can order the branching.
        
        return candidates.bit_count()

    # Since simple B&B might be slow, let's use the heuristic from Inspiration 1 (Local Search) 
    # but with bitmasks for speed, and run it for a longer time.
    # The "structural" change is the use of bitmasks for the graph representation and search.
    
    # Let's combine: Run a fast bitmask-based local search for a while, 
    # then if time remains, try B&B with the best found so far as lower bound.
    
    def local_search_bitmask(init_mask):
        current_mask = init_mask
        best_local_mask = init_mask
        best_local_size = init_mask.bit_count()
        
        # Precompute neighbor masks
        # We have adj_mask
        
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
                
            # Find candidates: not in current, not neighbor of any in current
            # blocked = OR of adj_mask[v] for all v in current
            blocked = 0
            temp = current_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                blocked |= adj_mask[idx]
                temp ^= lsb
            
            candidates = ((1 << N_S) - 1) & ~current_mask & ~blocked
            
            if candidates:
                # Pick a random candidate
                # To pick random bit from candidates:
                # Get list of set bits? Too slow.
                # Just pick the lowest set bit for now (deterministic) or use a random mask?
                # Let's pick the lowest set bit.
                v_bit = candidates & -candidates
                current_mask |= v_bit
                new_size = current_mask.bit_count()
                if new_size > best_local_size:
                    best_local_size = new_size
                    best_local_mask = current_mask
                continue
            
            # Stuck: try 1-swap
            # Pick a random vertex in current to remove
            # Then find a candidate to add
            if current_mask:
                # Pick random vertex from current
                # We can just pick the lowest set bit
                v_remove_bit = current_mask & -current_mask
                current_mask ^= v_remove_bit
                
                # Recompute blocked
                blocked = 0
                temp = current_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    blocked |= adj_mask[idx]
                    temp ^= lsb
                
                candidates = ((1 << N_S) - 1) & ~current_mask & ~blocked
                
                if candidates:
                    v_add_bit = candidates & -candidates
                    current_mask |= v_add_bit
                    new_size = current_mask.bit_count()
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local_mask = current_mask
                else:
                    # Swap failed, put it back
                    current_mask |= v_remove_bit
            
        return best_local_mask, best_local_size

    # Run multiple restarts of local search
    rng = np.random.RandomState(42)
    
    # Initial greedy sets
    # We can generate a few random greedy sets
    for r in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Randomized Greedy
        # Shuffle the order of vertices by using a random mask?
        # Simpler: Just run greedy on the original graph, but the "randomness" comes from 
        # the fact that we might start with different subsets?
        # Actually, let's just run the local search on the greedy_init result.
        
        init_mask = greedy_init()
        if init_mask == 0:
            continue
            
        final_mask, final_size = local_search_bitmask(init_mask)
        
        if final_size > best_size:
            best_size = final_size
            best_set_mask = final_mask

    # 4. Construct vectors
    vectors = []
    # Iterate through bits in best_set_mask
    temp = best_set_mask
    while temp:
        lsb = temp & -temp
        s_idx = lsb.bit_length() - 1
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
        temp ^= lsb

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
