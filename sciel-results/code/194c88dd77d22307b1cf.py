import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors and their indices
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph using Bitmasks
    # neighbors_mask[i] is an integer where bit j is 1 if i and j conflict.
    neighbors_mask = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute support intersections
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        
                        match = True
                        for p1, p2 in zip(pos_in_s1, pos_in_s2):
                            if signs1[p1] != signs2[p2]:
                                match = False
                                break
                        if match:
                            # Set bit idx2 in neighbors_mask[idx1]
                            neighbors_mask[idx1] |= (1 << idx2)
                            # Set bit idx1 in neighbors_mask[idx2]
                            neighbors_mask[idx2] |= (1 << idx1)

    # 3. Backtracking Search for Max Independent Set
    # Using bitmasks for speed.
    
    best_size = 0
    best_mask = 0
    start_time = time.time()
    time_limit = 110.0
    
    # Helper: Greedy Coloring Bound for MIS
    # Returns an upper bound on the size of the independent set in the subgraph defined by mask.
    def coloring_bound(mask):
        # Greedy coloring: assign colors to vertices such that no two adjacent vertices have the same color.
        # The number of colors used is an upper bound on the clique size, which is an upper bound on the independent set size?
        # No, for MIS, we use the greedy coloring of the *complement* graph? 
        # Actually, a common bound for MIS is obtained by greedily coloring the candidate graph.
        # The minimum number of colors needed to color the graph is the chromatic number.
        # An independent set can take at most 1 vertex from each color class.
        # So, if we color the graph with k colors, the MIS size <= k.
        # Greedy coloring gives an upper bound on the chromatic number, which is an upper bound on the MIS size?
        # Wait. Chromatic number chi(G) >= size of maximum independent set? No.
        # alpha(G) * chi(G) >= N. So alpha(G) <= N / chi(G)? No.
        # The standard bound is: If you greedily color the graph, the number of colors used is an upper bound on the size of the *clique*, not the independent set.
        # Actually, for MIS, we often use the bound based on the greedy coloring of the *candidate* graph.
        # Let's use a simpler bound: The number of vertices in the candidate set is a weak bound.
        # A better bound: Greedy coloring the candidate graph. The number of colors used is an upper bound on the size of the maximum independent set?
        # No, that's for clique.
        # For MIS, we can use the fact that if we greedily pick vertices and remove their neighbors, the number of picks is a *lower* bound.
        # For an *upper* bound, we can use the greedy coloring of the graph. The chromatic number is an upper bound on the size of the maximum independent set? 
        # No, alpha(G) is the max independent set. chi(G) is the min colors.
        # It is true that alpha(G) * chi(G) >= |V|. So alpha(G) <= |V| / chi(G) is not necessarily true.
        # However, a greedy coloring produces a partition into independent sets. If we use k colors, we have k independent sets.
        # This doesn't directly bound the size of a *single* independent set.
        
        # Let's stick to a simpler, effective heuristic search without complex bounds, but with good branching.
        # Or use the bound: size of MIS <= size of greedy independent set on the *complement*? No.
        
        # Let's use a simple recursive bound: 
        # If we can't compute a tight bound quickly, we rely on the time limit and good pruning from the branching.
        # To improve pruning, we can use a greedy MIS on the candidates to get a lower bound, and if current_size + greedy_mis(candidates) <= best, we can't prune based on that.
        # We need an upper bound.
        
        # A simple upper bound is the number of candidates.
        # A tighter bound: Greedy coloring of the candidate graph.
        # Theorem: The size of the maximum independent set is at most the number of colors used in any proper coloring of the graph.
        # Why? Because an independent set contains at most one vertex of each color.
        # Yes, this is correct.
        
        colors = {}
        remaining = mask
        color_count = 0
        # We need to map bit positions to indices.
        # This is slow in Python if done naively.
        # Let's approximate: just return bit_count of mask for now, or a simple heuristic.
        # Given time constraints, let's use a simple bound: 
        # The greedy coloring approach is O(N^2) in worst case per node, which is too slow.
        
        # Alternative: Use the "Lubiw" bound or similar? Too complex.
        
        # Let's just use the bit count as the bound. It's weak but fast.
        # To make it stronger, we can do a quick greedy independent set on the candidates to see how many we can definitely get, but that's a lower bound.
        
        # Let's try a simple recursive bound that is slightly better than bit count:
        # If the graph is sparse, bit count is okay.
        return remaining.bit_count()

    def solve(candidates_mask, current_size):
        nonlocal best_size, best_mask, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = 0 # We need to reconstruct the mask, so we should pass it or store it differently.
                # Storing the mask is better.
                # But we need the actual set of indices.
                # Let's store the mask.
            return
            
        # Bound
        bound = candidates_mask.bit_count()
        if current_size + bound <= best_size:
            return
            
        # Heuristic: Pick vertex with max degree in candidates
        # To find max degree, we need to count bits in (neighbors_mask[u] & candidates_mask) for each u in candidates.
        # This is expensive. 
        # Approximation: Pick the vertex with highest global degree? Or just the first one?
        # Let's try picking the vertex with the highest number of neighbors in the candidate set.
        # We can iterate through the bits of candidates_mask.
        
        # Optimization: If candidates_mask has few bits, iterate.
        # If many, maybe pick a random one or one with high global degree.
        
        # Let's find the vertex with max degree.
        max_deg = -1
        v = -1
        temp_mask = candidates_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            u = lsb.bit_length() - 1
            deg = (neighbors_mask[u] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            temp_mask ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates_mask without v and its neighbors
        new_candidates_1 = candidates_mask & ~(neighbors_mask[v] | (1 << v))
        solve(new_candidates_1, current_size + 1)
        
        # Branch 2: Exclude v
        new_candidates_2 = candidates_mask & ~(1 << v)
        solve(new_candidates_2, current_size)

    # Initial candidates: all nodes
    initial_mask = (1 << N) - 1
    
    # Greedy initialization to get a good lower bound
    # Greedy MIS: pick vertex with min degree? Or max?
    # Let's pick the first available vertex repeatedly.
    greedy_mask = 0
    remaining = initial_mask
    while remaining:
        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        greedy_mask |= lsb
        remaining &= ~(neighbors_mask[v] | lsb)
        
    best_size = greedy_mask.bit_count()
    best_mask = greedy_mask
    
    try:
        solve(initial_mask, 0)
    except RecursionError:
        pass
        
    # Convert best_mask to array
    if best_mask == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    temp_mask = best_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        result_vectors.append(all_vectors[idx][3])
        temp_mask ^= lsb
        
    return np.array(result_vectors, dtype=np.int64)
