import time
import random
import os

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def popcount(x):
        return bin(x).count("1")

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        
        # rem_adj[u] is a bitmask of vertices v such that edge (u,v) is still uncovered
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
            
        remaining_edges = len(edges)
        
        best_partition = None
        best_score = float('inf')
        
        # We will run multiple randomized greedy passes.
        num_restarts = 50 # Increased restarts, relying on bitmask speed
        
        # Global random instance for reproducibility within the instance solve
        rng = random.Random(seed + 12345)
        
        for r in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Copy state
            curr_rem_adj = list(rem_adj)
            curr_edges = remaining_edges
            partition = []
            
            # Active vertices mask
            active_mask = (1 << n) - 1
            
            while curr_edges > 0:
                if time.time() > deadline - 2.0:
                    break
                
                # Find a clique in the remaining graph induced by active vertices
                # Greedy approach:
                # 1. Start with an empty clique
                # 2. Candidates are active vertices
                # 3. Pick vertex v from candidates that maximizes degree in remaining graph (within candidates)
                #    Add randomness: pick from top K or use probability
                
                clique = []
                cand_mask = active_mask
                
                while cand_mask:
                    # Find vertex in cand_mask with max degree in curr_rem_adj (within cand_mask)
                    max_deg = -1
                    best_v = -1
                    candidates_list = []
                    
                    # Iterate through bits in cand_mask
                    m = cand_mask
                    while m:
                        # Get lowest set bit
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        
                        # Degree of v within cand_mask
                        deg = popcount(curr_rem_adj[v] & cand_mask)
                        
                        if deg > 0:
                            if deg > max_deg:
                                max_deg = deg
                                candidates_list = [v]
                            elif deg == max_deg:
                                candidates_list.append(v)
                    
                    if not candidates_list:
                        break
                    
                    # Random selection from top candidates
                    # To encourage diversity, pick randomly from candidates with high degree
                    # Or just pick the best one with some probability
                    if len(candidates_list) > 1 and rng.random() < 0.3:
                        v = rng.choice(candidates_list)
                    else:
                        v = candidates_list[0]
                    
                    clique.append(v)
                    
                    # Update cand_mask: must be neighbor of v in remaining graph
                    cand_mask &= curr_rem_adj[v]
                    # Remove v from cand_mask
                    cand_mask &= ~(1 << v)
                
                if not clique:
                    # Should not happen if curr_edges > 0, but safety check
                    break
                
                # Remove edges covered by this clique
                for i in range(len(clique)):
                    u = clique[i]
                    for j in range(i + 1, len(clique)):
                        w = clique[j]
                        # Check if edge exists in curr_rem_adj
                        if curr_rem_adj[u] & (1 << w):
                            curr_rem_adj[u] &= ~(1 << w)
                            curr_rem_adj[w] &= ~(1 << u)
                            curr_edges -= 1
                            
                partition.append(clique)
                
                # Update active_mask?
                # We don't strictly need to update active_mask if we check cand_mask properly,
                # but vertices with no remaining edges should not be picked as starters.
                # The loop `while cand_mask` handles this because if a vertex has no neighbors 
                # in the remaining graph, it won't be in `cand_mask` if we start from active_mask 
                # and intersect with neighbors? 
                # Wait. The initial `cand_mask` is `active_mask`.
                # If a vertex has degree 0 in remaining graph, it will be in `cand_mask` initially.
                # Then `deg` will be 0. It will be skipped in the candidate selection loop 
                # (since we check `if deg > 0`).
                # So it won't be added to clique.
                # If ALL vertices have deg 0, `candidates_list` is empty, break.
                # But `curr_edges > 0` implies some edge exists, so some vertex has deg > 0.
                # So `clique` will have at least 1 vertex?
                # If clique has 1 vertex, it covers 0 edges.
                # We must ensure clique size >= 2.
                # If the max clique found has size 1, it means no edges are available?
                # No, if there is an edge (u,v), then u and v are neighbors.
                # If we pick u, v is in `cand_mask` (neighbor of u).
                # So clique will have at least 2 vertices if any edge exists.
                
            # Fallback for remaining edges (if any, due to bugs or timeout)
            # In a correct implementation, curr_edges should be 0.
            if curr_edges > 0:
                for u in range(n):
                    mask = curr_rem_adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        mask ^= lsb
                        if u < v:
                            partition.append([u, v])
                            curr_rem_adj[u] &= ~(1 << v)
                            curr_rem_adj[v] &= ~(1 << u)
                            curr_edges -= 1

            score = len(partition)
            if score < best_score:
                best_score = score
                best_partition = partition
                
        if best_partition is None:
            # Fallback: single pass
            curr_rem_adj = list(rem_adj)
            curr_edges = remaining_edges
            partition = []
            active_mask = (1 << n) - 1
            while curr_edges > 0:
                clique = []
                cand_mask = active_mask
                while cand_mask:
                    max_deg = -1
                    best_v = -1
                    m = cand_mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        m ^= lsb
                        deg = popcount(curr_rem_adj[v] & cand_mask)
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                    if best_v == -1:
                        break
                    clique.append(best_v)
                    cand_mask &= curr_rem_adj[best_v]
                    cand_mask &= ~(1 << best_v)
                if not clique:
                    break
                for i in range(len(clique)):
                    u = clique[i]
                    for j in range(i + 1, len(clique)):
                        w = clique[j]
                        if curr_rem_adj[u] & (1 << w):
                            curr_rem_adj[u] &= ~(1 << w)
                            curr_rem_adj[w] &= ~(1 << u)
                            curr_edges -= 1
                partition.append(clique)
            
            for u in range(n):
                mask = curr_rem_adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    mask ^= lsb
                    if u < v:
                        partition.append([u, v])
                        curr_rem_adj[u] &= ~(1 << v)
                        curr_rem_adj[v] &= ~(1 << u)
            best_partition = partition
            
        return best_partition

    result1 = solve_instance(32, 0.6, 601, deadline)
    result2 = solve_instance(64, 0.55, 602, deadline)
    
    return [result1, result2]
