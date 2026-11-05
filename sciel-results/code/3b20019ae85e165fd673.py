import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init():
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    # Maximum matching on induced subgraph for Lovasz bound
    # Returns size of maximum matching in graph G[P]
    def max_matching(P):
        # Use Edmonds' blossom algorithm for maximum matching
        # For small graphs, we can use a simpler approach
        # Since N is at most 256, and we're looking for matching in induced subgraph
        
        # Extract vertices from P
        vertices = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        n = len(vertices)
        if n <= 1:
            return 0
        
        # Map vertex index to position in vertices list
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        
        # Build adjacency list for induced subgraph
        adj_list = [[] for _ in range(n)]
        for i, u in enumerate(vertices):
            for j, w in enumerate(vertices):
                if i != j and (adj[u] & (1 << w)):
                    adj_list[i].append(j)
        
        # Use greedy matching with augmenting paths
        match = [-1] * n  # match[i] = j means i is matched to j
        
        def find_augmenting_path(start):
            # BFS to find augmenting path from start
            parent = [-1] * n
            visited = [False] * n
            visited[start] = True
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        if match[v] == -1:
                            # Found augmenting path
                            return v
                        else:
                            queue.append(match[v])
            
            return -1
        
        # Simple augmenting path algorithm (not Edmonds' blossom, but works for many cases)
        # For bipartite-like structures this works well
        # For general graphs, we need blossom, but let's try this first
        
        matching_size = 0
        for u in range(n):
            if match[u] == -1:
                v = find_augmenting_path(u)
                if v != -1:
                    # Augment
                    while v != -1:
                        u = parent[v]
                        next_v = match[u]
                        match[u] = v
                        match[v] = u
                        v = next_v
                    matching_size += 1
        
        return matching_size

    # Lovasz bound: alpha(G[P]) <= |P| - |max_matching(G[P])|
    # Actually, the bound is: alpha(G) >= |V| - |max_matching(G)| is not correct
    # The correct Lovasz bound for independent set:
    # alpha(G) <= |V| - |M| where M is a maximum matching? No.
    # 
    # Actually, for any matching M, alpha(G) >= |V| - |M| is a lower bound.
    # The upper bound from Lovasz is more complex.
    # 
    # Let me use a simpler bound: 
    # The size of maximum independent set is at most |P| (trivial)
    # Or use: alpha(G) <= sum of (1/deg(v) + 1) for v in P (weak)
    # 
    # Actually, let's use the fact that in any graph, 
    # alpha(G) * omega(G) >= |V| where omega is clique number
    # But computing clique number is hard.
    # 
    # Let's use a greedy bound: color the graph and use max color class size? No.
    # 
    # Simple effective bound: 
    # For each vertex v in P, we can include v or exclude all its neighbors.
    # This gives a recursive bound.
    # 
    # Let's implement a simple B&B with the bound |P| and good branching.
    
    def branch_and_bound_mis(P, current_set, best_size, best_set, deadline):
        # P: bitmask of available vertices
        # current_set: bitmask of selected vertices so far
        # best_size: current best size
        # best_set: current best set
        # deadline: time limit
        
        if time.time() > deadline:
            return best_size, best_set
        
        # Bound: |current_set| + |P|
        if current_set.bit_count() + P.bit_count() <= best_size:
            return best_size, best_set
        
        if P == 0:
            if current_set.bit_count() > best_size:
                best_size = current_set.bit_count()
                best_set = current_set
            return best_size, best_set
        
        # Pick a vertex to branch on
        # Choose vertex with maximum degree in P for better pruning
        m = P
        best_v = -1
        max_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (adj[v] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            return best_size, best_set
        
        # Branch 1: Include best_v
        # Remove best_v and its neighbors from P
        new_P1 = P & ~((1 << best_v) | adj[best_v])
        new_set1 = current_set | (1 << best_v)
        best_size, best_set = branch_and_bound_mis(new_P1, new_set1, best_size, best_set, deadline)
        
        # Branch 2: Exclude best_v
        new_P2 = P & ~(1 << best_v)
        best_size, best_set = branch_and_bound_mis(new_P2, current_set, best_size, best_set, deadline)
        
        return best_size, best_set

    # Main: Use B&B with time limit
    best_size = 0
    best_set = 0
    
    # Initial greedy solution
    mask = greedy_init()
    mask = local_improve(mask)
    best_size = mask.bit_count()
    best_set = mask
    
    # Try B&B with increasing time limits
    # First, try with a short time limit to get a good initial bound
    bb_deadline = start_time + 10.0
    bb_size, bb_set = branch_and_bound_mis(cand_mask, 0, best_size, best_set, bb_deadline)
    if bb_size > best_size:
        best_size = bb_size
        best_set = bb_set
    
    # Use remaining time for more B&B or local search
    remaining_deadline = start_time + DEADLINE - 2.0  # 2 second safety margin
    
    # Continue B&B if we have time
    if time.time() < remaining_deadline:
        bb_size, bb_set = branch_and_bound_mis(cand_mask, 0, best_size, best_set, remaining_deadline)
        if bb_size > best_size:
            best_size = bb_size
            best_set = bb_set
    
    # Final local search polish
    if time.time() < remaining_deadline:
        final_mask = local_improve(best_set)
        if final_mask.bit_count() > best_size:
            best_size = final_mask.bit_count()
            best_set = final_mask
    
    # Convert best_set to list of values (1-based)
    result = []
    m = best_set
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
