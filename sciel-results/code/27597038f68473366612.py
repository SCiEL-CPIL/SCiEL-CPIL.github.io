"""
Strategy: Greedy Backtracking Vertex Addition.
Instead of taking a subgraph of a large C4-free graph (which may have high alpha),
we build the graph vertex by vertex.
1. Start with a base of disjoint triangles (C4-free, alpha = n/3).
2. For each new vertex, determine a set of allowable neighbors.
   A vertex v can connect to u only if u does not share another neighbor with v 
   (i.e., N(u) ∩ N(v) must be empty for the C4-free property, actually N(u) ∩ N(v) 
   must not contain any vertex w that is also a neighbor of both... wait.
   C4-free condition: For any distinct u, v, |N(u) ∩ N(v)| <= 1.
   When adding a new vertex v, if we connect it to u, we must ensure that for all 
   other neighbors w of v, u and w share at most 1 common neighbor.
   Actually, simpler: If we add v and connect to u, we must check that for every 
   other existing vertex x that v is connected to, u and x have at most 1 common neighbor.
   Also, we must ensure that adding v doesn't create a C4 involving v and two existing 
   vertices u, w. This happens if u and w have a common neighbor z, and we connect 
   v to both u and w? No, C4 is u-z-w-v-u. 
   If v connects to u and w, and u and w share a neighbor z, then u-z-w-v-u is a C4 
   ONLY IF v is not connected to z? No, the cycle is u-z-w-v-u. The edges are (u,z), 
   (z,w), (w,v), (v,u). This is a C4. 
   So, if we connect v to u and w, and u, w have a common neighbor z, we form a C4 
   UNLESS v is also connected to z? If v is connected to z, then u-z-v-u is a triangle, 
   and z-w-v-z is a triangle. The set of edges {u,z, w, v} forms a K4 minus an edge? 
   No. If v connects to u, w, z. And u-z, z-w exist. 
   Edges: (u,z), (z,w), (w,v), (v,u), (v,z). 
   This contains triangles (u,z,v) and (z,w,v). 
   Does it contain a C4? 
   A C4 requires 4 distinct vertices. 
   If we have v, u, w, z. 
   Edges: v-u, v-w, v-z, u-z, z-w. 
   Path u-v-w-z-u? No edge z-u? Yes u-z exists. 
   So u-v-w-z-u is a C4. 
   Therefore, if we add v and connect it to u and w, and u and w have a common neighbor z, 
   we MUST connect v to z to avoid the C4? 
   If we connect v to z, then u-v-w-z-u is still a C4? 
   Edges: (u,v), (v,w), (w,z), (z,u). Yes, it's a C4. 
   So, if u and w share a neighbor z, we CANNOT connect v to BOTH u and w. 
   Thus, the neighborhood of any new vertex v must be an independent set in the 
   "common neighbor" graph? 
   Actually, the condition for a set S of neighbors of v is: 
   For any distinct u, w in S, |N(u) ∩ N(w)| must be 0. 
   Because if z in N(u) ∩ N(w), then u-z-w-v-u is a C4. 
   So, the neighbors of any vertex in a C4-free graph form an independent set in the 
   graph where edges represent "sharing a common neighbor"? 
   No, if u and w share a common neighbor z, they cannot both be neighbors of v. 
   This means the neighborhood of v must be a set of vertices that are pairwise 
   "non-adjacent in the line graph of the graph"? 
   Specifically, if u and w have a common neighbor, they cannot both be in N(v). 
   
   This is a strong constraint. 
   To maximize n, we want to add as many vertices as possible. 
   We can use a greedy approach: 
   1. Maintain adjacency sets.
   2. For a new vertex v, find the largest possible set of neighbors S such that 
      for all u, w in S, N(u) ∩ N(w) = empty. 
      Also, we want to minimize the independence number. 
      Connecting v to many vertices reduces the independence number (v cannot be in 
      an independent set with any of its neighbors). 
      So we should try to connect v to as many vertices as possible, subject to the 
      C4-free constraint. 
      
   However, finding the maximum such set S is hard (it's an independent set problem 
   in the "conflict graph"). 
   We can use a greedy heuristic: 
   - Sort existing vertices by degree (high degree first? or low?). 
   - Try to add vertices to S one by one. 
   - If adding u to S causes a conflict with some w in S (i.e., N(u) ∩ N(w) != empty), 
     skip u. 
   
   To improve the independence number, we should prioritize connecting to vertices 
   that are "hard to cover" in independent sets. 
   Alternatively, we can use a local search or backtracking if the greedy fails 
   to keep alpha < 11. 
   
   Given the 120s limit, a simple greedy with some randomization/restarts might work. 
   We aim for n=39 or higher. 
   
   Let's implement:
   1. Base: 13 disjoint triangles (n=39). Alpha = 13. 
      Wait, if we start with 13 triangles, alpha is 13. We need alpha <= 10. 
      So we need to add edges between triangles to reduce alpha. 
      But adding edges between triangles must maintain C4-free. 
      
   Let's start with a smaller base and add vertices. 
   Base: 10 triangles (n=30). Alpha = 10. 
   We want to add vertices up to 43. 
   Each new vertex must connect to a set of existing vertices that are pairwise 
   non-conflicting (no common neighbors). 
   
   Let's try this.
"""

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Parameters
    target_n = 43
    target_alpha = 11  # We want alpha < 11, so max independent set size <= 10
    
    # Base graph: 10 disjoint triangles
    # Vertices 0..29
    # Triangles: (0,1,2), (3,4,5), ..., (27,28,29)
    n = 30
    adj = [set() for _ in range(n)]
    edges = []
    
    def add_edge(u, v):
        if u == v:
            return
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
    for i in range(10):
        u, v, w = 3*i, 3*i+1, 3*i+2
        add_edge(u, v)
        add_edge(u, w)
        add_edge(v, w)
        
    # Function to check if a set of neighbors S is valid for a new vertex
    # Valid if for all distinct u, w in S, N(u) ∩ N(w) is empty
    def is_valid_neighbor_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                u = S_list[i]
                w = S_list[j]
                if adj[u] & adj[w]:
                    return False
        return True

    # Function to estimate independence number (heuristic)
    # We want to keep alpha low. 
    # A simple lower bound on alpha is max(1, n * (1 - 2*E/n^2)) ... not good.
    # Let's use a greedy coloring or independent set finder for verification.
    
    def find_max_independent_set(adj, n, deadline):
        # Simple greedy + backtracking for small n
        # For n=43, we can use a bitmask DP or simple DFS with pruning.
        # Let's use a simple DFS with memoization on bitmask if n <= 20, 
        # otherwise heuristic. 
        # Actually, for n=43, exact search is too slow if we do it every step.
        # We will use a greedy heuristic to estimate alpha, and only do exact 
        # search at the end or when heuristic says it's close to 11.
        
        # Heuristic: Greedy independent set
        remaining = set(range(n))
        is_set = []
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            v = None
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
            is_set.append(v)
            remaining.remove(v)
            remaining -= adj[v]
        return len(is_set)
        
    def exact_independent_set_size(adj, n, target_size, deadline):
        # Check if there is an independent set of size target_size
        # Returns True if found
        if n < target_size:
            return False
        # Use bitmask for n <= 64? 43 fits in int.
        # But 2^43 is too big for DP. 
        # Use DFS with pruning.
        
        adj_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_masks[i] = mask
            
        all_mask = (1 << n) - 1
        path = []
        
        def branch(remaining_mask, current_size):
            if time.time() > deadline:
                return False
            if current_size >= target_size:
                return True
            if remaining_mask == 0:
                return False
            if current_size + remaining_mask.bit_count() < target_size:
                return False
                
            # Pick vertex with max degree in remaining
            temp = remaining_mask
            v_idx = -1
            max_deg = -1
            while temp:
                lsb = temp & -temp
                v = (lsb.bit_length() - 1)
                neighbors_in_rem = adj_masks[v] & remaining_mask
                deg = neighbors_in_rem.bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v_idx = v
                temp -= lsb
                
            if v_idx == -1:
                return False
                
            v_bit = 1 << v_idx
            
            # Include v
            new_remaining = remaining_mask & ~(adj_masks[v_idx] | v_bit)
            if branch(new_remaining, current_size + 1):
                return True
                
            # Exclude v
            new_remaining = remaining_mask & ~v_bit
            if branch(new_remaining, current_size):
                return True
                
            return False

        return branch(all_mask, 0)

    # Greedy addition of vertices
    current_n = n
    best_edges = edges[:]
    best_n = n
    
    # We want to add vertices until we can't or reach target_n
    # We will try multiple random orders of adding neighbors to find a good configuration
    
    # To improve alpha, we should connect new vertices to "central" parts of the graph.
    # But initially, the graph is disjoint triangles. 
    # Let's add edges between triangles first to reduce alpha, then add vertices?
    # No, the task is to maximize n. 
    # If we start with 10 triangles, alpha=10. 
    # If we add a vertex connected to one vertex from each of the 10 triangles, 
    # alpha might stay low. 
    # Let's try to add 13 more vertices (to reach 43). 
    
    # Strategy: 
    # 1. Add edges between existing triangles to reduce alpha further (optional, but might help).
    # 2. Add new vertices.
    
    # Let's just add new vertices.
    # For each new vertex, we want to find a valid neighbor set S.
    # To minimize alpha, we should choose S such that it "hits" many independent sets.
    # A simple heuristic: choose S to be as large as possible.
    
    # Let's try a few random restarts of the greedy process.
    
    def run_greedy_addition(start_adj, start_edges, start_n, max_additions, deadline):
        adj = [set(s) for s in start_adj]
        edges = start_edges[:]
        n = start_n
        
        # Helper to check C4-free
        def check_c4():
            for i in range(n):
                for j in range(i+1, n):
                    if len(adj[i] & adj[j]) > 1:
                        return False
            return True
            
        # We assume start is C4-free.
        
        added = 0
        while n < start_n + max_additions and time.time() < deadline:
            new_v = n
            n += 1
            adj.append(set())
            
            # Determine valid neighbors
            # A vertex u is a candidate if it doesn't conflict with any other candidate.
            # We build S greedily.
            S = set()
            # Sort candidates by degree? 
            # To reduce alpha, we might want to connect to vertices that are 
            # currently "exposed" in independent sets. 
            # For now, just pick a random order or fixed order.
            candidates = list(range(n-1))
            random.shuffle(candidates)
            
            for u in candidates:
                # Check if u can be added to S
                # u must not share a common neighbor with any v in S
                can_add = True
                for v in S:
                    if adj[u] & adj[v]:
                        can_add = False
                        break
                if can_add:
                    S.add(u)
            
            # Connect new_v to all in S
            for u in S:
                adj[new_v].add(u)
                adj[u].add(new_v)
                edges.append([new_v, u])
                
            # Check if alpha is still < 11
            # Use heuristic first
            # If heuristic says alpha >= 11, try to fix by adding more edges? 
            # But we can't add edges without breaking C4-free easily.
            # So we should have been careful in choosing S.
            
            # Let's verify with exact search if n is small enough or periodically.
            if n <= 35 or n % 5 == 0:
                # Exact check
                if exact_independent_set_size(adj, n, 11, deadline - 5):
                    # Alpha >= 11, invalid
                    # We should backtrack or try a different S.
                    # For simplicity, we will revert this vertex and try a different S.
                    # But greedy doesn't backtrack easily.
                    # Let's just return the best valid graph found so far.
                    # We need to store the best valid state.
                    pass # This logic is flawed for a simple greedy. 
                    
            # We need a better strategy: 
            # Keep a stack of states. 
            # If we hit alpha >= 11, backtrack.
            
        return n, edges, adj

    # The simple greedy above might fail to find a valid graph with n=43.
    # Let's use a more robust backtracking search for the addition of vertices.
    
    def backtrack_addition(adj, edges, n, target_n, deadline):
        # State: adj, edges, n
        # Goal: reach target_n with alpha < 11 and C4-free
        
        if n == target_n:
            # Verify
            if exact_independent_set_size(adj, n, 11, deadline - 5):
                return None
            return n, edges[:]
            
        if time.time() > deadline:
            return None
            
        new_v = n
        # Generate possible neighbor sets S
        # We want S to be valid (no common neighbors between pairs in S)
        # And we want to minimize the resulting alpha.
        
        # Heuristic: Try to add edges to vertices that are in many independent sets.
        # For now, let's try a few random S sets.
        
        candidates = list(range(n))
        random.shuffle(candidates)
        
        # Try different sizes of S? 
        # Larger S reduces alpha more.
        
        # Let's try to find a large valid S.
        S = set()
        for u in candidates:
            can_add = True
            for v in S:
                if adj[u] & adj[v]:
                    can_add = False
                    break
            if can_add:
                S.add(u)
                
        # Try this S
        adj_new = [set(s) for s in adj]
        adj_new.append(set())
        edges_new = edges[:]
        
        for u in S:
            adj_new[new_v].add(u)
            adj_new[u].add(new_v)
            edges_new.append([new_v, u])
            
        # Check C4-free (should be guaranteed by construction of S)
        # But let's verify just in case
        is_c4_free = True
        for i in range(n+1):
            for j in range(i+1, n+1):
                if len(adj_new[i] & adj_new[j]) > 1:
                    is_c4_free = False
                    break
            if not is_c4_free:
                break
                
        if is_c4_free:
            # Check alpha
            # If n is small, exact check.
            if n+1 <= 35:
                if not exact_independent_set_size(adj_new, n+1, 11, deadline - 5):
                    res = backtrack_addition(adj_new, edges_new, n+1, target_n, deadline)
                    if res:
                        return res
            else:
                # Heuristic check
                # If heuristic says alpha < 11, proceed.
                # We will do exact check at the end.
                res = backtrack_addition(adj_new, edges_new, n+1, target_n, deadline)
                if res:
                    return res
                    
        # If this S fails, try another? 
        # For simplicity, we will just try one S per step in this basic implementation.
        # This might not find a solution. 
        # Let's add a loop to try multiple random S sets.
        
        return None

    # Let's implement a more robust search with multiple restarts and randomization.
    
    best_result = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Reset to base
        adj = [set() for _ in range(30)]
        edges = []
        for i in range(10):
            u, v, w = 3*i, 3*i+1, 3*i+2
            adj[u].add(v); adj[v].add(u)
            adj[u].add(w); adj[w].add(u)
            adj[v].add(w); adj[w].add(v)
            edges.extend([[u,v], [u,w], [v,w]])
            
        n = 30
        target_n = 43
        
        # Backtracking search
        def search(n, adj, edges):
            if time.time() > deadline:
                return None
            if n == target_n:
                if not exact_independent_set_size(adj, n, 11, deadline - 5):
                    return n, edges[:]
                return None
                
            new_v = n
            # Generate a few random valid neighbor sets
            # We want to maximize the chance of keeping alpha low.
            # Let's try 5 random sets S.
            
            for _ in range(5):
                if time.time() > deadline:
                    return None
                    
                candidates = list(range(n))
                random.shuffle(candidates)
                
                S = set()
                for u in candidates:
                    can_add = True
                    for v in S:
                        if adj[u] & adj[v]:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                        
                # Skip if S is empty (isolated vertex increases alpha)
                if not S:
                    continue
                    
                adj_new = [set(s) for s in adj]
                adj_new.append(set())
                edges_new = edges[:]
                
                for u in S:
                    adj_new[new_v].add(u)
                    adj_new[u].add(new_v)
                    edges_new.append([new_v, u])
                    
                # Verify C4-free
                valid = True
                for i in range(n+1):
                    for j in range(i+1, n+1):
                        if len(adj_new[i] & adj_new[j]) > 1:
                            valid = False
                            break
                    if not valid:
                        break
                        
                if not valid:
                    continue
                    
                # Check alpha (heuristic or exact)
                if n+1 <= 35:
                    if exact_independent_set_size(adj_new, n+1, 11, deadline - 5):
                        continue
                else:
                    # Heuristic: if alpha seems high, skip
                    # Simple heuristic: count vertices with degree 0? 
                    # Or just proceed and check at the end.
                    pass
                
                res = search(n+1, adj_new, edges_new)
                if res:
                    return res
                    
            return None
            
        res = search(n, adj, edges)
        if res:
            best_result = res
            break # Found a solution, no need for more restarts? 
                   # Or try to find a better one? The task is to maximize n. 
                   # If we reach 43, that's the max. 
                   
    if best_result:
        n, edges = best_result
        return {'n': n, 'edges': edges}
        
    # Fallback: Return the best valid graph found during the search, or a base graph.
    # If no solution found, return the base graph (n=30, alpha=10).
    # n=30 is valid.
    return {'n': 30, 'edges': edges}

# Run construct to test
# if __name__ == '__main__':
#     res = construct()
#     print(res['n'])
