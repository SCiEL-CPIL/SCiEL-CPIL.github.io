import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # Safety margin before 90s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph
    # conflicts[i] is a set of j such that i+j is a square
    # 0-indexed internally, 1-indexed values
    conflicts = [set() for _ in range(N)]
    
    for i in range(N):
        val_i = i + 1
        for j in range(N):
            val_j = j + 1
            if val_i + val_j in squares:
                conflicts[i].add(j)
                
    # Identify forbidden vertices (diagonal constraint: a+a is square)
    forbidden = set()
    for i in range(N):
        if (i + 1) + (i + 1) in squares:
            forbidden.add(i)
            
    # Filter out forbidden vertices
    active_vertices = [i for i in range(N) if i not in forbidden]
    M = len(active_vertices)
    
    # Map original indices to new compact indices 0..M-1
    active_index_map = {v: i for i, v in enumerate(active_vertices)}
    
    # Build adjacency list for the conflict graph on active vertices
    adj = [set() for _ in range(M)]
    
    for i_idx, i in enumerate(active_vertices):
        for j in conflicts[i]:
            if j in active_index_map:
                j_idx = active_index_map[j]
                if i_idx != j_idx:
                    adj[i_idx].add(j_idx)
                    
    # Convert to lists for faster iteration
    adj_lists = [list(s) for s in adj]
    
    # Helper: Compute maximum cardinality matching in a graph
    # Uses Hopcroft-Karp algorithm for efficiency, or a simpler DFS-based approach
    # Given the small size (<= 256), a simple DFS-based matching might suffice, 
    # but Hopcroft-Karp is O(E sqrt(V)) which is very fast.
    
    def max_matching_size(vertices_list, adj_dict):
        """
        vertices_list: list of vertex indices in the subgraph
        adj_dict: dict mapping vertex -> list of neighbors (within the subgraph)
        Returns the size of the maximum matching.
        """
        if not vertices_list:
            return 0
        
        # Build local adjacency list for the subgraph
        # We need to map global indices to local indices or just use the sets directly
        # Let's use the standard DFS approach for matching which is O(VE)
        # V <= 256, E <= V^2/2. This is fast enough.
        
        # match_to[u] = v means u is matched with v
        # match_to[v] = u
        match_to = {}
        
        def try_kuhn(v, visited):
            for u in adj_dict.get(v, []):
                if u in visited:
                    continue
                visited.add(u)
                if u not in match_to or try_kuhn(match_to[u], visited):
                    match_to[u] = v
                    return True
            return False
        
        matching_size = 0
        for v in vertices_list:
            visited = set()
            if try_kuhn(v, visited):
                matching_size += 1
                
        return matching_size

    # Helper: Compute Lovasz Upper Bound for Maximum Independent Set
    # The Lovasz bound for the maximum independent set in a graph G is:
    # alpha(G) <= |V| - |M|, where M is a maximum matching in G.
    # Wait, the bound is actually: alpha(G) <= |V| - |M| is for the minimum vertex cover?
    # No.
    # Let's recall the Lovasz bound.
    # For a graph G, let M be a maximum matching.
    # The size of the maximum independent set alpha(G) satisfies:
    # alpha(G) <= |V| - |M|? No, that's not right.
    # Actually, the bound is:
    # alpha(G) <= |V| - |M| is incorrect.
    
    # Let's derive it or look it up.
    # A matching M covers 2|M| vertices.
    # The remaining |V| - 2|M| vertices are unmatched.
    # In any independent set, we can pick at most one vertex from each edge in the matching.
    # So we can pick at most |M| vertices from the matched vertices.
    # We can pick all |V| - 2|M| unmatched vertices.
    # So alpha(G) <= |M| + (|V| - 2|M|) = |V| - |M|.
    
    # Yes, the bound is alpha(G) <= |V| - |M|, where M is a maximum matching.
    
    def lovasz_bound(vertices_list):
        if not vertices_list:
            return 0
        
        # Build the induced subgraph adjacency
        v_set = set(vertices_list)
        induced_adj = {}
        for v in vertices_list:
            neighbors = []
            for u in adj_lists[v]:
                if u in v_set:
                    neighbors.append(u)
            induced_adj[v] = neighbors
            
        m_size = max_matching_size(vertices_list, induced_adj)
        return len(vertices_list) - m_size

    # Branch and Bound
    best_size = 0
    best_set = set()
    
    # Initial greedy solution to set a lower bound
    # Try multiple random orderings
    for _ in range(200):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(M))
        random.shuffle(order)
        candidate = set()
        for v in order:
            if not any(u in candidate for u in adj_lists[v]):
                candidate.add(v)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = set(candidate)
            
    # Recursive B&B
    # P: list of candidate vertices
    # X: list of vertices already added to the independent set
    def expand(P, X):
        nonlocal best_size, best_set
        
        if time.time() - start_time > DEADLINE:
            return
            
        if not P:
            if len(X) > best_size:
                best_size = len(X)
                best_set = set(X)
            return
            
        # Bound check using Lovasz
        # The max possible size is len(X) + alpha(G[P])
        # alpha(G[P]) <= |P| - |M(G[P])|
        bound = lovasz_bound(P)
        if len(X) + bound <= best_size:
            return
            
        # Heuristic: Pick a vertex to branch on
        # Common heuristic: pick vertex with highest degree in the subgraph induced by P
        P_set = set(P)
        degrees = {}
        for v in P:
            deg = 0
            for u in adj_lists[v]:
                if u in P_set:
                    deg += 1
            degrees[v] = deg
            
        # Sort P by degree descending
        P_sorted = sorted(P, key=lambda x: degrees[x], reverse=True)
        
        # Branch on the first vertex
        v = P_sorted[0]
        
        # Option 1: Include v in the independent set
        # New P = P \ {v} \ N(v)
        # N(v) are neighbors of v in the conflict graph.
        # So we remove v and all its neighbors from P.
        neighbors_v = adj_lists[v]
        P_new = [u for u in P_sorted if u != v and u not in neighbors_v]
        
        expand(P_new, X + [v])
        
        # Option 2: Exclude v
        P_new_excl = [u for u in P_sorted if u != v]
        expand(P_new_excl, X)

    # Initial call
    initial_P = list(range(M))
    # Sort by degree descending initially to help branching
    initial_degrees = [len(adj_lists[i]) for i in initial_P]
    initial_P_sorted = sorted(initial_P, key=lambda x: initial_degrees[x], reverse=True)
    
    expand(initial_P_sorted, [])
    
    # Convert best_set (indices) back to values
    result_values = [active_vertices[i] + 1 for i in best_set]
    
    # Fallback if B&B didn't find anything
    if not result_values:
        result_values = [1]
        
    # Verify validity
    squares_check = set()
    k = 1
    while k * k <= 512:
        squares_check.add(k * k)
        k += 1
        
    valid = True
    for i in range(len(result_values)):
        a = result_values[i]
        for j in range(i, len(result_values)):
            b = result_values[j]
            if a + b in squares_check:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed pattern
        return list(range(1, 257, 3))
        
    return result_values

if __name__ == "__main__":
    res = construct()
    print(len(res))
