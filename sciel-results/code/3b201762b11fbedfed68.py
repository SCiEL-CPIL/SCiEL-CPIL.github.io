import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 90s limit (or 60s if strictly enforced, but prompt says 90s kill)
    
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
                
    # Remove self-loops for the graph structure used in MIS, 
    # but handle unary constraints separately or by removing those vertices initially.
    # If 2*val is a square, val cannot be in the set.
    # Let's identify forbidden vertices.
    forbidden = set()
    for i in range(N):
        if (i + 1) + (i + 1) in squares:
            forbidden.add(i)
            
    # Filter out forbidden vertices from the candidate pool
    # This reduces N effectively.
    active_vertices = [i for i in range(N) if i not in forbidden]
    M = len(active_vertices)
    
    # Map original indices to new compact indices 0..M-1
    # Or just work with the original indices but skip forbidden ones in the graph.
    # It's cleaner to build a new graph on active_vertices.
    
    # adj[i] is a list of active neighbors of active vertex i
    adj = [set() for _ in range(M)]
    
    for i_idx, i in enumerate(active_vertices):
        for j in conflicts[i]:
            if j in forbidden:
                continue
            # Find index of j in active_vertices
            # Since M <= 256, linear search is ok, but let's use a dict for O(1)
            pass
            
    # Build a lookup for active vertex indices
    active_index_map = {v: i for i, v in enumerate(active_vertices)}
    
    for i_idx, i in enumerate(active_vertices):
        for j in conflicts[i]:
            if j in active_index_map:
                j_idx = active_index_map[j]
                if i_idx != j_idx:
                    adj[i_idx].add(j_idx)
                    
    # Convert sets to lists for faster iteration
    adj_lists = [list(s) for s in adj]
    
    # Branch and Bound for Maximum Independent Set
    # Using a greedy coloring heuristic to provide an upper bound.
    
    best_size = 0
    best_set = set()
    
    # Initial greedy solution to set a lower bound
    # Sort by degree (descending) or just random
    current_set = set()
    # Simple greedy: pick lowest degree first? Or highest? 
    # Usually for Independent Set, picking low degree vertices first is safer for size,
    # but high degree might block more conflicts. 
    # Let's try a few random greedy starts to get a good initial lower bound.
    
    for _ in range(100):
        if time.time() - start_time > DEADLINE:
            break
        # Random order
        order = list(range(M))
        random.shuffle(order)
        candidate = set()
        for v in order:
            if not any(u in candidate for u in adj_lists[v]):
                candidate.add(v)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = set(candidate)
            
    # The B&B algorithm
    # We maintain:
    # P: set of candidate vertices (current independent set candidates)
    # X: set of vertices already added to the independent set
    
    # To make it efficient, we represent P as a list or set.
    # The bound is the size of the maximum independent set in the subgraph induced by P.
    # A standard upper bound is the chromatic number of the complement graph, 
    # or more simply, a greedy coloring of the graph induced by P.
    # If we color the graph G[P] such that adjacent vertices have different colors,
    # then the max independent set size is at most the number of colors used? 
    # No, that's for Clique. For Independent Set, we look at the Complement Graph.
    # In the Complement Graph, an Independent Set in G is a Clique.
    # So we can use the greedy coloring bound on the Complement Graph.
    
    # Let's implement the standard Tomita algorithm for Maximum Clique on the Complement Graph.
    # Complement Graph: u and v are connected if they are NOT connected in G.
    # i.e., u and v are compatible (can both be in the set).
    
    # Build complement adjacency lists
    comp_adj = [set() for _ in range(M)]
    for i in range(M):
        for j in range(M):
            if i != j and j not in adj_lists[i]:
                comp_adj[i].add(j)
                
    comp_adj_lists = [list(s) for s in comp_adj]
    
    # Greedy Coloring of the complement graph induced by a set of vertices P
    # Returns the number of colors used (upper bound on max clique size in comp, 
    # which is max independent set size in G)
    def color_bound(P):
        # P is a list of vertices
        if not P:
            return 0
        
        # Greedy coloring: assign smallest possible color
        # We want to find a coloring that uses FEW colors to get a TIGHT bound?
        # Actually, for the bound to be useful, we want the number of colors to be as small as possible?
        # Wait. If the chromatic number is chi, then Max Clique <= chi? No.
        # In a graph, Max Clique <= Chromatic Number? No, Max Clique <= Chromatic Number is false.
        # Chromatic Number >= Max Clique.
        # So if we find a coloring with k colors, the Max Clique is at most k.
        # Yes. So we want to minimize k to get a tight upper bound.
        
        # Standard greedy coloring:
        # Order vertices, assign smallest color not used by neighbors.
        # The number of colors used is an upper bound on the clique number.
        
        colors = {}
        for v in P:
            used_colors = set()
            for u in comp_adj_lists[v]:
                if u in colors:
                    used_colors.add(colors[u])
            # Find smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            
        max_color = max(colors.values()) + 1 if colors else 0
        return max_color

    # Recursive B&B
    # P: list of candidate vertices
    # X: list of vertices already in the independent set (clique in comp)
    def expand(P, X):
        nonlocal best_size, best_set
        
        if time.time() - start_time > DEADLINE:
            return
            
        if not P:
            if len(X) > best_size:
                best_size = len(X)
                best_set = set(X)
            return
            
        # Bound check
        # The max possible size is len(X) + color_bound(P)
        # If this is <= best_size, prune
        bound = color_bound(P)
        if len(X) + bound <= best_size:
            return
            
        # Heuristic: Pick a vertex to branch on
        # Common heuristic: pick vertex with highest degree in the subgraph induced by P
        # Or simply the first one.
        # Let's sort P by degree in comp_adj induced by P, descending.
        
        # Calculate degrees within P
        P_set = set(P)
        degrees = {}
        for v in P:
            deg = 0
            for u in comp_adj_lists[v]:
                if u in P_set:
                    deg += 1
            degrees[v] = deg
            
        # Sort P by degree descending
        P_sorted = sorted(P, key=lambda x: degrees[x], reverse=True)
        
        # Branch on the first vertex
        v = P_sorted[0]
        
        # Option 1: Include v in the independent set
        # New P = P \ {v} \ N(v) in G (which is N(v) in Comp)
        # In Comp, neighbors of v are compatible? No.
        # In G, neighbors of v are INCOMPATIBLE.
        # So if we include v, we must remove all neighbors of v in G from P.
        # Neighbors in G are NOT in comp_adj.
        # So we remove {u in P | u not in comp_adj[v]}? 
        # Wait. comp_adj[v] contains u if u is compatible with v.
        # So incompatible vertices are those NOT in comp_adj[v] (and not v itself).
        
        # Let's be precise.
        # G is the conflict graph.
        # Comp is the compatibility graph.
        # If we pick v, we can only pick other vertices u that are compatible with v.
        # So the new candidate set P' should only contain vertices u in P such that u is compatible with v.
        # i.e., u in comp_adj[v].
        
        P_new = [u for u in P_sorted if u != v and u in comp_adj_lists[v]]
        
        # To maintain the bound efficiency, we should pass the sorted list.
        # But the order might change.
        
        expand(P_new, X + [v])
        
        # Option 2: Exclude v
        P_new_excl = [u for u in P_sorted if u != v]
        expand(P_new_excl, X)

    # Initial call
    # Start with all active vertices
    # To help the bound, we can order them initially
    initial_P = list(range(M))
    # Sort by degree in comp graph descending?
    # The color_bound function inside expand handles the local ordering for the bound,
    # but the branching order matters.
    
    # Let's run the B&B
    expand(initial_P, [])
    
    # Convert best_set (indices) back to values
    result_values = [active_vertices[i] + 1 for i in best_set]
    
    # Fallback if B&B didn't find anything (shouldn't happen, but just in case)
    if not result_values:
        # Use the greedy best
        result_values = [active_vertices[i] + 1 for i in best_set]
        if not result_values:
            # Absolute fallback: just take 1
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

# Run it
if __name__ == "__main__":
    res = construct()
    print(len(res))
