import time
import random
import sys

def construct():
    N = 18
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initial Graph: K6,6,6
    # Parts: 0-5, 6-11, 12-17
    # Edges between all different parts.
    edges = set()
    for i in range(N):
        for j in range(i+1, N):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                edges.add((i, j))
    
    best_graph = edges.copy()
    best_score = 12 # Known score for K6,6,6
    
    # Precomputation for fast triangle checks
    # We will maintain adjacency bitmasks
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # List of all possible triangles in K_N? No, we need to check dynamically.
    # Better: Precompute for each vertex v, the list of pairs (u, w) such that (v, u, w) is a triangle in the CURRENT graph.
    # This is expensive to update.
    
    # Alternative: Exact Max Triangle-Free Subset solver using bitmasks.
    # Input: adj list (bitmasks)
    # Output: Max size of subset S with no triangle.
    
    def max_triangle_free_size(adj):
        # Use branch and bound
        # We want to find max |S| such that G[S] has no triangle.
        # This is equivalent to finding the max independent set in the hypergraph of triangles.
        
        # Precompute triangles for the current graph?
        # It's better to check on the fly.
        
        # Heuristic: Order vertices by degree?
        # Let's use a simple recursive search with pruning.
        
        # To speed up, we can use a "conflict" check.
        # If we have selected set S, and we want to add v:
        # Check if there exist u, w in S such that (u, v, w) is a triangle.
        # This is equivalent to: (adj[u] & adj[v] & (1<<w)) != 0?
        # Actually, (u,v) and (v,w) and (u,w) must be edges.
        # So u and w must be in (adj[u] & adj[v])? No.
        # u and w must be common neighbors of v? No.
        # u and w must be such that (u,v) is edge, (v,w) is edge, (u,w) is edge.
        # So w must be in (adj[u] & adj[v]).
        
        # Let's precompute a list of triangles for the current graph to make the check O(1) or fast.
        # But the graph changes.
        
        # Given N=18, we can just iterate.
        
        best = 0
        
        # Use a bitmask for selected vertices
        selected = 0
        candidates = (1 << N) - 1
        
        # Sort vertices by something?
        # Let's stick to 0..17
        
        def search(cand, sel, size):
            nonlocal best
            if size > best:
                best = size
            
            if cand == 0:
                return
            
            # Pruning: if size + popcount(cand) <= best, return
            if size + bin(cand).count('1') <= best:
                return
            
            # Pick a vertex from candidates
            # Heuristic: Pick vertex with highest degree in induced subgraph of candidates?
            # Or just lowest index for simplicity first.
            v = (cand & -cand).bit_length() - 1
            
            # Option 1: Include v
            # Check if adding v creates a triangle
            valid = True
            # We need to check if there is a pair (u, w) in sel such that (u, v, w) is a triangle.
            # This is equivalent to:
            # There exists u in sel such that (u, v) is edge, and there exists w in sel such that (w, v) is edge and (u, w) is edge.
            # Let N_v = adj[v] & sel
            # We need to check if G[N_v] has an edge.
            # i.e., is there u, w in N_v such that (u, w) is edge?
            # This is checking if the induced subgraph on N_v is not empty.
            # i.e., for any u in N_v, is (adj[u] & N_v) != 0?
            
            N_v = adj[v] & sel
            if N_v != 0:
                # Check if there is an edge within N_v
                # Iterate through bits in N_v
                n_v_iter = N_v
                has_edge = False
                while n_v_iter:
                    u = (n_v_iter & -n_v_iter).bit_length() - 1
                    n_v_iter &= n_v_iter - 1
                    if (adj[u] & N_v) != 0:
                        has_edge = True
                        break
                if has_edge:
                    valid = False
            
            if valid:
                search(cand & ~(1 << v), sel | (1 << v), size + 1)
            
            # Option 2: Exclude v
            search(cand & ~(1 << v), sel, size)
            
        search(candidates, 0, 0)
        return best

    # Optimization: The above search is O(2^N) worst case.
    # For N=18, 2^18 = 262k.
    # With pruning, it should be fast.
    
    # Let's run a local search
    current_graph = best_graph
    current_score = best_score
    temperature = 1.0
    cooling = 0.99
    
    import math
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        # Check K4-freeness if we add edge (u,v)
        # If (u,v) is not in current_graph, try to add it.
        if (u, v) not in current_graph:
            # Check if adding (u,v) creates a K4
            # Common neighbors of u and v: cn = adj[u] & adj[v]
            # If there is an edge (w1, w2) in cn, then adding (u,v) creates K4 {u,v,w1,w2}
            cn = adj[u] & adj[v]
            creates_k4 = False
            cn_iter = cn
            while cn_iter:
                w1 = (cn_iter & -cn_iter).bit_length() - 1
                cn_iter &= cn_iter - 1
                if (adj[w1] & cn) != 0:
                    creates_k4 = True
                    break
            
            if not creates_k4:
                # Add edge
                current_graph.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                new_score = max_triangle_free_size(adj)
                
                if new_score < current_score:
                    best_graph = current_graph.copy()
                    best_score = new_score
                elif new_score == current_score:
                    # Accept
                    pass
                else:
                    # Reject or accept with prob
                    delta = new_score - current_score
                    if random.random() > math.exp(-delta / temperature):
                        # Rollback
                        current_graph.remove((u, v))
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        new_score = current_score
                    else:
                        current_score = new_score
            else:
                # Reject
                pass
        else:
            # Try to remove edge (u,v)
            current_graph.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            new_score = max_triangle_free_size(adj)
            
            if new_score < current_score:
                best_graph = current_graph.copy()
                best_score = new_score
            elif new_score == current_score:
                pass
            else:
                delta = new_score - current_score
                if random.random() > math.exp(-delta / temperature):
                    # Rollback
                    current_graph.add((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_score = current_score
                else:
                    current_score = new_score
            
        temperature *= cooling
        
    # Convert best_graph set to list of lists
    return [[u, v] for u, v in best_graph]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
