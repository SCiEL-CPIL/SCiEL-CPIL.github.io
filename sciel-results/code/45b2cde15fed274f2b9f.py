import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin out of 120s limit
    
    n = 16
    # Precompute all Q3 embeddings (subgraphs isomorphic to Q3)
    # Q3 is the 3-cube. It has 8 vertices and 12 edges.
    # We can represent a Q3 by its 8 vertices.
    # Since n=16 is small, we can enumerate all 8-subsets of vertices and check if they form a Q3.
    # However, checking if an 8-set forms a Q3 requires checking the adjacency structure.
    # A faster way: Generate all 8-tuples of distinct vertices, check if they induce a subgraph isomorphic to Q3.
    # Even better: Precompute the set of edges that form a Q3.
    
    # Let's generate all sets of 8 vertices that induce a Q3.
    # Actually, it's easier to just check for Q3 subgraphs on the fly or precompute the "forbidden edge sets".
    # But checking every 8-subset is C(16,8) = 12870. For each, checking isomorphism to Q3 is cheap.
    # Let's precompute a list of "Q3 patterns" as frozensets of edges.
    # Wait, the graph is simple. A Q3 subgraph is defined by 8 vertices and 12 specific edges between them.
    # We can iterate over all combinations of 8 vertices. For each combination, check if the induced subgraph is isomorphic to Q3.
    # If yes, record the set of edges in that subgraph.
    # Then, in the search, we want to avoid having *any* of these edge sets present.
    
    # Optimization: Instead of storing edge sets, we can just check if a specific set of 8 vertices forms a Q3.
    # Let's create a function `has_q3(adj, verts8)` that returns True if the induced subgraph on verts8 is Q3.
    # But for local search, we need to know which edges to remove.
    
    # Alternative approach: 
    # 1. Start with a dense graph (e.g., K16 or random p=0.9).
    # 2. While there is a Q3, remove an edge from one of the Q3s.
    # 3. Try to add back edges if it doesn't create a Q3.
    
    # Let's precompute all 8-vertex subsets that *can* be a Q3? No, any 8 vertices can potentially be the vertices of a Q3.
    # The structure is what matters.
    
    # Efficient check for Q3 in a subgraph of 8 vertices:
    # Q3 is bipartite, 4+4 split, each vertex degree 3 in the subgraph.
    # 1. Check if induced subgraph is bipartite.
    # 2. Check if all 8 vertices have degree 3 within the subgraph.
    # 3. Check if it's connected (implied by bipartite + degrees? No, could be two C4s? No, C4 is Q2. Two disjoint C4s would have degrees 2. So if all degrees are 3 and bipartite, is it Q3? 
    #    Q3 has 12 edges. Sum of degrees = 24. 8 vertices * 3 = 24.
    #    A connected bipartite graph with 8 vertices and 12 edges where all degrees are 3 is Q3.
    #    Is it possible to have a disconnected one? No, because if disconnected, components would have sum of degrees 2*E_comp.
    #    If a component has k vertices, sum of degrees = 2*E. If all degrees 3, sum = 3k. So 3k = 2E => k must be even.
    #    Possible splits: 4+4. A 4-vertex graph with all degrees 3 is K4, which is not bipartite.
    #    So a bipartite graph with all degrees 3 must be connected? 
    #    Actually, if it's bipartite and regular of degree 3, and has 8 vertices, it is Q3.
    
    def is_q3_subgraph(adj, verts):
        # verts is a tuple of 8 distinct vertices
        # Map local indices 0..7
        v_set = set(verts)
        # Check degrees
        for v in verts:
            deg = 0
            for u in verts:
                if u != v and adj[v][u]:
                    deg += 1
            if deg != 3:
                return False
        
        # Check bipartite
        # BFS/DFS
        color = {}
        for v in verts:
            color[v] = 0
        # It might be disconnected, but we established it must be connected if it's Q3. 
        # Let's just try to 2-color it.
        queue = []
        for v in verts:
            if color[v] == 0:
                # Start coloring
                # We need to ensure we don't use -1 for unvisited, but 0 is a valid color.
                # Let's use -1 for unvisited.
                pass
                
        color = {v: -1 for v in verts}
        for start in verts:
            if color[start] == -1:
                color[start] = 0
                q = [start]
                while q:
                    u = q.pop(0)
                    for v in verts:
                        if u != v and adj[u][v]:
                            if color[v] == -1:
                                color[v] = color[u] ^ 1
                                q.append(v)
                            elif color[v] == color[u]:
                                return False
        return True

    # Precompute all 8-combinations? C(16,8) = 12870.
    # This is small. We can check all of them in the loop.
    
    # To speed up, we can precompute the list of 8-sets.
    all_8_sets = list(itertools.combinations(range(n), 8))
    
    # Initial Graph: Random dense
    # Let's try starting with K16 and removing edges? Or random p=0.8?
    # K16 has 120 edges. Max possible is lower.
    # Let's start with a random graph with p=0.7 to avoid too many Q3s initially? 
    # Or start with K16 and do a "repair" phase?
    # Repair phase: While has_q3, remove random edge from a Q3.
    # Improvement phase: Try to add random missing edge. If no new Q3 created, keep it. Else revert.
    
    def get_q3(adj):
        """Returns a tuple of 8 vertices forming a Q3, or None."""
        # Shuffle to avoid bias
        local_sets = all_8_sets[:]
        random.shuffle(local_sets)
        for verts in local_sets:
            if is_q3_subgraph(adj, verts):
                return verts
        return None

    def has_q3_fast(adj):
        """Returns True if any Q3 exists."""
        return get_q3(adj) is not None

    # State
    adj = [[False] * n for _ in range(n)]
    
    # Initialize: Random graph p=0.8
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj[i][j] = True
                adj[j][i] = True
                
    # Repair Phase: Remove edges until Q3-free
    while time.time() < deadline - 20:
        q3_verts = get_q3(adj)
        if q3_verts is None:
            break
        # Find an edge within q3_verts to remove
        # Get all edges in this Q3
        edges_in_q3 = []
        for i in range(8):
            for j in range(i+1, 8):
                u = q3_verts[i]
                v = q3_verts[j]
                if adj[u][v]:
                    edges_in_q3.append((u, v))
        
        if not edges_in_q3:
            # Should not happen if is_q3_subgraph is correct
            break
            
        # Remove a random edge from the Q3
        u, v = random.choice(edges_in_q3)
        adj[u][v] = False
        adj[v][u] = False

    # Now we have a Q3-free graph. Let's try to add edges.
    best_edges = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
    
    # Local Search / Simulated Annealing
    # We will try to add edges. If adding an edge creates a Q3, we might still accept it with probability exp(-delta/T) if we are doing SA, 
    # but here we can just do a simple hill climbing: Add edge if it doesn't create a Q3.
    # To escape local optima, we can randomly remove an edge, then try to add.
    
    current_adj = [row[:] for row in adj]
    current_edges = best_edges
    
    # List of missing edges
    def get_missing_edges(adj):
        miss = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    miss.append((i, j))
        return miss

    # Optimization: Precompute which Q3s are "threatened" by a specific edge?
    # Too complex. Just check.
    
    # Time budget for search
    search_deadline = deadline - 5
    
    # Simple Hill Climbing with Random Restarts
    best_graph = [row[:] for row in current_adj]
    best_score = current_edges
    
    restarts = 0
    while time.time() < search_deadline:
        # If stuck or random restart
        if restarts > 50 or random.random() < 0.05:
            # Restart from current best or random?
            # Let's restart from a perturbed version of current best
            current_adj = [row[:] for row in best_graph]
            # Remove 1-2 random edges to escape
            for _ in range(2):
                missing = get_missing_edges(current_adj)
                if not missing: break
                u, v = random.choice(missing)
                current_adj[u][v] = True
                current_adj[v][u] = True
            # Wait, we want to remove edges to ensure it's Q3-free? 
            # Actually, if we start from a Q3-free graph, removing edges keeps it Q3-free.
            # So we can just start from best_graph and try to add edges.
            
        # Try to add edges
        missing = get_missing_edges(current_adj)
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > search_deadline: break
            
            # Tentatively add edge
            current_adj[u][v] = True
            current_adj[v][u] = True
            
            # Check if Q3-free
            if not has_q3_fast(current_adj):
                current_edges += 1
                if current_edges > best_score:
                    best_score = current_edges
                    best_graph = [row[:] for row in current_adj]
            else:
                # Revert
                current_adj[u][v] = False
                current_adj[v][u] = False
        
        restarts += 1

    # Return the best graph found
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i][j]:
                edges.append([i, j])
                
    return edges

# Run a quick self-test to ensure validity? 
# The evaluator will check.
# To be safe, let's do a final verification on the returned graph before returning.
# If invalid, fall back to K3,13.

def _is_valid_q3_free(edges):
    n = 16
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u][v] = True
            adj[v][u] = True
        else:
            return False
            
    # Check for Q3
    all_8_sets = list(itertools.combinations(range(n), 8))
    for verts in all_8_sets:
        # Check degrees
        valid = True
        for v in verts:
            deg = 0
            for u in verts:
                if u != v and adj[v][u]:
                    deg += 1
            if deg != 3:
                valid = False
                break
        if not valid:
            continue
            
        # Check bipartite
        color = {v: -1 for v in verts}
        for start in verts:
            if color[start] == -1:
                color[start] = 0
                q = [start]
                while q:
                    u = q.pop(0)
                    for v in verts:
                        if u != v and adj[u][v]:
                            if color[v] == -1:
                                color[v] = color[u] ^ 1
                                q.append(v)
                            elif color[v] == color[u]:
                                return False
    return True

# Wrap the main logic to include a final safety check
_original_construct = construct

def construct():
    try:
        res = _original_construct()
        if _is_valid_q3_free(res):
            return res
        else:
            # Fallback to K3,13
            return [[a, b] for a in range(3) for b in range(3, 16)]
    except Exception:
        return [[a, b] for a in range(3) for b in range(3, 16)]

# Ensure the function name is exactly construct
# The code above defines construct twice. The second one overrides the first.
# This is correct.

