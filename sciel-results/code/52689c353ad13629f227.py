import time
import random
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = t0 + 60.0
    n = 15

    # --- Bitmask utilities ---
    # 120 possible edges, index:
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        return u * n - u * (u + 1) // 2 + (v - u - 1)

    # Precompute adjacency bitmasks
    # adj[v] is a 120-bit int where bit e is set if edge e is incident to v
    adj_mask = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            e = edge_index(u, v)
            adj_mask[u] |= (1 << e)
            adj_mask[v] |= (1 << e)

    # Precompute degree count for each edge: adding edge e increments deg of its endpoints
    # But for a quick check, we precompute for each vertex v, the set of edges incident.
    # We'll maintain a bitmask `graph` and compute degrees on the fly or maintain an array.

    def get_degrees(graph):
        """Returns list of 15 degrees."""
        degs = [0] * n
        # This is slow if done by bit iteration, but n is small.
        # Better: precompute for each edge which vertices it connects.
        for u in range(n):
            mask = adj_mask[u] & graph
            # Count bits in mask
            degs[u] = bin(mask).count('1')
        return degs

    def is_valid(graph):
        """
        Check if the graph defined by bitmask `graph` is valid.
        Valid means: no simple path s-u-v-t (distinct vertices) with deg(s) == deg(t).
        """
        degs = get_degrees(graph)
        
        # We need to check all pairs of adjacent edges (u,v) and (v,w) where u != w,
        # and then check all s in N(u)\{v} and t in N(w)\{u}.
        # Actually, a path is s-u-v-t.
        # This requires:
        # (u,v) in E
        # (v,w) in E  -- wait, the path is s-u-v-t. So edges are (s,u), (u,v), (v,t).
        # Let's re-read carefully: "path s-u-v-t on edge (u, v)".
        # So the edges are (s,u), (u,v), (v,t).
        # Vertices s,u,v,t must be distinct.
        
        # Strategy:
        # For each pair of disjoint edges (e1, e2) that share a vertex v?
        # No, the middle edge is (u,v).
        # Let the middle edge be (u,v).
        # Then s is a neighbor of u, s != v, s != t.
        # t is a neighbor of v, t != u, t != s.
        
        # So:
        # 1. For each edge (u,v) in the graph:
        # 2. For each s in N(u) \ {v}:
        # 3. For each t in N(v) \ {u}:
        # 4. If s != t and (s,u) in E and (v,t) in E (which they are by definition of neighbor)
        #    and s,t,u,v distinct (s!=v, t!=u, s!=t covers most, need s!=v which is in neighbor def, t!=u same. Need s!=t.)
        #    Then check if deg(s) == deg(t). If yes, invalid.
        
        degs = get_degrees(graph)
        
        # Get neighbors lists for speed
        neighs = [[] for _ in range(n)]
        for v in range(n):
            mask = adj_mask[v] & graph
            # iterate bits
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                # Find the other vertex for edge idx
                # We need a mapping from edge index to (u,v)
                # Let's precompute that globally.
                neighs[v].append(edge_endpoints[idx][0] if edge_endpoints[idx][1] == v else edge_endpoints[idx][1])
                m ^= lsb

        # Check constraint
        for (u, v) in edge_endpoints:
            # Check if (u,v) is in graph
            idx = edge_index(u, v)
            if not (graph & (1 << idx)):
                continue
            
            for s in neighs[u]:
                if s == v:
                    continue
                deg_s = degs[s]
                for t in neighs[v]:
                    if t == u or t == s:
                        continue
                    if degs[t] == deg_s:
                        return False
        return True

    # Precompute edge_endpoints
    edge_endpoints = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_endpoints.append((u, v))
    # Note: edge_index(u,v) must correspond to the index in edge_endpoints
    # My edge_index formula: u*n - u*(u+1)//2 + (v-u-1)
    # Let's verify:
    # u=0, v=1: 0*15 - 0 + 0 = 0. edge_endpoints[0] is (0,1). Correct.
    # u=0, v=2: 0*15 - 0 + 1 = 1. edge_endpoints[1] is (0,2). Correct.
    # u=1, v=2: 1*15 - 1*2//2 + 0 = 15 - 1 + 0 = 14. 
    # edge_endpoints[14]: 
    # u=0: 14 edges (0,1)..(0,14) are 0..13
    # u=1: 14 edges (1,2)..(1,14) are 14..27
    # So index 14 is (1,2). Correct.

    # --- Initial Construction: K_{7,8} ---
    # A = 0..6, B = 7..14
    initial_graph = 0
    for i in range(7):
        for j in range(7, 15):
            idx = edge_index(i, j)
            initial_graph |= (1 << idx)
    
    current_graph = initial_graph
    best_graph = initial_graph
    best_score = 56  # 7*8
    
    # Verify initial validity
    if not is_valid(current_graph):
        # Should not happen
        pass

    # --- Local Search ---
    random.seed(42)
    
    # Helper to count edges
    def popcount(x):
        return bin(x).count('1')
    
    # Add/remove/flip edges
    # We perform random edge flips and accept if valid and score improves (or equal score for diversity? No, maximize score).
    # Actually, we want to maximize score. If we flip an edge and it stays valid, we accept if score goes up.
    # But we might get stuck. We can allow "sideways" moves or "worse" moves with annealing?
    # Given the tight constraint, let's stick to hill climbing with restarts.
    
    def flip_edge(graph, idx):
        return graph ^ (1 << idx)
    
    num_restarts = 50
    steps_per_restart = 5000
    
    for r in range(num_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        if r == 0:
            g = current_graph
        else:
            # Perturb best found
            g = best_graph
            # Remove random 5-10 edges
            remove_count = random.randint(5, 10)
            for _ in range(remove_count):
                # Pick random edge that is present
                # Get list of set bits
                m = g
                bits = []
                while m:
                    lsb = m & (-m)
                    bits.append(lsb.bit_length() - 1)
                    m ^= lsb
                if not bits:
                    break
                idx = random.choice(bits)
                g = g & ~(1 << idx)
                
        # Local search from g
        improved = True
        local_steps = 0
        while improved and local_steps < steps_per_restart:
            if time.time() > DEADLINE - 5:
                break
            improved = False
            # Try adding random edge
            # Find absent edges
            absent_edges = []
            present_edges = []
            # To speed up, we can just pick random indices and check
            # But let's pick random index
            
            # Random shuffle of all edges for diverse order
            # This is expensive. Let's just pick random indices.
            
            for _ in range(2000): # Try 2000 random flips per local iteration? 
                # No, standard hill climbing: pick one move.
                pass
            
            # Let's do: try all 120 edges, find one that improves.
            # This is O(120 * 15^2) per step. 120 * 225 ~ 27000 ops. Very fast.
            found_improvement = False
            # Randomize edge order
            order = list(range(120))
            random.shuffle(order)
            
            for idx in order:
                new_g = flip_edge(g, idx)
                new_score = popcount(new_g)
                # Only try adding edges if we want to improve
                # But we might need to remove? No, we start from a valid graph.
                # If we remove, score decreases. We only care about max score.
                # So we only accept moves that increase score.
                # But sometimes we need to remove to add more?
                # Since we started at 56 (K7,8), maybe we can add internal edges?
                # If we add an internal edge, score increases.
                # If it becomes invalid, we reject.
                
                if new_score > popcount(g):
                    if is_valid(new_g):
                        g = new_g
                        improved = True
                        found_improvement = True
                        break
                        # If we found one, restart the shuffle? Or just continue?
                        # Just break and restart loop.
                        
            if time.time() > DEADLINE - 5:
                break

        # End while improved
        if popcount(g) > best_score:
            best_score = popcount(g)
            best_graph = g
            if time.time() > DEADLINE - 10:
                break

    # Convert best_graph back to list of edges
    result_edges = []
    for idx in range(120):
        if best_graph & (1 << idx):
            u, v = edge_endpoints[idx]
            result_edges.append([u, v])
            
    # Final validation
    if not is_valid(best_graph):
        # Fallback to K7,8
        result_edges = []
        for i in range(7):
            for j in range(7, 15):
                result_edges.append([i, j])
        
    return result_edges
