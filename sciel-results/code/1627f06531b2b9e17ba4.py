import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Finds a K2,3-free graph on 20 vertices maximizing edges.
    Uses Simulated Annealing with a hard constraint on K2,3.
    """
    n = 20
    start_time = time.time()
    time_limit = 100.0  # Safety margin for 120s limit
    
    def is_k23_free(adj):
        """Check if the graph represented by adj is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                common = 0
                # Count common neighbors
                # Optimization: iterate over the smaller set
                if len(adj[u]) < len(adj[v]):
                    for w in adj[u]:
                        if w in adj[v]:
                            common += 1
                            if common > 2:
                                return False
                else:
                    for w in adj[v]:
                        if w in adj[u]:
                            common += 1
                            if common > 2:
                                return False
        return True

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def valid_move(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if u == v or v in adj[u]:
            return False
        
        # 1. u and v cannot have > 2 common neighbors
        common_uv = 0
        # We can optimize this check. 
        # If we add (u,v), then u and v gain each other as neighbor.
        # The condition "no two vertices have 3 common neighbors" means:
        # For any pair (x,y), |N(x) ∩ N(y)| <= 2.
        
        # Adding (u,v) affects:
        # 1. Pair (u,v): Their common neighbors are N(u) ∩ N(v). 
        #    Currently |N(u) ∩ N(v)| must be <= 2. If it is 3, we can't add.
        # 2. Pair (u, w) for w in N(v): 
        #    w is in N(v). u is in N(v) after addition.
        #    Common neighbors of u and w: (N(u) ∩ N(w)).
        #    Note: v is in N(w) (since w in N(v) and graph is undirected).
        #    v is now in N(u). So v becomes a common neighbor of u and w.
        #    So new common count for (u,w) is |N(u) ∩ N(w)| + 1 (for v).
        #    This must be <= 2. So |N(u) ∩ N(w)| must be <= 1.
        # 3. Pair (v, w) for w in N(u):
        #    Symmetric to above. |N(v) ∩ N(w)| must be <= 1.
        
        # Check 1: Common neighbors of u and v
        # Iterate over smaller set
        if len(adj[u]) < len(adj[v]):
            for w in adj[u]:
                if w in adj[v]:
                    common_uv += 1
                    if common_uv > 2:
                        return False
        else:
            for w in adj[v]:
                if w in adj[u]:
                    common_uv += 1
                    if common_uv > 2:
                        return False

        # Check 2: For all w in N(v), common(u, w) <= 1
        # Note: w cannot be u.
        for w in adj[v]:
            if w == u: continue
            common_uw = 0
            # Check intersection of adj[u] and adj[w]
            # Optimization: if len(adj[u]) < len(adj[w])
            if len(adj[u]) < len(adj[w]):
                for x in adj[u]:
                    if x in adj[w]:
                        common_uw += 1
                        if common_uw > 1:
                            return False
            else:
                for x in adj[w]:
                    if x in adj[u]:
                        common_uw += 1
                        if common_uw > 1:
                            return False

        # Check 3: For all w in N(u), common(v, w) <= 1
        for w in adj[u]:
            if w == v: continue
            common_vw = 0
            if len(adj[v]) < len(adj[w]):
                for x in adj[v]:
                    if x in adj[w]:
                        common_vw += 1
                        if common_vw > 1:
                            return False
            else:
                for x in adj[w]:
                    if x in adj[v]:
                        common_vw += 1
                        if common_vw > 1:
                            return False

        return True

    def try_add_edge(adj, u, v):
        if valid_move(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            return True
        return False

    def try_remove_edge(adj, u, v):
        if v in adj[u]:
            adj[u].remove(v)
            adj[v].remove(u)
            return True
        return False

    # Initialize with a random graph or a known good seed
    # Let's start with the 5 K4s seed to be safe, then randomize
    adj = [set() for _ in range(n)]
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                adj[base + a].add(base + b)
                adj[base + b].add(base + a)
    
    best_adj = [set(s) for s in adj]
    best_edges = count_edges(best_adj)
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 500000
    
    current_adj = [set(s) for s in adj]
    current_edges = best_edges
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random action: Add or Remove
        # Bias towards adding if possible to maximize edges
        action = random.random()
        
        if action < 0.7: # Try Add
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and v not in current_adj[u]:
                if valid_move(current_adj, u, v):
                    # Accept
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_adj = [set(s) for s in current_adj]
                        best_edges = current_edges
                else:
                    # Reject
                    pass
        else: # Try Remove (to escape local minima)
            if current_edges == 0:
                continue
            # Pick a random existing edge
            # Efficient way: pick random u, then random v in adj[u]
            u = random.randint(0, n - 1)
            if not current_adj[u]:
                continue
            v = random.choice(list(current_adj[u]))
            
            # Always accept removal? Or with probability?
            # In SA for maximization, removing an edge decreases objective.
            # Accept with probability exp(-delta/T) where delta = -1 (change in edges)
            # So prob = exp(1/T) which is > 1? No.
            # Standard SA: minimize energy. Here maximize edges.
            # Let E = -edges. Delta E = +1 for removal.
            # Accept if Delta E < 0 or with prob exp(-Delta E / T) = exp(-1/T).
            if random.random() < (2.71828 ** (-1.0 / T)):
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                current_edges -= 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Final local search: Greedy add
    # Try to add any remaining valid edges
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if v not in current_adj[u]:
                    if valid_move(current_adj, u, v):
                        current_adj[u].add(v)
                        current_adj[v].add(u)
                        current_edges += 1
                        changed = True
        if current_edges > best_edges:
            best_adj = [set(s) for s in current_adj]
            best_edges = current_edges

    # Convert back to list of edges
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
