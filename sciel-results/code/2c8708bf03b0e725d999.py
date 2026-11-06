import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) keeps the graph K2,3-free.
        Assumes (u,v) is NOT currently an edge.
        """
        # 1. Check common neighbors of u and v
        # If they already have >2 common neighbors, adding edge doesn't change that, 
        # but wait, if they are not neighbors, their common neighbors are just N(u) & N(v).
        # The condition is |N(u) cap N(v)| <= 2.
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # 2. Check impact on other pairs.
        # Adding (u,v) means v is added to N(u) and u is added to N(v).
        
        # For any w in N(v) (w != u, which is true since u not in N(v)):
        # Pair (u, w):
        # N(u) becomes N(u) U {v}.
        # Common neighbors of u and w: (N(u) U {v}) & N(w) = (N(u) & N(w)) U ({v} & N(w)).
        # Since w in N(v), v in N(w). So {v} is in N(w).
        # So count increases by 1.
        # We need new count <= 2, so old count <= 1.
        
        # Pair (v, w):
        # N(v) becomes N(v) U {u}.
        # Common neighbors of v and w: (N(v) U {u}) & N(w) = (N(v) & N(w)) U ({u} & N(w)).
        # Count increases by 1 if u in N(w).
        # We need new count <= 2.
        
        mask_v = adj[v]
        # We know u is not in mask_v.
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            mask_v ^= lsb
            
            # Check pair (u, w)
            # Old common neighbors of u and w: (adj[u] & adj[w]).bit_count()
            # New common neighbors: old + 1 (because v is in N(w) and v is added to N(u))
            # Condition: old + 1 <= 2 => old <= 1
            if (adj[u] & adj[w]).bit_count() > 1:
                return False
            
            # Check pair (v, w)
            # Old common neighbors of v and w: (adj[v] & adj[w]).bit_count()
            # New common neighbors: old + (1 if u in N(w) else 0)
            # Condition: old + (1 if u in N(w) else 0) <= 2
            if (adj[w] >> u) & 1:
                # u is in N(w), so count increases by 1
                if (adj[v] & adj[w]).bit_count() > 1:
                    return False
            else:
                # u not in N(w), count unchanged. 
                # We know old count <= 2 from previous feasibility.
                pass

        return True

    def can_remove_edge(adj, u, v):
        """
        Check if removing edge (u,v) keeps the graph K2,3-free.
        Removing edges never creates new common neighbors, only removes them.
        So it's always safe.
        """
        return True

    def greedy_build(seed=None):
        """Greedy random construction."""
        rng = random.Random(seed)
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for (u, v) in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def simulated_annealing(adj, max_time):
        """
        Simulated annealing to maximize edges in K2,3-free graph.
        """
        rng = random.Random()
        adj = list(adj)
        current_edges = edge_count(adj)
        best_adj = list(adj)
        best_edges = current_edges
        
        T = 1.0
        min_T = 0.01
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while time.time() < max_time:
            # Generate a neighbor
            move_type = rng.random()
            new_adj = adj[:]
            
            if move_type < 0.5:
                # Try to add an edge
                # Pick a random missing edge
                # To avoid O(n^2) scan every time, we can pick random u, v and check
                # Or maintain a list of missing edges. For n=20, scanning is fine.
                u, v = rng.choice(all_pairs)
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                        delta_m = 1
                    else:
                        continue # Invalid move
                else:
                    continue # Already present, invalid move for "add"
            
            elif move_type < 0.8:
                # Try to remove an edge
                # Pick a random existing edge
                # Find existing edges
                existing = []
                for u in range(n):
                    mask = adj[u]
                    v = u + 1
                    while v < n:
                        if (mask >> v) & 1:
                            existing.append((u, v))
                        v += 1
                if not existing:
                    continue
                u, v = rng.choice(existing)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                delta_m = -1
            
            else:
                # Try a swap: remove (u,v), add (x,y)
                # Pick random existing (u,v)
                existing = []
                for u in range(n):
                    mask = adj[u]
                    v = u + 1
                    while v < n:
                        if (mask >> v) & 1:
                            existing.append((u, v))
                        v += 1
                if not existing:
                    continue
                u, v = rng.choice(existing)
                
                # Pick random missing (x,y)
                # Ensure (x,y) is not (u,v)
                x, y = rng.choice(all_pairs)
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                if (adj[x] >> y) & 1:
                    continue # Not missing
                
                # Remove (u,v)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                # Add (x,y)
                if can_add_edge(new_adj, x, y):
                    new_adj[x] |= (1 << y)
                    new_adj[y] |= (1 << x)
                    delta_m = 0 # Removed 1, added 1
                else:
                    continue # Invalid swap

            # Evaluate new state
            new_edges = current_edges + delta_m
            
            # Acceptance criterion
            if new_edges > current_edges:
                accept = True
            else:
                # Delta m is 0 or negative
                delta_e = new_edges - current_edges
                if T > min_T:
                    accept = rng.random() < math.exp(delta_e / T)
                else:
                    accept = False
            
            if accept:
                adj = new_adj
                current_edges = new_edges
                if current_edges > best_edges:
                    best_adj = list(adj)
                    best_edges = current_edges
            
            # Cooling
            T = max(min_T, T * 0.995)

        return best_adj, best_edges

    # Initialize with a good greedy graph
    best_adj = None
    best_count = 0
    
    # Try a few different greedy seeds
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        adj = greedy_build(seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt

    # Run Simulated Annealing on the best initial graph
    if best_adj is not None and time.time() < deadline - 5.0:
        # Run SA for a few seconds or until time runs out
        # We can run it in chunks to allow for other strategies if needed, 
        # but SA is likely the best single strategy here.
        sa_adj, sa_count = simulated_annealing(best_adj, deadline - 5.0)
        if sa_count > best_count:
            best_adj = sa_adj
            best_count = sa_count

    # Fallback
    if best_adj is None:
        best_adj = greedy_build()
        best_count = edge_count(best_adj)

    return adj_to_edges(best_adj)
