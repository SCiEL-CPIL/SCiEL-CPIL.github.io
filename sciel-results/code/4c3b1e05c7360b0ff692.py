import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Helper to generate edge list from index
    def edge_from_idx(idx):
        # 0..104
        # 0: (0,1), 1: (0,2), ..., 14: (0,14)
        # 15: (1,2), ...
        # Formula:
        # u = ?
        # v = ?
        # We can just precompute a list of all edges
        pass

    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    edge_index.update({(v, u): i for i, (u, v) in enumerate(all_edges)})

    # 1. Construct initial graph using Havel-Hakimi for sequence:
    # 14, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    target_deg = [14, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Havel-Hakimi implementation
    def havel_hakimi(deg_seq):
        # deg_seq is list of (vertex, degree)
        # Returns set of edges
        edges = set()
        # Make a copy
        deg = {v: d for v, d in deg_seq}
        
        # We need to repeatedly pick the vertex with highest degree
        # To be deterministic or random? Random might help avoid bad structures.
        # Let's pick highest degree, break ties randomly.
        
        while True:
            # Find max degree vertex
            max_deg = -1
            candidates = []
            for v, d in deg.items():
                if d > max_deg:
                    max_deg = d
                    candidates = [v]
                elif d == max_deg:
                    candidates.append(v)
            
            if max_deg <= 0:
                break
            
            if max_deg == 0:
                # Check if all are 0
                if all(d == 0 for d in deg.values()):
                    break
                else:
                    # Invalid sequence? Should not happen if graphical
                    return None

            u = random.choice(candidates)
            if deg[u] < 0:
                return None
            
            # Connect u to deg[u] other vertices with highest degrees
            # Exclude u itself
            others = [(v, d) for v, d in deg.items() if v != u]
            others.sort(key=lambda x: -x[1])
            
            to_connect = others[:deg[u]]
            
            if len(to_connect) < deg[u]:
                return None # Not enough neighbors
            
            for v, _ in to_connect:
                edges.add(edge_index[(u, v)])
                deg[u] -= 1
                deg[v] -= 1
                
            if deg[u] != 0:
                return None # Should be 0
            
            if any(d < 0 for d in deg.values()):
                return None

        return edges

    # Initialize
    current_edges = havel_hakimi([(i, d) for i, d in enumerate(target_deg)])
    
    if current_edges is None:
        # Fallback to random dense graph
        current_edges = set()
        for e in range(total_edges):
            if random.random() < 0.6:
                current_edges.add(e)

    # State tracking
    adj = [set() for _ in range(n)]
    deg = [0] * n
    
    def update_state(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = all_edges[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    adj, deg = update_state(current_edges)

    # Validity Check
    def is_valid(edge_set, adj, deg):
        for e in edge_set:
            u, v = all_edges[e]
            # s in N(u) \ {v}, t in N(v) \ {u}
            s_set = adj[u] - {v}
            t_set = adj[v] - {u}
            # Group by degree
            s_by_deg = {}
            for s in s_set:
                if deg[s] not in s_by_deg:
                    s_by_deg[deg[s]] = set()
                s_by_deg[deg[s]].add(s)
                
            t_by_deg = {}
            for t in t_set:
                if deg[t] not in t_by_deg:
                    t_by_deg[deg[t]] = set()
                t_by_deg[deg[t]].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    # Check if the specific sets cause a violation
                    # A violation exists if there is ANY s in S_d and ANY t in T_d
                    # such that s != t? No, s and t are endpoints of path s-u-v-t.
                    # s is neighbor of u, t is neighbor of v.
                    # s and t are distinct because s is neighbor of u (not v) and t is neighbor of v (not u)?
                    # Actually s could be t?
                    # Path s-u-v-t.
                    # If s == t, it's a cycle s-u-v-s (length 3).
                    # The problem says "simple three-edge path... endpoints s, t".
                    # Usually simple path implies distinct vertices.
                    # "s != t" is explicitly stated in the prompt: "s != t".
                    # So if s == t, it is NOT a violation?
                    # "Reject if any simple three-edge path has endpoints of equal degree... s != t".
                    # So if s == t, we ignore it.
                    
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    
                    # If S_set and T_set share a vertex, that vertex is s=t.
                    # We only care if there is a pair s, t with s != t.
                    # This happens if:
                    # 1. len(S_set) >= 2 and len(T_set) >= 1? No.
                    # We need existence of s in S, t in T such that s != t.
                    # This is false ONLY IF S == T and len(S) == 1.
                    # If S == {x} and T == {x}, then only pair is (x, x) which is s=t. Valid.
                    # If S == {x, y} and T == {x}, pairs are (x, x) [ok], (y, x) [s!=t, violation].
                    # So violation if NOT (len(S)==1 and len(T)==1 and S==T).
                    
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
        return True

    # Local Search
    best_edges = current_edges.copy()
    best_score = len(current_edges)
    
    # Check initial validity
    if not is_valid(current_edges, adj, deg):
        # Try to fix or restart
        pass # Let local search handle it

    # Precompute missing edges
    missing = [i for i in range(total_edges) if i not in current_edges]
    random.shuffle(missing)

    # Greedy Add
    for idx in missing:
        if time.time() > deadline:
            break
        u, v = all_edges[idx]
        # Temp add
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        current_edges.add(idx)
        
        if is_valid(current_edges, adj, deg):
            best_edges = current_edges.copy()
            best_score = len(current_edges)
            missing.remove(idx)
        else:
            # Rollback
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
            current_edges.discard(idx)

    # Simulated Annealing / Random Walk
    # We will try to add edges, remove edges, and swap.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Choose move
        r = random.random()
        
        if r < 0.4:
            # Add edge
            if not missing:
                continue
            idx = random.choice(missing)
            u, v = all_edges[idx]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add(idx)
            
            if is_valid(current_edges, adj, deg):
                best_edges = current_edges.copy()
                best_score = len(current_edges)
                missing.remove(idx)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
                current_edges.discard(idx)
                
        elif r < 0.7:
            # Remove edge
            if not current_edges:
                continue
            idx = random.choice(list(current_edges))
            u, v = all_edges[idx]
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
            current_edges.discard(idx)
            missing.append(idx)
            
            # This move reduces score, only accept if it helps validity?
            # For now, always rollback unless we are in a "repair" phase.
            # Better: only accept if it allows adding more later?
            # Simpler: Random walk allows temporary degradation.
            # But we track best_edges.
            # Let's just rollback if it doesn't immediately improve or maintain validity?
            # Actually, removing an edge might be necessary to add two others.
            # We will rely on the "Add" moves to find improvements.
            # So, rollback this removal to keep density high?
            # No, let's keep it removed for a few steps?
            # Let's just rollback to maintain density, as we are maximizing edges.
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            current_edges.add(idx)
            missing.remove(idx)

        else:
            # Swap: Remove one, Add one
            if not current_edges or not missing:
                continue
            rem_idx = random.choice(list(current_edges))
            add_idx = random.choice(missing)
            
            if rem_idx == add_idx:
                continue
                
            ru, rv = all_edges[rem_idx]
            au, av = all_edges[add_idx]
            
            # Remove rem
            adj[ru].remove(rv)
            adj[rv].remove(ru)
            deg[ru] -= 1
            deg[rv] -= 1
            current_edges.discard(rem_idx)
            
            # Add add
            adj[au].add(av)
            adj[av].add(au)
            deg[au] += 1
            deg[av] += 1
            current_edges.add(add_idx)
            
            if is_valid(current_edges, adj, deg):
                # Score is same, but structure changed.
                # Keep it if valid.
                # Update missing list
                missing.remove(add_idx)
                missing.append(rem_idx)
            else:
                # Rollback add
                adj[au].remove(av)
                adj[av].remove(au)
                deg[au] -= 1
                deg[av] -= 1
                current_edges.discard(add_idx)
                
                # Rollback remove
                adj[ru].add(rv)
                adj[rv].add(ru)
                deg[ru] += 1
                deg[rv] += 1
                current_edges.add(rem_idx)
                
                # Restore missing
                if rem_idx not in missing:
                    missing.append(rem_idx)
                if add_idx in missing:
                    missing.remove(add_idx)

    # Final result
    result = []
    for e in best_edges:
        u, v = all_edges[e]
        result.append([u, v])
        
    return result
