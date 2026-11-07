import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin
    n = 15
    total_edges = n * (n - 1) // 2
    
    # Precompute edge map
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Reverse map: (u, v) -> index
    edge_index = {}
    for i, (u, v) in edge_map.items():
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    def is_valid(edge_set):
        """
        Fast validity check.
        Returns True if no path s-u-v-t (s!=t) exists with deg(s)==deg(t).
        Uses degree grouping to avoid O(deg_u * deg_v) checks when possible.
        """
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all edges (u, v)
        for e in edge_set:
            u, v = edge_map[e]
            
            # Group neighbors of u (excluding v) by degree
            s_by_deg = {}
            for s in adj[u]:
                if s == v:
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            # Group neighbors of v (excluding u) by degree
            t_by_deg = {}
            for t in adj[v]:
                if t == u:
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            # Check for conflicts
            # A conflict exists if there is a degree d such that:
            # There is s in s_by_deg[d] and t in t_by_deg[d] with s != t.
            # This fails ONLY if for all d, the sets s_by_deg[d] and t_by_deg[d]
            # are either disjoint (in terms of vertex identity, which is always true if they are different vertices)
            # Wait, s and t are vertices. s != t is the condition.
            # If s_by_deg[d] = {a} and t_by_deg[d] = {a}, then s==t, so no violation for that pair.
            # If s_by_deg[d] = {a} and t_by_deg[d] = {b} with a!=b, violation.
            # If s_by_deg[d] = {a, b} and t_by_deg[d] = {a}, then pair (b, a) is a violation.
            
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    # We need to check if there exist s in S_set, t in T_set such that s != t.
                    # This is true UNLESS S_set == T_set == {x} for some x.
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    def find_violation(edge_set):
        """
        Find one violating path (s, u, v, t) or return None.
        """
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        for e in edge_set:
            u, v = edge_map[e]
            s_by_deg = {}
            for s in adj[u]:
                if s == v:
                    continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u:
                    continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    # Find a violating pair
                    for s in S_set:
                        for t in T_set:
                            if s != t:
                                return (s, u, v, t)
        return None

    # Strategy:
    # 1. Structural Initialization: K_{7,8} bipartite graph.
    #    A = 0..6, B = 7..14.
    #    Degrees: A=8, B=7.
    #    This is a valid base.
    A = list(range(7))
    B = list(range(7, 15))
    
    best_edges_set = set()
    
    # We will run a few iterations with different randomizations of the greedy add order
    # and potentially different initial structures if K_{7,8} fails (it shouldn't).
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        random.seed(restart * 12345)
        
        current_edges = set()
        for u in A:
            for v in B:
                current_edges.add(edge_index[(u, v)])
                
        if not is_valid(current_edges):
            # Fallback: try a star
            for v in range(1, n):
                current_edges.add(edge_index[(0, v)])
            if not is_valid(current_edges):
                continue # Should not happen
        
        best_in_restart = current_edges.copy()
        
        # Greedy Add Phase
        # We want to add internal edges (within A or within B).
        # Adding cross edges is already done (complete bipartite).
        
        missing = [i for i in range(total_edges) if i not in current_edges]
        
        # Prioritize internal edges?
        # Let's just shuffle all missing.
        random.shuffle(missing)
        
        for idx in missing:
            if time.time() > deadline:
                break
            
            # Check if adding this edge keeps validity
            # Optimization: only check if the edge is "safe" locally?
            # No, degrees change globally. Must do full check or incremental.
            # Full check is O(E * deg^2). With E~100, deg~10, it's ~10^4 ops. Fast enough.
            
            new_set = current_edges | {idx}
            if is_valid(new_set):
                current_edges = new_set
                if len(current_edges) > len(best_in_restart):
                    best_in_restart = current_edges.copy()
        
        # Local Search / Refinement
        # Try to add edges that were previously rejected?
        # Or try removing an edge to allow adding two?
        # Given the score of inspirations (4.0), simple greedy on K_{7,8} is strong.
        # Let's try a few rounds of "try to add any missing edge in random order"
        
        remaining_missing = [i for i in range(total_edges) if i not in current_edges]
        
        # Multiple passes
        for _ in range(5):
            if time.time() > deadline:
                break
            if not remaining_missing:
                break
            
            random.shuffle(remaining_missing)
            progress = False
            for idx in remaining_missing[:]: # copy to allow removal
                if time.time() > deadline:
                    break
                
                new_set = current_edges | {idx}
                if is_valid(new_set):
                    current_edges = new_set
                    remaining_missing.remove(idx)
                    if len(current_edges) > len(best_in_restart):
                        best_in_restart = current_edges.copy()
                    progress = True
            
            if not progress:
                break
        
        if len(best_in_restart) > len(best_edges_set):
            best_edges_set = best_in_restart.copy()

    # Final check
    if not is_valid(best_edges_set):
        # Fallback to empty or star
        best_edges_set = set()
        for v in range(1, n):
            best_edges_set.add(edge_index[(0, v)])
        if not is_valid(best_edges_set):
            best_edges_set = set()

    # Convert to list of lists
    result = []
    for e in best_edges_set:
        u, v = edge_map[e]
        result.append([int(u), int(v)])
        
    return result
