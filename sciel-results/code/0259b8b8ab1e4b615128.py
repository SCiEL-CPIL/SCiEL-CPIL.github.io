import time
import random
import numpy as np

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115

    def time_ok():
        return time.time() < deadline

    # State
    adj = np.zeros((n, n), dtype=bool)
    colors = np.full((n, n), -1, dtype=int)
    edges = [] # list of [u, v, c]
    
    def add_edge(u, v, c):
        adj[u, v] = True
        adj[v, u] = True
        colors[u, v] = c
        colors[v, u] = c
        edges.append([u, v, c])

    def remove_edge(u, v):
        c = colors[u, v]
        adj[u, v] = False
        adj[v, u] = False
        colors[u, v] = -1
        colors[v, u] = -1
        # Remove from list
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    def get_4_cycles():
        """Yields all 4-cycles in the graph."""
        # To avoid duplicates, enforce u < v < w < x?
        # A 4-cycle is u-v-w-x-u.
        # We can iterate u, v, w, x.
        # But checking all quadruples is O(n^4) = 160,000.
        # Better: iterate pairs (u, w) and common neighbors.
        cycles = []
        for u in range(n):
            for w in range(u + 1, n):
                if not adj[u, w]:
                    # u and w are not connected, so they can be opposite vertices in a 4-cycle
                    # Common neighbors v, x
                    # v, x must be distinct and not u, w
                    common = []
                    for v in range(n):
                        if v != u and v != w and adj[u, v] and adj[w, v]:
                            common.append(v)
                    if len(common) >= 2:
                        for i in range(len(common)):
                            for j in range(i + 1, len(common)):
                                v = common[i]
                                x = common[j]
                                # Cycle u-v-w-x-u
                                # Check if edges (v,w) and (x,u) exist?
                                # adj[v,w] is true by definition of common neighbor?
                                # No, common neighbor of u and w means adj[u,v] and adj[w,v].
                                # We need adj[v,w] (true) and adj[x,u] (true).
                                # Wait, the cycle is u-v-w-x-u.
                                # Edges: (u,v), (v,w), (w,x), (x,u).
                                # adj[u,v] is true.
                                # adj[v,w] is true.
                                # adj[w,x] is true.
                                # adj[x,u] is true.
                                # So yes, any pair of common neighbors v, x forms a 4-cycle.
                                cycles.append((u, v, w, x))
        return cycles

    def is_rainbow(cycle):
        u, v, w, x = cycle
        c1 = colors[u, v]
        c2 = colors[v, w]
        c3 = colors[w, x]
        c4 = colors[x, u]
        return len({c1, c2, c3, c4}) == 4

    def find_violations():
        cycles = get_4_cycles()
        violations = []
        for c in cycles:
            if not is_rainbow(c):
                violations.append(c)
        return violations

    def is_color_valid(u, v, new_c):
        """Check if coloring (u,v) with new_c creates any non-rainbow 4-cycle."""
        # Find all 4-cycles containing (u,v)
        # Cycle must be u-v-w-x-u or u-v-x-w-u?
        # In a 4-cycle u-v-w-x-u, (u,v) is an edge.
        # The other edges are (v,w), (w,x), (x,u).
        # So we need to find w, x such that adj[v,w], adj[w,x], adj[x,u] are true.
        # And w != u, x != v, w != x.
        
        for w in range(n):
            if w == u or w == v:
                continue
            if not adj[v, w]:
                continue
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if not adj[w, x] or not adj[x, u]:
                    continue
                # Cycle u-v-w-x-u
                c_vw = colors[v, w]
                c_wx = colors[w, x]
                c_xu = colors[x, u]
                
                # Check if the other 3 are distinct
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    # This cycle is already non-rainbow due to other edges.
                    # Changing (u,v) won't fix it, but might it make it "more" non-rainbow?
                    # No, it's already invalid.
                    # However, if we are trying to *fix* a violation, we assume the rest is valid?
                    # Not necessarily.
                    # If the rest is invalid, we can't rely on this check.
                    # But if the rest is invalid, the graph is already broken.
                    # We should only call this if we are trying to make a move that *preserves* validity
                    # or *improves* it.
                    # If we assume current graph is valid, then c_vw, c_wx, c_xu are distinct.
                    # If not, we should return False to avoid making things worse?
                    # Actually, if c_vw == c_wx, the cycle is invalid regardless of new_c.
                    # So adding new_c doesn't *create* a new violation, it just doesn't fix the existing one.
                    # But we want to ensure we don't *create* a violation where there was none.
                    # If the cycle was valid before (new_c was different from others), and we change to new_c
                    # that matches one of them, we create a violation.
                    # If the cycle was already invalid, it stays invalid.
                    # So we should only care about cycles that were *valid* before?
                    # No, we just need to ensure that for ALL cycles containing (u,v),
                    # the new coloring results in a rainbow cycle.
                    # If a cycle is already non-rainbow, it will remain non-rainbow (or become rainbow?
                    # If we change new_c to match one of the others, it's still non-rainbow.
                    # If we change new_c to be distinct from others, it might become rainbow?
                    # No, if c_vw == c_wx, it's non-rainbow regardless of c_uv.
                    # So we can't fix it by changing c_uv.
                    # Therefore, if a cycle is already non-rainbow due to other edges,
                    # we can't fix it by recoloring (u,v).
                    # So we should return False if any cycle containing (u,v) is non-rainbow
                    # due to edges OTHER than (u,v).
                    return False

                # Check if new_c clashes with existing
                if new_c in (c_vw, c_wx, c_xu):
                    return False
        return True

    # Initialization
    def init_graph():
        nonlocal adj, colors, edges
        adj = np.zeros((n, n), dtype=bool)
        colors = np.full((n, n), -1, dtype=int)
        edges = []
        
        # Start with a random dense graph
        # p = 0.7 -> ~130 edges
        p = 0.7
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    c = random.randint(0, num_colors - 1)
                    add_edge(u, v, c)

    # Main Loop
    best_edges = []
    best_count = 0
    
    init_graph()
    
    iterations = 0
    while time_ok() and iterations < 5000:
        iterations += 1
        violations = find_violations()
        
        if not violations:
            # Valid graph
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [e[:] for e in edges]
            
            # Try to add edges
            added = False
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            random.shuffle(missing)
            for u, v in missing:
                # Find a valid color
                valid_c = None
                for c in range(num_colors):
                    if is_color_valid(u, v, c):
                        valid_c = c
                        break
                if valid_c is not None:
                    add_edge(u, v, valid_c)
                    added = True
                    break # Restart loop
            
            if not added:
                # Try recoloring to enable adding?
                # Or just break if we've tried many times
                # Let's try a random recolor
                if edges:
                    eu, ev, ec = random.choice(edges)
                    for new_c in range(num_colors):
                        if new_c == ec: continue
                        if is_color_valid(eu, ev, new_c):
                            # Recolor
                            remove_edge(eu, ev)
                            add_edge(eu, ev, new_c)
                            break
        else:
            # Fix a violation
            cycle = random.choice(violations)
            u, v, w, x = cycle
            # Edges: (u,v), (v,w), (w,x), (x,u)
            cycle_edges = [(u, v), (v, w), (w, x), (x, u)]
            
            fixed = False
            for eu, ev in cycle_edges:
                old_c = colors[eu, ev]
                for new_c in range(num_colors):
                    if new_c == old_c: continue
                    if is_color_valid(eu, ev, new_c):
                        # Recolor
                        remove_edge(eu, ev)
                        add_edge(eu, ev, new_c)
                        fixed = True
                        break
                if fixed: break
            
            if not fixed:
                # Remove an edge
                eu, ev = random.choice(cycle_edges)
                remove_edge(eu, ev)
                
    return best_edges
