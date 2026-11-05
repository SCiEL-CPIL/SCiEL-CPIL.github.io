import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 148.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q): points [x:y:z], adjacency if x1*x2+y1*y2+z1*z2 == 0
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))

    m = len(edges)

    def count_c4(u, v):
        # number of 4-cycles u-a-b-v where a~u, b~v, a~b, a!=b
        nu = adj[u]
        nv = adj[v]
        c = 0
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    c += 1
        return c

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    # ---- Adjust to exactly 91 edges using local-search-quality moves ----
    if m > 91:
        # Greedily remove edges that are in the fewest 4-cycles
        to_remove = m - 91
        for _ in range(to_remove):
            best = None
            best_c = None
            for (u, v) in list(edges):
                c = count_c4(u, v)
                if best_c is None or c < best_c:
                    best_c = c
                    best = (u, v)
            edges.remove(best)
            remove_edge(best[0], best[1])
    elif m < 91:
        to_add = 91 - m
        edge_set = set(edges)
        for _ in range(to_add):
            best = None
            best_c = None
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in edge_set:
                        continue
                    c = count_c4(u, v)
                    if best_c is None or c < best_c:
                        best_c = c
                        best = (u, v)
            edges.append(best)
            edge_set.add(best)
            add_edge(best[0], best[1])

    # ---- Local search: edge swaps to reduce total C4 count ----
    # Precompute all possible edges and non-edges for faster lookup
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def compute_total_c4():
        """Exact count of 4-cycles in current graph."""
        total = 0
        e_list = list(edges)
        L = len(e_list)
        for i in range(L):
            a, b = e_list[i]
            for j in range(i+1, L):
                c, d = e_list[j]
                if len({a,b,c,d}) < 4:
                    continue
                if (c in adj[a] and d in adj[b]) or (d in adj[a] and c in adj[b]):
                    total += 1
        return total

    # Maintain running total C4 count for faster evaluation
    total_c4_count = compute_total_c4()

    def delta_c4_swap(ru, rv, au, av):
        """
        Compute exact delta in total C4 count when removing edge (ru,rv) and adding edge (au,av).
        Delta = (new total) - (old total)
        
        Removing (ru,rv) destroys all C4s that use this edge.
        Adding (au,av) creates all C4s that use this new edge.
        
        A C4 uses edge (x,y) if there exist distinct a,b with x~a, a~b, b~y, and {x,y,a,b} all distinct,
        forming a cycle x-a-b-y-x. But we need to be careful: the C4 is x-a-b-y-x, so the edge (x,y) is 
        one side, and we need paths of length 2 from x to y that don't use edge (x,y) directly.
        
        Actually, a 4-cycle containing edge (u,v) is: u-v-a-b-u where v~a, a~b, b~u, and {u,v,a,b} distinct.
        Equivalently: for edge (u,v), count pairs (a,b) with v~a, a~b, b~u, a!=u, b!=v, a!=b.
        
        Let's define: c4_through_edge(u,v) = number of 4-cycles that include edge (u,v).
        Then delta = c4_through_edge(au,av) - c4_through_edge(ru,rv)
        
        But wait - this isn't quite right because when we remove (ru,rv) and add (au,av), the graph changes,
        so c4_through_edge values change for other edges too. However, if we only care about the net change
        in total C4 count, and if the new edge (au,av) doesn't share vertices with (ru,rv) in a way that 
        creates/destroys C4s involving both... Actually this gets complicated.
        
        Simpler approach: just compute the exact delta by:
        1. Save current adj state
        2. Perform the swap
        3. Compute new total C4
        4. Restore adj state
        5. Return new_total - old_total
        
        This is O(E^2) per evaluation which is too slow. Instead, let's use a smarter incremental approach.
        
        Key insight: The only C4s affected by removing edge (ru,rv) are those that contain (ru,rv).
        The only C4s affected by adding edge (au,av) are those that contain (au,av).
        If these two sets are disjoint (which they will be if the edges don't interact in complex ways),
        then delta = c4s_created_by(au,av) - c4s_destroyed_by(ru,rv)
        
        c4s_destroyed_by(ru,rv) = number of 4-cycles in current graph that contain edge (ru,rv)
        c4s_created_by(au,av) = number of 4-cycles that would be formed if we added (au,av) to the graph
                                 with (ru,rv) still present (approximation)
        
        For better accuracy, we can compute:
        - Remove (ru,rv) first, then compute how many C4s are created by adding (au,av) to that graph.
        - This equals: number of length-2 paths from au to av in the graph without (ru,rv), where the path
          uses vertices not creating a triangle, etc.
        
        Let me implement a cleaner version:
        """
        # c4s that contain edge (ru,rv) in current graph
        c4_ru_rv = 0
        for a in adj[ru]:
            if a == rv:
                continue
            for b in adj[rv]:
                if b == ru or b == a:
                    continue
                if a in adj[b]:
                    c4_ru_rv += 1
        
        # Now simulate: remove (ru,rv), then count c4s created by adding (au,av)
        # A C4 containing new edge (au,av) has form: au-av-x-y-au where av~x, x~y, y~au
        # We need to count such cycles in the graph with (ru,rv) removed.
        
        # Temporarily remove (ru,rv)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        
        c4_au_av = 0
        for a in adj[au]:
            if a == av:
                continue
            for b in adj[av]:
                if b == au or b == a:
                    continue
                if a in adj[b]:
                    c4_au_av += 1
        
        # Restore (ru,rv)
        adj[ru].add(rv)
        adj[rv].add(au)  # BUG: should be add(rv) not add(au)
        adj[rv].discard(au)
        adj[rv].add(rv)  # This is wrong, let me fix
        
        return c4_au_av - c4_ru_rv

    def delta_c4_swap_fixed(ru, rv, au, av):
        """Compute exact delta in total C4 for swap: remove (ru,rv), add (au,av)."""
        # Count C4s destroyed by removing (ru,rv)
        c4_ru_rv = 0
        for a in adj[ru]:
            if a == rv:
                continue
            for b in adj[rv]:
                if b == ru or b == a:
                    continue
                if a in adj[b]:
                    c4_ru_rv += 1
        
        # Temporarily remove (ru,rv)
        adj[ru].discard(rv)
        adj[rv].discard(ru)
        
        # Count C4s created by adding (au,av) to the modified graph
        c4_au_av = 0
        for a in adj[au]:
            if a == av:
                continue
            for b in adj[av]:
                if b == au or b == a:
                    continue
                if a in adj[b]:
                    c4_au_av += 1
        
        # Restore (ru,rv)
        adj[ru].add(rv)
        adj[rv].add(ru)
        
        return c4_au_av - c4_ru_rv

    # Steepest descent with periodic exact check
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # Track best solution found
    best_edges = list(edges)
    best_total = total_c4_count
    
    for it in range(500000):
        if time.time() > deadline - 5.0:
            break
        
        # Periodically verify and update best
        if it % 100 == 0:
            current_total = compute_total_c4()
            if current_total < best_total:
                best_total = current_total
                best_edges = list(edges)
        
        # Generate candidates
        rem_cands = []
        for e in edges:
            c = count_c4(e[0], e[1])
            rem_cands.append((c, e))
        rem_cands.sort()
        
        add_cands = []
        for u in range(n):
            for v in range(u+1, n):
                k = (u, v)
                if k in edge_set:
                    continue
                c = count_c4(u, v)
                add_cands.append((c, k))
        add_cands.sort()
        
        # Try top removals x top additions
        improved = False
        for rc, re in rem_cands[:20]:
            if improved:
                break
            ru, rv = re
            for ac, ae in add_cands[:20]:
                au, av = ae
                d = delta_c4_swap_fixed(ru, rv, au, av)
                if d < 0:
                    # Commit the swap
                    remove_edge(ru, rv)
                    edge_set.discard((ru, rv) if ru < rv else (rv, ru))
                    add_edge(au, av)
                    edge_set.add((au, av) if au < av else (av, au))
                    edges.remove(re)
                    edges.append((au, av))
                    total_c4_count += d
                    improved = True
                    break
        
        if not improved:
            # Try random perturbations to escape local minima
            for _ in range(5):
                re = edges[np.random.randint(len(edges))]
                while True:
                    u = np.random.randint(n)
                    v = np.random.randint(n)
                    if u == v:
                        continue
                    k = (u, v) if u < v else (v, u)
                    if k not in edge_set:
                        break
                ru, rv = re
                au, av = k
                d = delta_c4_swap_fixed(ru, rv, au, av)
                # Accept with some probability even if worse (simulated annealing style)
                if d < 0 or np.random.random() < np.exp(-d / max(1.0, 10.0 - it/10000)):
                    remove_edge(ru, rv)
                    edge_set.discard((ru, rv) if ru < rv else (rv, ru))
                    add_edge(au, av)
                    edge_set.add((au, av) if au < av else (av, au))
                    edges.remove(re)
                    edges.append((au, av))
                    total_c4_count += d
                    improved = True
                    break
        
        # Check if we improved the best
        if improved and it % 50 == 0:
            current_total = compute_total_c4()
            if current_total < best_total:
                best_total = current_total
                best_edges = list(edges)

    # Use the best solution found
    edges = best_edges
    return [[e[0], e[1]] for e in edges]
