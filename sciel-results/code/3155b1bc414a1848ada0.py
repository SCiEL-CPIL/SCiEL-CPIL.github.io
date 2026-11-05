import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_structures(edges, adj):
        """Build edge-to-index, and for each edge its 'conflict' set (edges that cannot share a color class).
        Two edges conflict if they share a vertex OR are joined by an edge (i.e., there is an edge connecting
        an endpoint of one to an endpoint of the other)."""
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        # For each edge, precompute the set of edges that conflict with it
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex with (u,v)
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
        # Convert to lists for speed
        conflicts = [list(c) for c in conflicts]
        return edge_to_idx, conflicts

    def greedy_induced_matching(remaining, conflicts, adj, edges, rng):
        """Greedy: pick edges in order of fewest conflicts among remaining, add if compatible with current class."""
        if not remaining:
            return []
        # Sort by degree in conflict graph (among remaining)
        rem_set = set(remaining)
        # Count conflicts within remaining
        deg = {}
        for i in remaining:
            deg[i] = sum(1 for j in conflicts[i] if j in rem_set)
        order = sorted(remaining, key=lambda i: deg[i])
        cls = []
        cls_set = set()
        for i in order:
            if i not in rem_set:
                continue
            # Check if i conflicts with any edge in cls
            ok = True
            for j in cls_set:
                if j in conflicts[i]:
                    ok = False
                    break
            if ok:
                cls.append(i)
                cls_set.add(i)
        return cls

    def hill_climb_class(cls, remaining, conflicts, rng, max_iters=2000):
        """
        Given a class (list of edge indices) that is an induced matching, try to grow it
        by adding edges from 'remaining' and/or removing edges from cls, via 1-D flips.
        Objective: maximize len(cls).
        """
        cls_set = set(cls)
        best_len = len(cls)
        
        # Precompute: for each edge in remaining, how many conflicts it has with cls
        # We'll do local search: try adding an edge from remaining, or removing one from cls.
        
        for it in range(max_iters):
            improved = False
            # Try adding a random edge from remaining
            if remaining:
                candidates = [i for i in remaining if i not in cls_set]
                if candidates:
                    i = rng.choice(candidates)
                    # Check if i can be added (no conflict with cls)
                    can_add = True
                    for j in cls_set:
                        if j in conflicts[i]:
                            can_add = False
                            break
                    if can_add:
                        cls_set.add(i)
                        improved = True
                        if len(cls_set) > best_len:
                            best_len = len(cls_set)
            
            # Try removing a random edge from cls (to free up conflicts for others)
            if cls_set:
                i = rng.choice(list(cls_set))
                cls_set.discard(i)
                improved = True
            
            if not improved:
                break
            
            # After each move, try to greedily fill back
            # Greedy fill: add edges from remaining that don't conflict with cls_set
            changed = True
            while changed:
                changed = False
                for i in list(remaining):
                    if i in cls_set:
                        continue
                    can_add = True
                    for j in cls_set:
                        if j in conflicts[i]:
                            can_add = False
                            break
                    if can_add:
                        cls_set.add(i)
                        changed = True
                        if len(cls_set) > best_len:
                            best_len = len(cls_set)
        
        return list(cls_set), best_len

    def color_graph_layered(edges, adj, time_budget=40.0, seed=0):
        """
        Layered approach: repeatedly extract a large induced matching as a color class,
        then hill-climb each class. Finally try to merge classes.
        """
        m = len(edges)
        if m == 0:
            return [], 0
        
        edge_to_idx, conflicts = build_structures(edges, adj)
        rng = random.Random(seed)
        
        t_start = time.time()
        t_limit = t_start + time_budget
        
        # Phase 1: Extract color classes greedily
        remaining = set(range(m))
        classes = []
        
        while remaining and time.time() < t_limit:
            # Greedy induced matching
            cls = greedy_induced_matching(list(remaining), conflicts, adj, edges, rng)
            if not cls:
                break
            # Remove from remaining
            for i in cls:
                remaining.discard(i)
            classes.append(set(cls))
        
        # Phase 2: Hill-climb each class to grow it
        for idx in range(len(classes)):
            if time.time() > t_limit:
                break
            cls_list, _ = hill_climb_class(list(classes[idx]), remaining, conflicts, rng, max_iters=500)
            classes[idx] = set(cls_list)
        
        # Phase 3: Try to merge pairs of classes into one induced matching
        # If two classes can be combined into a single induced matching, merge them.
        merged = True
        while merged and time.time() < t_limit:
            merged = False
            new_classes = []
            i = 0
            while i < len(classes):
                if i + 1 < len(classes):
                    # Try to merge classes[i] and classes[i+1]
                    combined = classes[i] | classes[i+1]
                    # Check if combined is an induced matching
                    valid = True
                    for a in combined:
                        for b in combined:
                            if a < b and b in conflicts[a]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_classes.append(combined)
                        i += 2
                        merged = True
                    else:
                        new_classes.append(classes[i])
                        i += 1
                else:
                    new_classes.append(classes[i])
                    i += 1
            classes = new_classes
        
        # Phase 4: Try to move edges between classes to reduce total count
        # For each edge in a class, try moving it to another class if compatible
        for _ in range(3):  # A few passes
            if time.time() > t_limit:
                break
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    if time.time() > t_limit:
                        break
                    # Try to move some edges from class i to class j
                    for e in list(classes[i]):
                        # Check if e can be added to class j
                        can_add = True
                        for f in classes[j]:
                            if f in conflicts[e]:
                                can_add = False
                                break
                        if can_add:
                            classes[i].discard(e)
                            classes[j].add(e)
        
        # Remove empty classes
        classes = [c for c in classes if c]
        
        # Build color array
        colors = [-1] * m
        for c_idx, cls in enumerate(classes):
            for e in cls:
                colors[e] = c_idx
        
        return colors, len(classes)

    def verify_coloring(colors, conflicts):
        """Verify that no two conflicting edges share a color."""
        m = len(colors)
        for i in range(m):
            ci = colors[i]
            for j in conflicts[i]:
                if j > i and colors[j] == ci:
                    return False
        return True

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Pre-generate graphs
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))
    
    best_results = [None] * len(graphs)
    best_scores = [float('inf')] * len(graphs)
    
    # Allocate time per graph: ~45s each
    time_per_graph = 45.0
    
    for idx, (edges, adj) in enumerate(graphs):
        t_start = time.time()
        t_limit = t_start + time_per_graph
        
        # Multiple restarts with different seeds
        restart = 0
        while time.time() < t_limit:
            restart += 1
            inner_limit = min(t_limit, time.time() + 8.0)
            
            colors, obj = color_graph_layered(edges, adj, time_budget=inner_limit - time.time(), seed=restart)
            
            # Verify
            if colors is not None:
                edge_to_idx, conflicts = build_structures(edges, adj)
                if verify_coloring(colors, conflicts):
                    if obj < best_scores[idx]:
                        best_scores[idx] = obj
                        best_results[idx] = colors
                else:
                    # Invalid, skip
                    pass
            
            # Check conjecture bound
            d = specs[idx][1]
            conjecture = 5 * d * d // 4
            if best_scores[idx] <= conjecture:
                break
        
        # Fallback: simple greedy if no valid result
        if best_results[idx] is None:
            edge_to_idx, conflicts = build_structures(edges, adj)
            remaining = set(range(len(edges)))
            classes = []
            while remaining:
                cls = greedy_induced_matching(list(remaining), conflicts, adj, edges, random.Random(0))
                for i in cls:
                    remaining.discard(i)
                classes.append(set(cls))
            colors = [-1] * len(edges)
            for c_idx, cls in enumerate(classes):
                for e in cls:
                    colors[e] = c_idx
            best_results[idx] = colors
            best_scores[idx] = len(classes)
    
    return best_results
