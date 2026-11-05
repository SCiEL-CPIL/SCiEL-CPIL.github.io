"""Strong edge colouring via iterative max induced matching removal with local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

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

    def build_data(edges):
        """Build adjacency and edge-index structures."""
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # For each edge, compute the set of vertices "near" it (endpoints + neighbors of endpoints)
        # Two edges conflict if they share a vertex or are joined by an edge
        # i.e., edge e1=(a,b) and e2=(c,d) conflict if:
        #   - they share a vertex (a==c, a==d, b==c, b==d)
        #   - or there's an edge connecting {a,b} to {c,d} (a-c, a-d, b-c, b-d in adj)
        # Actually the condition is: no two edges in the same colour class share a vertex
        # or are joined by an edge. So e1 and e2 conflict if:
        #   - they share a vertex, OR
        #   - there exists an edge (x,y) where x in e1, y in e2 (i.e., x in {a,b}, y in {c,d} and (x,y) in edges)
        # Let's precompute for each edge the set of edges it conflicts with.
        
        edge_set = set(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        m = len(edges)
        
        # conflict[i] = set of j such that edge i and edge j cannot share a colour
        conflict = [set() for _ in range(m)]
        
        for i, (a, b) in enumerate(edges):
            # All edges sharing vertex a or b
            for v in (a, b):
                for w in adj[v]:
                    e2 = (min(v, w), max(v, w))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflict[i].add(j)
                    # Also edges "joined by an edge": an edge (x,y) with x in {a,b} and y in {c,d}
                    # This is already covered above? Let's think...
                    # e1 = (a,b), e2 = (c,d). They are joined by an edge if there's an edge
                    # connecting a vertex of e1 to a vertex of e2. So if (a,c) in edges, or (a,d), or (b,c), or (b,d).
                    # The above loop: for v=a, for w in adj[a]: e2=(min(a,w),max(a,w)). This finds edges incident to a.
                    # If w=c, then e2=(a,c) which is an edge joining e1 and e2. Good.
                    # If w=d, then e2=(a,d) which joins them. Good.
                    # Similarly for v=b.
                    # But wait, we also need to catch the case where e2 shares a vertex with e1.
                    # If c==a, then e2=(a,d) which is found when v=a, w=d. And it shares vertex a.
                    # So the above covers both cases. 
                    
        return n, adj, edge_index, m, conflict

    def greedy_induced_matching(conflict, m, remaining):
        """Find a large induced matching from remaining edges using greedy with min-degree ordering."""
        if not remaining:
            return []
        
        # Count conflicts within remaining
        rem_set = set(remaining)
        deg = {}
        for i in rem_set:
            deg[i] = len(conflict[i] & rem_set)
        
        matching = []
        used = set()
        
        # Sort by degree (ascending) for greedy
        candidates = sorted(rem_set, key=lambda x: deg[x])
        
        for i in candidates:
            if i in used:
                continue
            matching.append(i)
            used.add(i)
            # Remove all edges conflicting with i
            for j in conflict[i]:
                if j in rem_set:
                    used.add(j)
        
        return matching

    def color_with_iterative_removal(edges, n, adj, edge_index, m, conflict, time_budget):
        """Iteratively remove maximum induced matchings to assign colours."""
        remaining = set(range(m))
        colour = [None] * m
        num_colours = 0
        
        while remaining and time.time() < time_budget:
            # Find a large induced matching
            matching = greedy_induced_matching(conflict, m, remaining)
            if not matching:
                # Fallback: assign each remaining edge a new colour
                for i in remaining:
                    colour[i] = num_colours
                    num_colours += 1
                break
            
            for i in matching:
                colour[i] = num_colours
            
            # Remove matched edges and all edges conflicting with them
            to_remove = set()
            for i in matching:
                to_remove.add(i)
                to_remove.update(conflict[i])
            remaining -= to_remove
            num_colours += 1
        
        return colour, num_colours

    def local_search_improve(colour, num_colours, edges, conflict, m, time_budget):
        """Try to reduce number of colours via local search."""
        # Build list of edges per colour
        colour_classes = defaultdict(list)
        for i in range(m):
            colour_classes[colour[i]].append(i)
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            for c in range(num_colours):
                if c not in colour_classes or len(colour_classes[c]) == 0:
                    continue
                # Try to move an edge from colour c to a lower colour
                for i in colour_classes[c]:
                    for c2 in range(c):
                        if c2 not in colour_classes or len(colour_classes[c2]) == 0:
                            continue
                        # Check if edge i can go to colour c2
                        can_move = True
                        for j in colour_classes[c2]:
                            if i in conflict[j] or j in conflict[i]:
                                can_move = False
                                break
                        if can_move:
                            colour[i] = c2
                            colour_classes[c].remove(i)
                            colour_classes[c2].append(i)
                            if not colour_classes[c]:
                                del colour_classes[c]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Recompute num_colours
        if colour_classes:
            num_colours = max(colour_classes.keys()) + 1
        else:
            num_colours = 0
        
        return colour, num_colours

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    for spec in specs:
        if time.time() > DEADLINE - 5:
            # Emergency: use simple greedy for remaining
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            results.append([colour[e] for e in edges])
            continue
        
        edges = instance(*spec)
        n, adj, edge_index, m, conflict = build_data(edges)
        
        # Get time budget for this instance
        time_per_instance = (DEADLINE - time.time()) / max(1, len(specs) - specs.index(spec))
        time_budget = time.time() + min(time_per_instance, 40.0)
        
        colour, num_colours = color_with_iterative_removal(edges, n, adj, edge_index, m, conflict, time_budget)
        
        # Try local search improvement
        colour, num_colours = local_search_improve(colour, num_colours, edges, conflict, m, time_budget)
        
        results.append(colour)
    
    return results

# EVOLVE-BLOCK-END
