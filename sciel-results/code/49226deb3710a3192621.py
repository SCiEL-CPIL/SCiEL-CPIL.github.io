"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def get_neighborhood(edge, adj):
        """Get all vertices that conflict with this edge (endpoints + their neighbors)."""
        u, v = edge
        near = {u, v}
        near.update(adj[u])
        near.update(adj[v])
        return near

    def greedy_color(edges, adj, order=None):
        """Greedy coloring with optional custom edge order."""
        if order is None:
            order = list(range(len(edges)))
        
        colour = {}
        # Track which colors are used
        max_color = 0
        
        for idx in order:
            e = edges[idx]
            near = get_neighborhood(e, adj)
            
            # Find taken colors
            taken = set()
            for other_idx, c in colour.items():
                if other_idx == idx:
                    continue
                oe = edges[other_idx]
                # Check if edges conflict: share vertex or are joined by an edge
                ou, ov = oe
                eu, ev = e
                # Share a vertex
                if eu in (ou, ov) or ev in (ou, ov):
                    taken.add(c)
                # Joined by an edge (one endpoint of e is neighbor of one endpoint of oe)
                elif eu in adj[ou] or eu in adj[ov] or ev in adj[ou] or ev in adj[ov]:
                    taken.add(c)
            
            # Assign smallest available color
            c = 0
            while c in taken:
                c += 1
            colour[idx] = c
            if c > max_color:
                max_color = c
        
        return colour, max_color + 1

    def local_improve(edges, adj, colour, max_colors, max_iter=5000):
        """Try to reduce max color by reassigning edges."""
        n_edges = len(edges)
        # Precompute conflict matrix for speed
        # conflict[i][j] = True if edges i and j conflict
        conflict = [[False] * n_edges for _ in range(n_edges)]
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Share vertex
                if ui in (uj, vj) or vi in (uj, vj):
                    conflict[i][j] = conflict[j][i] = True
                # Joined by edge
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i][j] = conflict[j][i] = True
        
        current_max = max_colors - 1
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            # Try to reduce the highest color
            if current_max <= 0:
                break
            
            # Find edges with the highest color
            high_edges = [i for i in range(n_edges) if colour[i] == current_max]
            if not high_edges:
                current_max -= 1
                continue
            
            random.shuffle(high_edges)
            
            for e_idx in high_edges:
                # Try to reassign this edge to a lower color
                taken = set()
                for j in range(n_edges):
                    if j != e_idx and conflict[e_idx][j]:
                        taken.add(colour[j])
                
                # Find smallest available color
                new_c = 0
                while new_c in taken:
                    new_c += 1
                
                if new_c < colour[e_idx]:
                    colour[e_idx] = new_c
                    # Check if we can reduce max
                    all_colors = set(colour.values())
                    new_max = max(all_colors) if all_colors else 0
                    if new_max < current_max:
                        current_max = new_max
                    improved = True
                    break
                else:
                    # Try swapping: see if we can move another edge from this color
                    # to free up a slot, then reassign e_idx
                    # For simplicity, just try direct reassignment for now
                    pass

        return colour, current_max + 1

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n_edges = len(edges)
        
        # Compute edge "constraint degree" for ordering
        # An edge is more constrained if its endpoints have many neighbors
        edge_constraints = []
        for i, (u, v) in enumerate(edges):
            # Number of other edges that conflict with this one
            count = 0
            for j in range(n_edges):
                if i == j:
                    continue
                uj, vj = edges[j]
                if u in (uj, vj) or v in (uj, vj) or u in adj[uj] or u in adj[vj] or v in adj[uj] or v in adj[vj]:
                    count += 1
            edge_constraints.append(count)
        
        # Order by decreasing constraint (most constrained first)
        order = sorted(range(n_edges), key=lambda i: -edge_constraints[i])
        
        # Greedy coloring with constrained order
        colour, num_colors = greedy_color(edges, adj, order)
        
        # Local improvement
        if time.time() < deadline - 10:
            colour, num_colors = local_improve(edges, adj, colour, num_colors, max_iter=10000)
        
        # Convert back to list in original edge order
        result = [colour[i] for i in range(n_edges)]
        out.append(result)
        
        # Check deadline
        if time.time() > deadline - 5:
            # Not enough time for remaining instances, use simple greedy
            break
    
    # If we ran out of time, fill remaining with simple greedy
    while len(out) < 3:
        spec = ((40, 4, 101), (60, 5, 102), (80, 6, 103))[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colour, _ = greedy_color(edges, adj)
        result = [colour[i] for i in range(len(edges))]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
