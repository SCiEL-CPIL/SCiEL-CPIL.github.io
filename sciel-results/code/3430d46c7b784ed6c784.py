"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

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

    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u
            for w in adj.get(u, set()):
                if w > u:
                    e2 = (u, w)
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
                elif w < u:
                    e2 = (w, u)
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
            # Edges sharing vertex v
            for w in adj.get(v, set()):
                if w > v:
                    e2 = (v, w)
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
                elif w < v:
                    e2 = (w, v)
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            conflict[j].add(i)
            # Edges joined by an edge: for each neighbor of u, check if there's an edge from that neighbor to some other
            # Actually, "joined by an edge" means there exists an edge (x,y) where x is endpoint of e1 and y is endpoint of e2
            # More precisely: edges e1=(a,b) and e2=(c,d) are joined if there's an edge in the graph connecting {a,b} to {c,d}
            # That is, any of (a,c), (a,d), (b,c), (b,d) is an edge in the graph
            for w in adj.get(u, set()):
                if w not in (v,):
                    e2 = (v, w) if w > v else (w, v)
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i and j not in conflict[i]:
                            # Check if this is a "joined" edge, not sharing vertex
                            if w != u and w != v:
                                conflict[i].add(j)
                                conflict[j].add(i)
            for w in adj.get(v, set()):
                if w not in (u,):
                    e2 = (u, w) if w > u else (w, u)
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i and j not in conflict[i]:
                            if w != u and w != v:
                                conflict[i].add(j)
                                conflict[j].add(i)
        
        return conflict

    def greedy_coloring(conflict, m, time_limit=5.0):
        """Greedy coloring of conflict graph, trying to minimize colors used."""
        start = time.time()
        # Order edges by degree in conflict graph (descending)
        order = sorted(range(m), key=lambda i: len(conflict[i]), reverse=True)
        
        color = [-1] * m
        used_colors = set()
        
        for i in order:
            # Find available color
            forbidden = set()
            for j in conflict[i]:
                if color[j] >= 0:
                    forbidden.add(color[j])
            
            c = 0
            while c in forbidden:
                c += 1
            color[i] = c
            used_colors.add(c)
        
        # Try to reduce colors via recoloring
        for _ in range(20):
            if time.time() - start > time_limit:
                break
            improved = False
            for i in order:
                old_c = color[i]
                if old_c == 0:
                    continue
                # Try to recolor i with a lower color
                forbidden = set()
                for j in conflict[i]:
                    if color[j] >= 0:
                        forbidden.add(color[j])
                for new_c in range(old_c):
                    if new_c not in forbidden:
                        color[i] = new_c
                        improved = True
                        break
            if not improved:
                break
        
        return color, len(used_colors)

    def local_search_coloring(conflict, m, time_limit=10.0):
        """Local search to improve coloring."""
        start = time.time()
        # Start with greedy
        color, nc = greedy_coloring(conflict, m, time_limit=1.0)
        
        # Try to reduce colors
        for target_nc in range(nc - 1, 0, -1):
            if time.time() - start > time_limit:
                break
            # Try to merge color classes
            success = False
            for c1 in range(target_nc, nc):
                if time.time() - start > time_limit:
                    break
                for c2 in range(target_nc):
                    if time.time() - start > time_limit:
                        break
                    # Try to recolor all vertices with color c1 to some other color < target_nc
                    # such that no conflicts arise
                    vertices_c1 = [i for i in range(m) if color[i] == c1]
                    if not vertices_c1:
                        continue
                    
                    # Try to reassign each vertex in vertices_c1 to a color in [0, target_nc-1]
                    # such that no conflicts
                    feasible = True
                    new_color = color[:]
                    for i in vertices_c1:
                        forbidden = set()
                        for j in conflict[i]:
                            if new_color[j] >= 0 and new_color[j] != c1:
                                forbidden.add(new_color[j])
                        found = False
                        for new_c in range(target_nc):
                            if new_c not in forbidden:
                                new_color[i] = new_c
                                found = True
                                break
                        if not found:
                            feasible = False
                            break
                    if feasible:
                        color = new_color
                        success = True
                        break
                if success:
                    break
            if not success:
                break
        
        return color, max(color) + 1

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_graph(edges, adj)
        
        # Try local search
        color, nc = local_search_coloring(conflict, m, time_limit=8.0)
        
        return [color[i] for i in range(m)], nc

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() - start_time > DEADLINE - 5:
            # Fallback: use simple greedy
            edges = instance(*spec)
            m = len(edges)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            color = [0] * m
            for i, (u, v) in enumerate(edges):
                forbidden = set()
                for j in range(i):
                    u2, v2 = edges[j]
                    if u2 == u or u2 == v or v2 == u or v2 == v:
                        forbidden.add(color[j])
                    else:
                        # Check if joined by edge
                        if v2 in adj.get(u, set()) or v2 in adj.get(v, set()) or u2 in adj.get(u, set()) or u2 in adj.get(v, set()):
                            forbidden.add(color[j])
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
            out.append(color)
        else:
            colors, nc = solve_instance(*spec)
            out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
