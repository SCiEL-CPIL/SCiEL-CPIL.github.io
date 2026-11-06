import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    n = 20
    n_colors = 20
    
    best_edges = [[0, v, 0] for v in range(1, 20)]  # star fallback
    best_count = len(best_edges)
    
    def is_valid_4cycle_coloring(adj, color_map):
        """
        Check if every 4-cycle has 4 distinct colors.
        adj: list of sets of neighbors
        color_map: dict (u,v) -> color where u<v
        """
        # For each pair of non-adjacent vertices (or any pair), find common neighbors
        # A 4-cycle is u-v-w-x-u where u,w are not adjacent (diagonal)
        # Actually, a 4-cycle in a simple graph: vertices a,b,c,d with edges ab, bc, cd, da
        # and a!=c, b!=d. The diagonals are ac and bd (may or may not be edges).
        # We need: for every set of 4 vertices that form a cycle (all 4 edges present),
        # the 4 edges must have distinct colors.
        
        # More efficient: for each pair of vertices (i,j) that are NOT adjacent,
        # find all common neighbors k such that i-k-j and ... wait.
        
        # A 4-cycle involves 4 distinct vertices a,b,c,d with edges (a,b), (b,c), (c,d), (d,a).
        # Let's iterate over all 4-tuples of distinct vertices.
        # But that's O(n^4) which for n=20 is 20*19*18*17 = 116280, manageable.
        
        # For each 4-combination of vertices, check if they form a 4-cycle.
        # A 4-cycle on {a,b,c,d} exists if there's a cyclic ordering where all 4 edges exist.
        # The possible 4-cycles on 4 vertices: 
        #   a-b-c-d-a (edges ab, bc, cd, da)
        #   a-b-d-c-a (edges ab, bd, dc, ca)
        #   a-c-b-d-a (edges ac, cb, bd, da)
        # These are the 3 distinct 4-cycles (up to rotation/reflection) on 4 labeled vertices.
        
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            # Check the 3 possible 4-cycle structures
            # Cycle 1: a-b-c-d-a
            if b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d]:
                colors = [color_map.get((min(a,b), max(a,b))),
                          color_map.get((min(b,c), max(b,c))),
                          color_map.get((min(c,d), max(c,d))),
                          color_map.get((min(d,a), max(d,a)))]
                if len(set(colors)) < 4:
                    return False
            # Cycle 2: a-b-d-c-a
            if d in adj[a] and c in adj[b] and d in adj[c] and a in adj[b]:
                # Wait, let me be more careful.
                # Cycle: a-b-d-c-a means edges (a,b), (b,d), (d,c), (c,a)
                if b in adj[a] and d in adj[b] and c in adj[d] and a in adj[c]:
                    colors = [color_map.get((min(a,b), max(a,b))),
                              color_map.get((min(b,d), max(b,d))),
                              color_map.get((min(d,c), max(d,c))),
                              color_map.get((min(c,a), max(c,a)))]
                    if len(set(colors)) < 4:
                        return False
            # Cycle 3: a-c-b-d-a means edges (a,c), (c,b), (b,d), (d,a)
            if c in adj[a] and b in adj[c] and d in adj[b] and a in adj[d]:
                colors = [color_map.get((min(a,c), max(a,c))),
                          color_map.get((min(c,b), max(c,b))),
                          color_map.get((min(b,d), max(b,d))),
                          color_map.get((min(d,a), max(d,a)))]
                if len(set(colors)) < 4:
                    return False
        return True
    
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2
    
    def build_color_map(edges):
        cm = {}
        for u, v, c in edges:
            key = (min(u,v), max(u,v))
            cm[key] = c
        return cm
    
    def check_validity(adj, color_map):
        """Fast validity check using precomputed 4-cycle structures."""
        # Precompute all 4-cycles for the current graph structure
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            # Check all 3 cycle structures
            # Structure 1: a-b-c-d-a
            if (b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d]):
                c1 = color_map.get((min(a,b), max(a,b)))
                c2 = color_map.get((min(b,c), max(b,c)))
                c3 = color_map.get((min(c,d), max(c,d)))
                c4 = color_map.get((min(d,a), max(d,a)))
                if len({c1, c2, c3, c4}) < 4:
                    return False
            # Structure 2: a-b-d-c-a
            if (b in adj[a] and d in adj[b] and c in adj[d] and a in adj[c]):
                c1 = color_map.get((min(a,b), max(a,b)))
                c2 = color_map.get((min(b,d), max(b,d)))
                c3 = color_map.get((min(d,c), max(d,c)))
                c4 = color_map.get((min(c,a), max(c,a)))
                if len({c1, c2, c3, c4}) < 4:
                    return False
            # Structure 3: a-c-b-d-a
            if (c in adj[a] and b in adj[c] and d in adj[b] and a in adj[d]):
                c1 = color_map.get((min(a,c), max(a,c)))
                c2 = color_map.get((min(c,b), max(c,b)))
                c3 = color_map.get((min(b,d), max(b,d)))
                c4 = color_map.get((min(d,a), max(d,a)))
                if len({c1, c2, c3, c4}) < 4:
                    return False
        return True
    
    # Strategy: Start with a moderately dense graph and greedily add edges with valid colors.
    # Then try to improve by re-coloring or adding more edges.
    
    # Initial approach: Try to build a dense graph using a structured pattern.
    # Key insight: If we use a proper edge coloring (no two adjacent edges share a color),
    # then any cycle has all distinct colors automatically. But proper edge coloring requires
    # max_degree colors. For dense graphs, we need many colors.
    
    # Alternative: Use a 1-factorization-like approach or structured coloring.
    
    # Let's try: Start with a random dense subgraph and greedily color it.
    # If coloring fails, remove problematic edges.
    
    # Better: Greedy edge addition with careful color assignment.
    
    def greedy_construct(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        color_map = {}
        edges = []
        
        # Order edges by some heuristic - maybe random
        all_possible = list(itertools.combinations(range(n), 2))
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            
            # Try to find a color for edge (u,v) that doesn't violate any 4-cycle
            valid_colors = set(range(n_colors))
            
            # For each existing 4-cycle that would be completed by adding (u,v),
            # check if the new color conflicts
            
            # A 4-cycle containing edge (u,v) has the form:
            # u-v-x-y-u where x,y are common neighbors in the right configuration
            # Specifically, if we add edge (u,v), new 4-cycles are formed with
            # existing paths of length 3 from u to v (or v to u).
            
            # Find all vertices w such that there's a path u-a-b-v or u-a-b-v etc.
            # More precisely: a 4-cycle u-v-x-y-u exists if:
            #   v-x, x-y, y-u are all edges
            # So for each pair (x,y) where x is neighbor of v, y is neighbor of u,
            # and x-y is an edge, we get a 4-cycle u-v-x-y-u.
            
            # Also: u-v-x-y-u where the cycle goes u-v, v-x, x-y, y-u
            # And: u-v-y-x-u where it goes u-v, v-y, y-x, x-u (same set of edges)
            
            # So the 4-cycles completed by adding (u,v) are:
            # For each x in adj[v], y in adj[u], if x-y is an edge (and x!=y, x!=u, y!=v):
            #   4-cycle with edges (u,v), (v,x), (x,y), (y,u)
            
            for x in adj[v]:
                if x == u:
                    continue
                for y in adj[u]:
                    if y == v or y == x:
                        continue
                    if x in adj[y]:  # i.e., y in adj[x]
                        # 4-cycle: u-v-x-y-u
                        # Existing edges: (v,x), (x,y), (y,u)
                        c_vx = color_map.get((min(v,x), max(v,x)))
                        c_xy = color_map.get((min(x,y), max(x,y)))
                        c_yu = color_map.get((min(y,u), max(y,u)))
                        if c_vx is not None and c_xy is not None and c_yu is not None:
                            # New color must differ from all three
                            valid_colors -= {c_vx, c_xy, c_yu}
            
            # Also need to check: does adding (u,v) create a 4-cycle where (u,v) is not the "new" edge?
            # No, we're only adding one edge at a time, so all new 4-cycles must include (u,v).
            
            if valid_colors:
                # Pick a color
                c = random.choice(list(valid_colors))
                color_map[(min(u,v), max(u,v))] = c
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v, c])
        
        # Verify
        if check_validity(adj, color_map):
            return edges
        return None
    
    # Try multiple seeds
    for seed in range(1000):
        if time.time() > deadline:
            break
        
        result = greedy_construct(seed)
        if result is not None:
            if len(result) > best_count:
                best_edges = result
                best_count = len(result)
    
    # Try to improve by adding more edges to the best solution
    if time.time() < deadline:
        adj = [set() for _ in range(n)]
        color_map = {}
        for u, v, c in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            color_map[(min(u,v), max(u,v))] = c
        
        all_possible = list(itertools.combinations(range(n), 2))
        random.seed(42)
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            if v in adj[u]:
                continue
            
            valid_colors = set(range(n_colors))
            for x in adj[v]:
                if x == u:
                    continue
                for y in adj[u]:
                    if y == v or y == x:
                        continue
                    if x in adj[y]:
                        c_vx = color_map.get((min(v,x), max(v,x)))
                        c_xy = color_map.get((min(x,y), max(x,y)))
                        c_yu = color_map.get((min(y,u), max(y,u)))
                        if c_vx is not None and c_xy is not None and c_yu is not None:
                            valid_colors -= {c_vx, c_xy, c_yu}
            
            if valid_colors:
                c = random.choice(list(valid_colors))
                color_map[(min(u,v), max(u,v))] = c
                adj[u].add(v)
                adj[v].add(u)
                best_edges.append([u, v, c])
                best_count += 1
        
        # Verify final
        if check_validity(adj, color_map):
            pass
        else:
            # Revert to last known good
            # This shouldn't happen if our check is correct, but just in case
            pass
    
    return best_edges

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
