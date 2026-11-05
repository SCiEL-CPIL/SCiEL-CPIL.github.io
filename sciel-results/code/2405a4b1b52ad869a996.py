import time
import random
from collections import defaultdict, deque
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate a random graph with max degree d
        # The original code uses a specific generation method.
        # We must replicate the graph generation exactly to ensure the conflict graph is the same.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, adj):
        """
        Build the conflict graph where vertices are edges of G.
        Two edges conflict if they share a vertex or are joined by an edge.
        This is the square of the line graph L(G).
        """
        m = len(edges)
        if m == 0:
            return []
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        
        for i, (u, v) in enumerate(edges):
            # 1. Edges sharing a vertex with (u,v)
            for x in adj[u]:
                e2 = (min(u, x), max(u, x))
                if e2 in edge_to_idx:
                    conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                e2 = (min(v, x), max(v, x))
                if e2 in edge_to_idx:
                    conflicts[i].add(edge_to_idx[e2])
            
            # 2. Edges joined by an edge to (u,v)
            # An edge (a,b) is joined to (u,v) if (u,a) in E, (u,b) in E, (v,a) in E, or (v,b) in E.
            # This is equivalent to: (a,b) is incident to a neighbor of u or v.
            # We already covered incident-to-neighbor in step 1?
            # Step 1 covers edges (u,x) and (v,x).
            # If e2=(a,b) conflicts because (u,a) in E, then a is in N(u).
            # b is in N(a).
            # My step 1 logic: 
            #   for x in adj[u]: e2=(u,x). This only finds edges incident to u.
            #   It does NOT find edges (a,b) where a in N(u) and b in N(a) but a!=u and b!=v.
            #   
            # Let's redo the conflict detection correctly.
            # Conflict if:
            # - share vertex: e2 is incident to u or v.
            # - joined by edge: there exists edge (u, a) in E such that a is in e2.
            #   i.e., e2 = (a, b) and (u, a) in E.
            #   This means a is in N(u) and b is in N(a).
            #   So for each x in N(u) U N(v), for each y in N(x), edge (x,y) is a candidate.
            
            candidates = set()
            for x in adj[u]:
                candidates.add(x)
            for x in adj[v]:
                candidates.add(x)
            candidates.add(u)
            candidates.add(v)
            
            # Iterate over all pairs of candidates? No, that's O(d^4) per edge.
            # Better:
            # For each x in N(u) U N(v) U {u, v}:
            #   For each y in N(x):
            #     e2 = (min(x,y), max(x,y))
            #     If e2 in E, it conflicts.
            
            seen_neighbors = set()
            for x in adj[u]:
                if x not in seen_neighbors:
                    seen_neighbors.add(x)
                    for y in adj[x]:
                        e2 = (min(x, y), max(x, y))
                        if e2 in edge_to_idx:
                            conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                if x not in seen_neighbors:
                    seen_neighbors.add(x)
                    for y in adj[x]:
                        e2 = (min(x, y), max(x, y))
                        if e2 in edge_to_idx:
                            conflicts[i].add(edge_to_idx[e2])
            
            # Also need to include edges incident to u or v (share vertex)
            # These are covered if we consider x=u, y in N(u) -> e2=(u,y).
            # But my loop above only iterates x in N(u) and N(v).
            # I need to include u and v themselves in the "center" of the star expansion.
            
            # Let's restart the logic for correctness.
            # Conflict set C(e1) = { e2 : dist(e1, e2) <= 1 in Line Graph }
            # dist(e1, e2) = 0 if e1=e2
            # dist(e1, e2) = 1 if they share a vertex.
            # In Square of Line Graph: dist <= 2 in Line Graph?
            # No, "Induced Matching" means no two edges share a vertex AND no edge connects an endpoint of e1 to an endpoint of e2.
            # This is exactly: e1 and e2 are at distance > 1 in the Line Graph.
            # Wait. If e1=(u,v) and e2=(a,b).
            # If they share a vertex, dist_LG(e1,e2)=1.
            # If they don't share a vertex but (u,a) in E, then e1 and (u,a) share u, and (u,a) and e2 share a.
            # So dist_LG(e1, e2) = 2.
            # So the conflict graph is the Square of the Line Graph, L(G)^2.
            # Vertices in L(G)^2 are adjacent if dist_LG <= 2.
            
            # So for each edge e1=(u,v):
            # Neighbors in L(G)^2 are all edges e2 such that:
            # 1. dist_LG(e1, e2) = 1 (share vertex)
            # 2. dist_LG(e1, e2) = 2 (joined by edge)
            
            # Implementation:
            # Get all edges at dist 1 from e1:
            #   Edges incident to u or v.
            # Get all edges at dist 2 from e1:
            #   Edges incident to neighbors of u or neighbors of v.
            
            # Let's just compute the set of "forbidden" edges for e1.
            forbidden_edges = set()
            
            # Dist 1
            for x in adj[u]:
                forbidden_edges.add((min(u, x), max(u, x)))
            for x in adj[v]:
                forbidden_edges.add((min(v, x), max(v, x)))
            
            # Dist 2
            # For each neighbor x of u, look at neighbors y of x.
            # Edge (x,y) is at dist 2 from (u,v) if x is neighbor of u.
            # But (x,y) might also be dist 1 if y==v or x==v etc.
            # We just want the union.
            for x in adj[u]:
                for y in adj[x]:
                    forbidden_edges.add((min(x, y), max(x, y)))
            for x in adj[v]:
                for y in adj[x]:
                    forbidden_edges.add((min(x, y), max(x, y)))
            
            # Remove e1 itself if it got added (it will, since u is neighbor of v? No, u is not neighbor of u.
            # But if triangle u-v-w, then (u,v) is dist 1 from (v,w).
            # (u,v) is dist 2 from (u,w)? No, (u,v) and (u,w) share u.
            # Is (u,v) in forbidden_edges?
            # Dist 1: (u,v) is incident to u. Yes.
            # So we must remove e1.
            forbidden_edges.discard((u, v))
            
            for e2 in forbidden_edges:
                if e2 in edge_to_idx:
                    conflicts[i].add(edge_to_idx[e2])
            
            conflicts[i].discard(i)
            
        return [list(c) for c in conflicts]

    def hopcroft_karp(n_left, n_right, left_edges, right_edges):
        """
        Standard Hopcroft-Karp implementation.
        left_edges: list of lists, left_edges[i] = list of right nodes connected to left i.
        right_edges: list of lists, right_edges[j] = list of left nodes connected to right j.
        Returns matching size and the matching dict (left -> right).
        """
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            q = deque()
            found = False
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            while q:
                u = q.popleft()
                for v in left_edges[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found
        
        def dfs(u):
            for v in left_edges[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        
        return matching, pair_u

    def find_max_induced_matching(conflicts, remaining_edges):
        """
        Find a Maximum Induced Matching in the subgraph induced by remaining_edges.
        We model this as finding a Maximum Independent Set in the conflict subgraph.
        MIS is NP-hard, but for small graphs we can use heuristics or exact search.
        However, the directive suggests "bipartite matching reduction".
        MIS is not generally reducible to bipartite matching.
        BUT, if the conflict graph is bipartite, we can use Max Flow.
        Is the conflict graph (Square of Line Graph) bipartite?
        Line Graph L(G) is perfect. Square of a perfect graph is not necessarily perfect.
        However, for bounded degree, maybe we can use a heuristic or a small exact solver.
        
        Given the small size (m <= 80*6/2 = 240 edges max, likely less),
        we can use a branch-and-bound or a greedy heuristic with local search for MIM.
        
        Let's use a Greedy + Local Search approach for MIM first, as it's fast.
        Then we will use the exact backtracking solver for the whole coloring problem.
        """
        m_rem = len(remaining_edges)
        if m_rem == 0:
            return []
        
        # Map remaining edges to local indices 0..m_rem-1
        rem_idx_map = {e: i for i, e in enumerate(remaining_edges)}
        
        # Build local conflict graph
        local_conflicts = [set() for _ in range(m_rem)]
        for i, e in enumerate(remaining_edges):
            global_idx = None # We need global index to look up conflicts?
            # Actually, we can just rebuild conflicts for the subset or filter.
            # Filtering is easier.
            pass
        
        # Rebuild conflicts for the subset
        # This is O(m^2) which is fine for m<240.
        for i in range(m_rem):
            e1 = remaining_edges[i]
            for j in range(i + 1, m_rem):
                e2 = remaining_edges[j]
                # Check if e1 and e2 conflict
                # Conflict if share vertex or joined by edge.
                # We can use the global conflict list if we had a global index map.
                # Let's create a global index map for all edges once.
                pass
        
        # To avoid rebuilding from scratch every time, we should pass global conflicts.
        # But for simplicity in this step, let's assume we have a helper.
        
        # Actually, let's just implement a fast heuristic for MIM on the subset.
        # Greedy: pick edge with fewest conflicts, remove it and its conflicts.
        # Then local search: try to swap edges in/out to increase size.
        
        # For now, let's return a simple greedy MIM.
        # This is a placeholder for the "Maximum Induced Matching" part.
        # We will rely on the Exact Solver for the final score.
        
        # Greedy MIM
        available = set(range(m_rem))
        mim = []
        
        # Order by degree in local conflict graph
        # We need local degrees.
        # Let's compute local conflicts once for this subset.
        local_conf = [set() for _ in range(m_rem)]
        for i in range(m_rem):
            for j in range(i+1, m_rem):
                # Check conflict between remaining_edges[i] and remaining_edges[j]
                # This is expensive if done naively.
                # But m is small.
                pass
        
        # This is getting complicated. Let's stick to the Exact Backtracking Solver for the whole problem.
        # The "MIM" strategy is essentially what the Exact Solver does if we search for k colors.
        # If we can find a valid coloring with k colors, we are done.
        
        return [] # Placeholder

    def exact_coloring(edges, conflicts, k, time_limit):
        """
        Try to color the graph with k colors using backtracking and forward checking.
        Returns a valid coloring (list of colors) or None.
        """
        m = len(edges)
        colors = [-1] * m
        start_t = time.time()
        
        # Order edges by conflict degree (most constrained first)
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        
        # Precompute conflict masks for speed?
        # Bitsets are fast in Python 3.11+? Or just sets.
        # Let's use sets of neighbors.
        neigh = [set(c) for c in conflicts]
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            
            if idx == m:
                return True
            
            i = order[idx]
            
            # Try all k colors
            for c in range(k):
                # Check if color c is valid for edge i
                valid = True
                for j in neigh[i]:
                    if colors[j] == c:
                        valid = False
                        break
                
                if valid:
                    colors[i] = c
                    if backtrack(idx + 1):
                        return True
                    colors[i] = -1
                    
            return False
            
        if backtrack(0):
            return colors
        return None

    def greedy_color_with_local_search(edges, conflicts):
        """
        Generate a good initial coloring using greedy + local search.
        """
        m = len(edges)
        if m == 0:
            return [], 0
            
        # Greedy
        colors = [-1] * m
        order = list(range(m))
        order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if colors[j] != -1:
                    forbidden.add(colors[j])
            c = 0
            while c in forbidden:
                c += 1
            colors[i] = c
            
        max_color = max(colors) + 1 if colors else 0
        
        # Local Search: Recoloring to reduce max color
        # Try to move edges from max_color to lower colors
        for _ in range(100): # Iterations
            improved = False
            for i in range(m):
                if colors[i] == max_color - 1:
                    forbidden = set()
                    for j in conflicts[i]:
                        if colors[j] != -1:
                            forbidden.add(colors[j])
                    for c in range(max_color - 1):
                        if c not in forbidden:
                            colors[i] = c
                            improved = True
                            break
            if not improved:
                break
            max_color = max(colors) + 1 if colors else 0
            
        return colors, max_color

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    results = []
    
    for n, d, seed in specs:
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        conflicts = build_conflict_graph(edges, adj)
        m = len(edges)
        
        # 1. Get a good upper bound using Greedy + Local Search
        init_colors, ub = greedy_color_with_local_search(edges, conflicts)
        
        # 2. Try to improve using Exact Backtracking
        # Start from lower bound?
        # Lower bound: max degree in conflict graph + 1?
        # Or just start from ub and go down.
        best_colors = init_colors
        best_k = ub
        
        # Try k = ub - 1, ub - 2, ...
        # We have a time budget.
        # Let's allocate 40s per graph.
        graph_start = time.time()
        
        for k in range(ub - 1, 0, -1):
            if time.time() - graph_start > 40:
                break
            if time.time() > deadline - 10:
                break
            
            # Try exact coloring with k colors
            # Use a randomized backtracking to escape local minima?
            # Or just standard.
            
            # To make it faster, we can use a heuristic:
            # If k is close to ub, it's likely to succeed.
            
            # We need a timeout for the backtracking itself.
            # Let's give it 5s per k attempt.
            success_colors = exact_coloring(edges, conflicts, k, time.time() + 5.0)
            
            if success_colors is not None:
                best_colors = success_colors
                best_k = k
                # If we found a solution, we can try to go lower.
                # But if k is small, it's harder.
                # Let's continue the loop.
        
        # Store result
        # The return format is a list of edge colours.
        # The problem says "colours follow the sorted edge order".
        # My edges are already sorted.
        results.append(best_colors)
        
    return results

# Test locally
if __name__ == "__main__":
    res = construct()
    print("Result lengths:", [len(r) for r in res])
    for i, r in enumerate(res):
        if r:
            print(f"Graph {i}: Max color {max(r)+1}, Total colors used {len(set(r))}")
