import time
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    # 1. Generate all possible candidate edges not in base graph
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
    
    m = len(candidates)
    
    # 2. Precompute distances in base graph
    # For a path graph, distance is |u-v|, but we can BFS to be general if needed.
    # Here, base is a path, so dist(u,v) = |u-v|.
    # However, adding edges changes distances.
    
    # We need to solve: Minimize sum(x_e) subject to:
    # 1. For every pair (u, v), dist(u, v) <= 4.
    # 2. For every triangle (u, v, w), sum(x_e for e in {(u,v), (v,w), (u,w)}) <= 2.
    #    Note: Base edges are fixed. If a triangle involves base edges, it's a constraint on candidates.
    #    E.g., if (u,v) and (v,w) are base edges, we cannot add (u,w).
    #    If (u,v) is base and (v,w) is candidate, we cannot add (u,w).
    
    # Let's identify conflicts (triangles).
    # A triangle is a set of 3 vertices {u, v, w}.
    # Let E_base(u,v) be 1 if (u,v) is base, 0 otherwise.
    # Let x(u,v) be 1 if (u,v) is added, 0 otherwise.
    # Constraint: E_base(u,v) + E_base(v,w) + x(u,v) + x(v,w) + E_base(u,w) + x(u,w) <= 2 is NOT quite right.
    # The graph must be triangle-free. This means no 3 vertices can have all 3 edges present.
    # Edges present = Base Edges OR Added Edges.
    # For each triangle {u,v,w}:
    # (Base(u,v) OR x(u,v)) AND (Base(v,w) OR x(v,w)) AND (Base(u,w) OR x(u,w)) == False
    # This is equivalent to:
    # (Base(u,v) + x(u,v)) + (Base(v,w) + x(v,w)) + (Base(u,w) + x(u,w)) <= 2
    # Since Base is 0 or 1, and x is 0 or 1.
    # If Base(u,v)=1, Base(v,w)=1, Base(u,w)=1 -> 3 <= 2 (Impossible, but base is path, so no base triangles).
    # If Base(u,v)=1, Base(v,w)=1, Base(u,w)=0 -> 1 + x(u,v) + 1 + x(v,w) + 0 + x(u,w) <= 2
    #   => 2 + x(u,v) + x(v,w) + x(u,w) <= 2 => x(u,v)=0, x(v,w)=0, x(u,w)=0.
    #   So if two edges of a triangle are base, the third CANNOT be added.
    # If Base(u,v)=1, Base(v,w)=0, Base(u,w)=0 -> 1 + x(u,v) + x(v,w) + x(u,w) <= 2
    #   => 1 + x(u,v) + x(v,w) + x(u,w) <= 2 => x(u,v) + x(v,w) + x(u,w) <= 1.
    #   So at most one of the three edges can be added (if one is base, the other two candidates can't both be added, and the base edge is fixed).
    #   Wait, x(u,v) is 0 if (u,v) is base. So x(u,v)=0.
    #   Then 0 + x(v,w) + x(u,w) <= 1. So we can't add both (v,w) and (u,w).
    
    # 3. Diameter Constraint
    # This is harder to linearize directly because distance depends on the set of edges.
    # However, for small graphs (n=24), we can use a different approach:
    # The diameter constraint is: For all u, v, there exists a path of length <= 4.
    # This is equivalent to: The 4-th power of the adjacency matrix (A + I)^4 has no zeros off-diagonal?
    # Or simpler: BFS from every node must reach every other node within 4 steps.
    
    # Since ILP with diameter constraints is complex to formulate (exponential in constraints if done naively with paths),
    # and n=24 is small, we can use a **Branch and Bound** or **Local Search** with ILP-like relaxation?
    # Actually, let's look at the structure.
    # The path P_24 has diameter 23.
    # We need to reduce diameter to 4.
    # This requires "long" shortcuts.
    
    # Alternative Strategy: **Greedy Set Cover with Conflict Graph**
    # We want to select a subset of candidate edges S such that:
    # 1. S is an independent set in the Conflict Graph G_conflict.
    #    G_conflict has vertices = candidates. Edge between c1, c2 if adding both creates a triangle.
    # 2. The graph G_base + S has diameter <= 4.
    
    # Let's construct G_conflict.
    # Vertices are candidates (u, v).
    # Two candidates (u1, v1) and (u2, v2) conflict if they form a triangle with a base edge or each other.
    # Actually, the constraint is "Triangle Free".
    # A set of edges S is valid (triangle-free with base) if:
    # For any triangle {a,b,c} in K_24, the number of edges from (Base ∪ S) in that triangle is <= 2.
    # This is equivalent to: S does not contain a set of edges that completes a triangle with Base or other S edges.
    # This is exactly the Independent Set problem in a graph where edges represent "conflict".
    # Conflict definition:
    # - (u,v) conflicts with (v,w) if (u,w) is Base. (Adding both creates triangle u-v-w with base u-w)
    # - (u,v) conflicts with (u,w) if (v,w) is Base.
    # - (u,v) conflicts with (v,w) if (u,w) is in S? No, that's dynamic.
    # - Actually, if we pick (u,v) and (v,w), we must ensure (u,w) is NOT in Base AND NOT in S.
    #   If (u,w) is in Base, conflict.
    #   If (u,w) is in S, conflict.
    #   So (u,v) and (v,w) are incompatible if (u,w) is in Base OR (u,w) is in S.
    #   Since S is what we're choosing, this is a standard Independent Set formulation on a graph where:
    #   Edge between candidate c1 and c2 exists if c1 and c2 "complete a triangle" with any Base edge OR with each other.
    #   Wait, if c1=(u,v) and c2=(v,w), they form a triangle with (u,w).
    #   If (u,w) is Base, they conflict.
    #   If (u,w) is a candidate, they conflict IF BOTH ARE CHOSEN.
    #   So the Conflict Graph G_conf has an edge between c1 and c2 if:
    #   1. They share a vertex (say v) and the third vertex pair is in Base.
    #   2. They form a triangle with a third candidate c3? No, that's a 3-clique constraint.
    #   Independent Set in G_conf where G_conf has edges for "immediate triangle with Base" is not sufficient.
    #   We also need to ensure that if we pick (u,v) and (v,w), we didn't pick (u,w).
    #   This is equivalent to saying: The selected edges S must not contain any triangle in the graph K_n.
    #   This is the **Triangle-Free Independent Set** problem? No, it's just finding a subset of edges that is triangle-free.
    #   This is equivalent to finding an Independent Set in the **Line Graph** of the "triangle hypergraph"?
    #   Actually, it's simpler: A set of edges is triangle-free iff it is an Independent Set in the graph H where H's vertices are edges of K_n, and two vertices in H are connected if they form a triangle in K_n.
    #   Yes! Let H be the graph where V(H) = E(K_n). (e1, e2) in E(H) iff e1 and e2 share a vertex and the third edge exists in K_n.
    #   Wait, any two edges sharing a vertex in K_n form a potential triangle with the third edge.
    #   The condition is: The set of edges S must not contain a triangle.
    #   This is equivalent to S being an Independent Set in the graph H_conf where:
    #   V(H_conf) = Candidates
    #   (c1, c2) in E(H_conf) if c1 and c2 can be part of a triangle together with ANY edge (Base or Candidate).
    #   Actually, if we just ensure that for every triangle {u,v,w} in K_n, we don't pick all 3 edges, that's a 3-uniform hypergraph independent set.
    #   But for n=24, we can just use a simple heuristic:
    #   **Greedy Local Search**:
    #   1. Start with a valid solution (e.g., the hub one, or random).
    #   2. Try to remove edges (if diameter stays <= 4).
    #   3. Try to swap edges (remove one, add one) to reduce count or improve diameter.
    
    # Let's implement a robust Local Search with a strong initialization.
    # Initialization: Use the "Hub" strategy but optimize it.
    # Or better: Random restarts + Greedy Add + Greedy Remove.
    
    def get_adjacency(added_edges):
        adj = {i: set() for i in range(n)}
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def check_triangle_free(added_edges):
        # Check if Base + Added is triangle-free
        # For each added edge (u,v), check if any common neighbor w exists in Base+Added
        # This is O(m * deg)
        adj = get_adjacency(added_edges)
        # We only need to check triangles involving at least one added edge
        # Or just check all triangles in the final graph
        # n=24 is small, checking all triangles is O(n^3) or O(sum deg^2)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[u]:
                        if w > v: # u < v < w
                            if w in adj[v]:
                                return False
        return True

    def get_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d == -1 for d in dist):
                return 999
            max_d = max(max_d, max(dist))
        return max_d

    def is_valid(added_edges):
        if not check_triangle_free(added_edges):
            return False
        adj = get_adjacency(added_edges)
        if get_diameter(adj) > 4:
            return False
        return True

    # Strategy:
    # 1. Generate a large set of candidate edges.
    # 2. Use a "Greedy Add" phase to build a solution that satisfies diameter, ensuring triangle-freeness at each step.
    #    - Sort candidates by "potential" (e.g., distance they shorten).
    #    - Add edge if it doesn't create a triangle and improves diameter or is necessary.
    # 3. Use "Greedy Remove" phase to minimize edge count.
    # 4. Use "Swap" phase (Local Search) to escape local minima.
    
    # To be efficient, we precompute which edges are "safe" to add given current set?
    # No, triangle-freeness is global.
    
    # Let's try a specific structured approach first, then random search if time permits.
    # The "Hub" approach found 10 edges.
    # Can we do better?
    # Lower bound:
    # Diameter 4 means every node is within 4 steps of every other.
    # In a path, nodes 0 and 23 are 23 apart.
    # We need to bridge this.
    # Each edge (u,v) reduces the distance between u and v to 1.
    # This creates shortcuts.
    
    # Let's run a Local Search.
    
    best_edges = []
    best_score = float('inf')
    
    def evaluate(added_edges):
        # Returns (valid, diameter, triangle_free)
        if not check_triangle_free(added_edges):
            return False, 999, False
        adj = get_adjacency(added_edges)
        d = get_diameter(adj)
        return d <= 4, d, True

    # Initialization 1: The Hub Strategy
    hub = 11
    targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
    init_edges = [[hub, v] for v in targets]
    valid, d, tf = evaluate(init_edges)
    if valid:
        best_edges = init_edges
        best_score = len(init_edges)
    
    # Initialization 2: Random Greedy
    # Pick random edges that don't form triangles, until diameter <= 4.
    
    def random_greedy_init():
        added = []
        added_set = set()
        # Shuffle candidates
        cand_list = candidates[:]
        import random
        random.shuffle(cand_list)
        
        for u, v in cand_list:
            # Check if adding (u,v) creates a triangle
            # Triangles formed by (u,v) must involve a neighbor of u and neighbor of v that are connected.
            # Or (u,v) + 2 existing edges.
            # Existing edges: Base + Added.
            # Let's just check triangle-freeness of the new set.
            # Optimization: Only check triangles involving (u,v).
            # Triangle (u,v,w) exists if w is neighbor of u and neighbor of v in (Base+Added).
            # Let N_u = adj[u] in (Base+Added)
            # Let N_v = adj[v] in (Base+Added)
            # Intersection N_u ∩ N_v. If any w in intersection, triangle.
            # Note: (u,v) is not in Base, so u is not neighbor of v yet.
            
            # We need current adjacency.
            # This is expensive inside the loop.
            # Given n=24, m~276, checking full triangle freeness is O(n^3) ~ 13k ops.
            # 276 * 13k = 3.5M ops. Fast enough.
            
            new_edges = added + [[u, v]]
            new_set = added_set | {(min(u,v), max(u,v))}
            
            if check_triangle_free(new_edges):
                added = new_edges
                added_set = new_set
                # Check diameter
                adj = get_adjacency(added)
                if get_diameter(adj) <= 4:
                    return added
        return added # Might not satisfy diameter

    # Local Search:
    # 1. Remove edges if possible.
    # 2. Swap edges.
    
    def local_search(edges):
        current_edges = [e[:] for e in edges]
        current_set = set((min(u,v), max(u,v)) for u,v in current_edges)
        
        # Phase 1: Greedy Removal
        changed = True
        while changed:
            changed = False
            for i in range(len(current_edges)):
                # Try removing current_edges[i]
                new_edges = current_edges[:i] + current_edges[i+1:]
                new_set = current_set - {(min(current_edges[i][0], current_edges[i][1]), max(current_edges[i][0], current_edges[i][1]))}
                
                # Check validity
                # Triangle free is guaranteed (subset of triangle-free set)
                # Check Diameter
                adj = get_adjacency(new_edges)
                if get_diameter(adj) <= 4:
                    current_edges = new_edges
                    current_set = new_set
                    changed = True
                    break # Restart loop as indices changed
        
        # Phase 2: Swaps
        # Try swapping one edge with another candidate
        # This is expensive. O(m_current * m_candidates)
        # m_current ~ 10, m_candidates ~ 276. 2760 swaps.
        # Each swap check: Triangle free + Diameter.
        
        improved = True
        while improved:
            improved = False
            for i in range(len(current_edges)):
                # Remove current_edges[i]
                e_out = current_edges[i]
                base_for_swap = current_edges[:i] + current_edges[i+1:]
                
                # Try adding each candidate not in current_set and not in base_for_swap
                for u, v in candidates:
                    cand = (min(u,v), max(u,v))
                    if cand in current_set or cand in base_set:
                        continue
                    if cand == (min(e_out[0], e_out[1]), max(e_out[0], e_out[1])):
                        continue
                    
                    new_edges = base_for_swap + [[u, v]]
                    # Check triangle free
                    if not check_triangle_free(new_edges):
                        continue
                    # Check diameter
                    adj = get_adjacency(new_edges)
                    if get_diameter(adj) <= 4:
                        # Valid swap
                        # We keep same size, so no improvement in count.
                        # But it might enable further removals?
                        # For now, let's just accept if it leads to a state from which we can remove more?
                        # This is getting complex.
                        # Let's stick to Removal first.
                        # If we want to improve score, we need fewer edges.
                        # Swaps don't reduce count directly.
                        # But they might allow a subsequent removal.
                        # Let's try: Swap, then try to remove one edge from the new set.
                        # If we can remove one, net -1.
                        
                        # Try to remove any edge from new_edges
                        for j in range(len(new_edges)):
                            new_edges_minus = new_edges[:j] + new_edges[j+1:]
                            adj2 = get_adjacency(new_edges_minus)
                            if get_diameter(adj2) <= 4:
                                # Found a better solution (size - 1)
                                current_edges = new_edges_minus
                                current_set = set((min(x,y), max(x,y)) for x,y in current_edges)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                break
        
        return current_edges

    # Run Local Search on initial solution
    if best_edges:
        best_edges = local_search(best_edges)
        if best_edges and is_valid(best_edges):
            best_score = len(best_edges)
    
    # Try Random Greedy Initializations
    for _ in range(20):
        if time.time() - start_time > 45:
            break
        init = random_greedy_init()
        if init and is_valid(init):
            # Local Search on this
            ls_res = local_search(init)
            if ls_res and is_valid(ls_res):
                if len(ls_res) < best_score:
                    best_score = len(ls_res)
                    best_edges = ls_res
    
    # Final Check
    if not best_edges or not is_valid(best_edges):
        # Fallback to known valid solution
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        best_edges = [[hub, v] for v in targets]
    
    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
