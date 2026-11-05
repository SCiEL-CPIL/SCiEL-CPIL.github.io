import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # 1. Precompute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d and a specific cycle order.
    # There are 3 distinct 4-cycles on any set of 4 vertices (up to rotation/reflection).
    # Cycle 1: a-b-c-d-a
    # Cycle 2: a-b-d-c-a
    # Cycle 3: a-c-b-d-a
    cycles = []
    vertices = list(range(n))
    for combo in combinations(vertices, 4):
        a, b, c, d = combo
        # Edges as (u, v) with u < v for storage consistency
        # Cycle 1: (a,b), (b,c), (c,d), (d,a)
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,c), max(b,c))
        e3 = (min(c,d), max(c,d))
        e4 = (min(d,a), max(d,a))
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: (a,b), (b,d), (d,c), (c,a)
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,d), max(b,d))
        e3 = (min(d,c), max(d,c))
        e4 = (min(c,a), max(c,a))
        cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: (a,c), (c,b), (b,d), (d,a)
        e1 = (min(a,c), max(a,c))
        e2 = (min(c,b), max(c,b))
        e3 = (min(b,d), max(b,d))
        e4 = (min(d,a), max(d,a))
        cycles.append((e1, e2, e3, e4))
        
    num_cycles = len(cycles)
    
    # 2. Define the nodes of the conflict graph
    # Each node represents a potential edge (u, v) with a specific color c.
    # Total nodes = E(K_n) * num_colours = 190 * 20 = 3800
    # We want to select a subset of these nodes such that:
    # a) No two nodes have the same (u, v). (At most one color per edge)
    # b) No two nodes share a vertex. (Proper edge coloring)
    # c) For any 4-cycle in the graph, if all 4 edges are selected, their colors must be distinct.
    
    # Let's index the nodes.
    # Node ID = (u * n + v) * num_colours + c, where u < v
    # Actually, let's map (u, v, c) to an integer ID.
    # There are 190 unique undirected edges.
    edge_indices = {}
    node_id_map = {}
    nodes = []
    node_u = []
    node_v = []
    node_c = []
    
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            for c in range(num_colours):
                node_id_map[(u, v, c)] = idx * num_colours + c
                nodes.append((u, v, c))
                node_u.append(u)
                node_v.append(v)
                node_c.append(c)
            idx += 1
            
    total_nodes = len(nodes) # 3800
    
    # 3. Build the Conflict Graph
    # Conflict exists if:
    # A) Same edge (u, v) but different color. (We can only pick one color for an edge)
    # B) Shared vertex. (Proper edge coloring: adjacent edges must have different colors)
    #    Wait, the problem statement says "edge-coloured... every 4-cycle is rainbow".
    #    It does NOT explicitly state "proper edge coloring" (adjacent edges different colors).
    #    However, if two adjacent edges have the same color, do they form a 4-cycle?
    #    No, a 4-cycle involves 4 distinct vertices.
    #    Let's re-read carefully: "edge-coloured with 20 colours so every 4-cycle is rainbow".
    #    It does NOT forbid adjacent edges having the same color, UNLESS that creates a non-rainbow 4-cycle.
    #    But a 4-cycle has 4 distinct edges. If two adjacent edges in a 4-cycle have the same color, the cycle is not rainbow.
    #    So, the constraint "every 4-cycle is rainbow" implicitly enforces that in any 4-cycle, all 4 edge colors are distinct.
    #    Does it enforce proper edge coloring globally?
    #    If edge (u,v) and (v,w) have the same color, they don't form a 4-cycle by themselves.
    #    However, if there is a path u-v-w-x-u, and (u,v) and (v,w) have same color, the cycle is not rainbow.
    #    So, if the graph contains a 4-cycle, adjacent edges in that cycle must differ.
    #    But if the graph is C4-free, we can have adjacent edges with same color.
    #    The current program's `can_colour` function enforces proper edge coloring (forbidden_proper).
    #    Is this necessary?
    #    If we don't enforce proper edge coloring, we might get a graph with adjacent edges same color.
    #    Is that allowed?
    #    "edge-coloured ... so every 4-cycle is rainbow".
    #    If a graph has no 4-cycles, the condition is vacuously true.
    #    So, a star graph with all edges color 0 is valid?
    #    Yes, a star has no 4-cycles.
    #    So, we do NOT need to enforce proper edge coloring globally.
    #    We only need to ensure that IF a 4-cycle exists, its edges are rainbow.
    #    
    #    This changes the conflict graph!
    #    Conflicts are ONLY:
    #    1. Same edge (u, v) with different colors. (We can't have two edges between u and v)
    #    2. Non-rainbow 4-cycle.
    #       This is a hypergraph constraint, not a pairwise conflict.
    #       MWIS on a graph handles pairwise conflicts.
    #       The 4-cycle constraint is a 4-way constraint.
    #       
    #    How to handle 4-way constraints in MWIS?
    #    We can't directly put 4-cycle violations into a pairwise conflict graph easily without explosion.
    #    However, we can observe:
    #    If we select a set of edges S, it is valid if for every 4-cycle C in S, the colors are distinct.
    #    
    #    Alternative Approach:
    #    Since n=20 is small, and we want to maximize edges, maybe we can use a different formulation.
    #    Let's stick to the Greedy approach but improve it?
    #    The directive says "STRUCTURAL change".
    #    
    #    Let's reconsider the "Conflict Graph" idea.
    #    Can we convert the 4-cycle constraint into pairwise conflicts?
    #    No, because the validity depends on the combination of 4 edges.
    #    
    #    However, we can use a Branch and Bound that checks the 4-cycle constraint explicitly at the leaves or during pruning.
    #    But B&B on 3800 nodes is huge.
    #    
    #    Let's look at the "Inspiration 1" and "Current Program".
    #    Current program score 4.26.
    #    Inspiration 1 score 3.36.
    #    
    #    Maybe a better structural change is:
    #    **Simulated Annealing / Local Search on the Color Assignment for a Fixed Topology?**
    #    Or **Local Search on the Topology for a Fixed Coloring Strategy?**
    #    
    #    Let's try a different angle:
    #    The problem is equivalent to finding a subgraph H of K_n and a coloring c: E(H) -> {0..19} such that every C4 in H is rainbow.
    #    
    #    What if we fix the topology to be a known dense structure, like a Turan graph or a bipartite graph, and then optimize the coloring?
    #    Or fix the coloring pattern and optimize the topology?
    #    
    #    Let's try **Local Search on the Edge Set**.
    #    Start with a dense graph (e.g., complete graph or random dense).
    #    Assign random colors.
    #    Count violations (non-rainbow 4-cycles).
    #    Move: Remove an edge involved in a violation, or Change color of an edge involved in a violation.
    #    Goal: Minimize violations, then Maximize edges.
    #    
    #    This is a different optimization method (Local Search / Simulated Annealing) compared to Greedy Construction.
    #    
    #    Let's implement Simulated Annealing.
    #    State: (adj_matrix, color_matrix)
    #    Objective: Maximize (num_edges * 1000 - num_violations)
    #    Actually, we want a VALID graph. So we should penalize violations heavily.
    #    Score = num_edges - 1000 * num_violations.
    #    
    #    Moves:
    #    1. Add a random missing edge with a random color.
    #    2. Remove a random existing edge.
    #    3. Change color of a random existing edge.
    #    
    #    This allows "jumping" over local optima that greedy gets stuck in.
    
    def count_violations(adj, colors):
        """
        Returns the number of non-rainbow 4-cycles.
        adj: n x n, 1 if edge exists, 0 otherwise.
        colors: n x n, color of edge, -1 if no edge.
        """
        violations = 0
        # Iterate over all precomputed 4-cycles
        for e1, e2, e3, e4 in cycles:
            u1, v1 = e1
            u2, v2 = e2
            u3, v3 = e3
            u4, v4 = e4
            
            # Check if all edges exist
            if adj[u1, v1] == 0 or adj[u2, v2] == 0 or adj[u3, v3] == 0 or adj[u4, v4] == 0:
                continue
                
            c1 = colors[u1, v1]
            c2 = colors[u2, v2]
            c3 = colors[u3, v3]
            c4 = colors[u4, v4]
            
            if len({c1, c2, c3, c4}) < 4:
                violations += 1
                
        return violations

    def get_score(adj, colors):
        edges = int(np.sum(adj)) // 2
        violations = count_violations(adj, colors)
        # We want to maximize edges, but violations must be 0.
        # If violations > 0, the score is very low.
        return edges - 1000 * violations

    # Initialize
    # Start with a random dense graph? Or a bipartite graph?
    # Bipartite graphs have many 4-cycles.
    # Let's start with a random graph with p=0.5.
    
    best_adj = None
    best_colors = None
    best_score = -np.inf
    
    # Simulated Annealing Parameters
    T_start = 1000.0
    T_end = 1.0
    alpha = 0.999
    max_iterations = 50000
    
    # We will run multiple restarts if time permits.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Init state
        adj = np.zeros((n, n), dtype=int)
        colors = np.full((n, n), -1, dtype=int)
        
        # Random initialization
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    c = random.randint(0, num_colours - 1)
                    colors[u, v] = c
                    colors[v, u] = c
                    
        current_score = get_score(adj, colors)
        best_adj = adj.copy()
        best_colors = colors.copy()
        best_score = current_score
        
        T = T_start
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Choose a move
            move_type = random.randint(0, 2)
            new_adj = adj.copy()
            new_colors = colors.copy()
            
            if move_type == 0: # Add edge
                # Pick random non-edge
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and adj[u, v] == 0:
                        break
                new_adj[u, v] = 1
                new_adj[v, u] = 1
                c = random.randint(0, num_colours - 1)
                new_colors[u, v] = c
                new_colors[v, u] = c
                
            elif move_type == 1: # Remove edge
                # Pick random edge
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and adj[u, v] == 1:
                        break
                new_adj[u, v] = 0
                new_adj[v, u] = 0
                new_colors[u, v] = -1
                new_colors[v, u] = -1
                
            else: # Change color
                # Pick random edge
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and adj[u, v] == 1:
                        break
                old_c = colors[u, v]
                while True:
                    new_c = random.randint(0, num_colours - 1)
                    if new_c != old_c:
                        break
                new_colors[u, v] = new_c
                new_colors[v, u] = new_c
                
            new_score = get_score(new_adj, new_colors)
            delta = new_score - current_score
            
            if delta > 0:
                accept = True
            else:
                accept = random.random() < np.exp(delta / T)
                
            if accept:
                adj = new_adj
                colors = new_colors
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    best_colors = colors.copy()
                    
            T *= alpha
            
        # End SA loop
        if time.time() > deadline:
            break

    # Final Verification and Cleanup
    # Ensure best_adj and best_colors are valid
    if best_adj is None:
        # Fallback: Star
        adj = np.zeros((n, n), dtype=int)
        colors = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 1
            adj[v, 0] = 1
            colors[0, v] = 0
            colors[v, 0] = 0
        best_adj = adj
        best_colors = colors
        
    # Check validity of best solution
    violations = count_violations(best_adj, best_colors)
    if violations > 0:
        # If the SA didn't find a valid solution, try a greedy repair or fallback
        # Greedy repair: Remove edges involved in violations
        # Or just use the greedy result from the previous strategy?
        # Let's do a quick greedy repair.
        # We have best_adj and best_colors.
        # We can try to remove edges that are in violations.
        # But this might be slow.
        # Let's just return the best valid found. If none, return star.
        
        # Actually, let's run the greedy algorithm from the previous program as a fallback
        # if SA fails to find a valid graph.
        # But we want to be different.
        # Let's try a simple local search repair:
        # While violations > 0:
        #   Find a violating cycle.
        #   Remove one edge from it.
        
        adj_repair = best_adj.copy()
        colors_repair = best_colors.copy()
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            viol = count_violations(adj_repair, colors_repair)
            if viol == 0:
                break
                
            # Find one violating cycle
            found = False
            for e1, e2, e3, e4 in cycles:
                u1, v1 = e1
                u2, v2 = e2
                u3, v3 = e3
                u4, v4 = e4
                
                if adj_repair[u1, v1] == 1 and adj_repair[u2, v2] == 1 and adj_repair[u3, v3] == 1 and adj_repair[u4, v4] == 1:
                    c1 = colors_repair[u1, v1]
                    c2 = colors_repair[u2, v2]
                    c3 = colors_repair[u3, v3]
                    c4 = colors_repair[u4, v4]
                    if len({c1, c2, c3, c4}) < 4:
                        # Remove a random edge from this cycle
                        edges_in_cycle = [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]
                        rem_u, rem_v = random.choice(edges_in_cycle)
                        adj_repair[rem_u, rem_v] = 0
                        adj_repair[rem_v, rem_u] = 0
                        colors_repair[rem_u, rem_v] = -1
                        colors_repair[rem_v, rem_u] = -1
                        found = True
                        break
            if not found:
                break
                
        if count_violations(adj_repair, colors_repair) == 0:
            best_adj = adj_repair
            best_colors = colors_repair
        else:
            # Fallback to star
            adj_f = np.zeros((n, n), dtype=int)
            col_f = np.full((n, n), -1, dtype=int)
            for v in range(1, n):
                adj_f[0, v] = 1
                adj_f[v, 0] = 1
                col_f[0, v] = 0
                col_f[v, 0] = 0
            best_adj = adj_f
            best_colors = col_f

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] == 1:
                edges_list.append([u, v, int(best_colors[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
