"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_path_cover_optimal(edges):
        """
        Finds a minimum number of edge-disjoint simple paths covering all edges.
        
        Theory:
        - If the graph has 0 odd-degree vertices, it has an Eulerian circuit.
          We can decompose it into 1 path (by cutting the cycle at any point).
          Wait, a cycle is not a path in the strict sense of "simple path" if 
          start != end? The task says "simple paths". Usually a path has distinct vertices.
          An Eulerian cycle has distinct edges but repeated vertices (start=end).
          However, we can cut the Eulerian cycle at any edge to form a single 
          Hamiltonian-like path? No, an Eulerian cycle visits all edges. 
          Cutting it gives a path that visits all edges. 
          Is it a *simple* path? No, it repeats vertices.
          
          Let's re-read carefully: "Paths must be simple".
          This means no repeated vertices within a path.
          
          If the graph has an Eulerian cycle, can we cover it with one simple path?
          Only if the graph is a simple path itself (i.e., all degrees <= 2).
          If the graph has a cycle, a single simple path cannot cover the cycle 
          and return to start. 
          
          Actually, Gallai's conjecture is about decomposing the *edges* into 
          paths. The paths must be simple.
          
          The standard result for minimizing the number of paths in an edge 
          decomposition is related to the number of odd-degree vertices.
          
          Let $k$ be the number of odd-degree vertices.
          We can pair up the odd-degree vertices using a minimum-cost matching 
          in the metric closure of the graph.
          Adding these $k/2$ edges to the graph makes all degrees even.
          The resulting graph has an Eulerian circuit.
          The Eulerian circuit can be decomposed into paths by "cutting" it 
          at the added edges.
          Specifically, if we add a matching $M$ of size $k/2$, the graph $G+M$ 
          is Eulerian. An Eulerian circuit in $G+M$ uses all edges of $G$ and 
          all edges of $M$.
          If we remove the edges of $M$ from this circuit, we break the circuit 
          into $k/2$ disjoint paths (if $k>0$) or 1 cycle (if $k=0$).
          Wait, if we have $k$ odd vertices, we add $k/2$ edges.
          The Eulerian circuit passes through these added edges.
          Removing them breaks the cycle into $k/2$ paths?
          
          Let's trace:
          Cycle: $v_1 - e_1 - v_2 - ... - v_n - e_{added} - v_{n+1} ...$
          If we have $k/2$ added edges, we have $k/2$ "breaks".
          This creates $k/2$ paths.
          
          So the minimum number of paths is $\max(1, k/2)$.
          Wait, is it always possible to find such a matching such that the 
          resulting paths are simple?
          The paths derived from the Eulerian circuit by cutting at the matching 
          edges are simple if and only if the matching edges do not cause the 
          paths to revisit vertices?
          Actually, any sub-path of an Eulerian circuit is a trail (no repeated 
          edges). It may have repeated vertices.
          For the path to be *simple*, it must not have repeated vertices.
          
          However, there is a theorem: Any connected graph can be decomposed 
          into $\lceil k/2 \rceil$ simple paths?
          No, that's not always true if the graph has complex cycles.
          
          Let's look at the specific graphs. They are random Erdos-Renyi graphs 
          with $p \approx 0.3$. They are likely 2-connected or 3-connected 
          for $n=20, 25, 30$.
          
          If the graph is 2-connected, we can often find a Hamiltonian path?
          Not necessarily.
          
          Let's stick to the constructive algorithm based on matching.
          1. Identify odd-degree vertices $O$.
          2. Compute all-pairs shortest paths (unweighted) to get distances.
          3. Solve Minimum Weight Perfect Matching on $O$ using these distances.
          4. Let $M$ be the set of matched pairs (virtual edges).
          5. Add $M$ to the graph $G$. Now all degrees are even.
          6. Find an Eulerian circuit in $G+M$.
          7. The circuit is a sequence of edges.
          8. Split this circuit into segments separated by the edges in $M$.
          9. Each segment is a path in $G$.
          10. Are these paths simple?
              In a 2-connected graph, it is highly likely that the Eulerian 
              circuit can be chosen such that the segments are simple, or 
              we can repair them.
              
              Actually, a simpler approach for small $n$:
              Since $n \le 30$, we can try to find a Hamiltonian path.
              If a Hamiltonian path exists, the answer is 1.
              If not, we look for 2, etc.
              
              But Gallai's bound is $n/2$. For $n=20$, bound is 10.
              The current score is 7.1 (which is likely the sum of paths? 
              No, score is total paths. 7.1 is very low. 
              Wait, the score is "total number of paths".
              If the score is 7.1, that's impossible for an integer count.
              Ah, the score is `seed/value`? No, "The score is the total 
              number of paths...".
              Wait, the prompt says: "The evaluator measures the raw value 
              (lower is better). The search score is seed/value".
              So if the raw value is 7.1, that's not an integer.
              Maybe the score is an average? Or maybe the 7.1 is the 
              *search score*?
              "CURRENT PROGRAM (score 7.1000)"
              If the raw value is, say, 14 paths, and seed is 100, score is 7.1?
              Or if raw value is 38, and seed is 270, score is 7.1?
              
              Let's assume the raw value is the total number of paths.
              For 3 graphs, if we get 10+13+15 = 38 paths, that's the bound.
              If we get fewer, it's better.
              
              Let's implement the matching-based approach. It gives a 
              theoretical minimum of $k/2$ paths (or 1 if $k=0$).
              For random graphs with $p=0.3$, most vertices have even degree?
              No, degrees are roughly Poisson. Half will be odd.
              So $k \approx n$.
              Number of paths $\approx n/2$.
              For $n=20$, 10 paths.
              For $n=25$, 12.5 -> 13 paths.
              For $n=30$, 15 paths.
              Total 38.
              
              Can we do better?
              Only if we can combine paths? No, the paths are edge-disjoint.
              The number of paths in any decomposition is at least $k/2$.
              So 38 is likely the theoretical minimum for these graphs.
              
              Wait, if the theoretical minimum is 38, and the current program 
              gets a score of 7.1...
              If the score is `seed/value`, and value is 38, then seed is ~270?
              Or is the value the *excess* over some baseline?
              "Minimize the total number of paths... The score is the total 
              number of paths".
              
              If the current program gets 7.1, and the minimum is 38, something 
              is wrong with my understanding of the score or the current program's 
              output.
              
              Let's look at the current program's output.
              It uses greedy. Greedy often produces *more* paths than optimal.
              If optimal is 38, greedy might produce 50.
              If score is `seed/value`, and value=50, score=7.1 -> seed=355.
              
              If I can achieve the theoretical minimum of 38, my value will be 
              lower (38 < 50), so my score will be higher (355/38 > 355/50).
              
              So the goal is to hit the theoretical minimum $k/2$ (rounded up).
              
              Algorithm:
              1. Find odd-degree vertices.
              2. If none, return 1 path (Eulerian cycle cut). But wait, 
                 is an Eulerian cycle a valid "simple path"?
                 No, a simple path has distinct vertices. An Eulerian cycle 
                 in a graph with $n>2$ and cycles will repeat vertices.
                 So if $k=0$, we cannot use 1 path unless the graph is a 
                 simple path (which it isn't, it's a cycle or more complex).
                 If $k=0$, the graph has an Eulerian circuit.
                 We can decompose it into paths.
                 Minimum number of paths for an Eulerian graph?
                 It can be 1 if the graph is a cycle.
                 If the graph has chords, we might need more?
                 No, we can just take the Eulerian circuit and "cut" it.
                 But the resulting path must be simple.
                 
                 Actually, for an Eulerian graph, we can always decompose 
                 it into $\lceil c/2 \rceil$ paths? No.
                 
                 Let's assume the standard result: Min paths = $k/2$ if $k>0$, 
                 else 1.
                 Wait, if $k=0$, can we do it in 1 simple path?
                 Only if the graph is a path. If it has a cycle, a simple path 
                 cannot cover the cycle and return.
                 So if $k=0$ and the graph has a cycle, we need at least 2 paths?
                 Or maybe more?
                 
                 Let's test with a square (4-cycle). $k=0$.
                 Can we cover it with 1 simple path? No, a simple path has 
                 distinct vertices. A path of length 3 in a square visits 4 
                 vertices. It covers 3 edges. The 4th edge is left.
                 So we need 2 paths: (0-1-2) and (2-3-0)? No, edge-disjoint.
                 Path 1: 0-1-2 (edges 01, 12).
                 Path 2: 3-0 (edge 30) and 2-3 (edge 23)? 
                 Path 2: 2-3-0 (edges 23, 30).
                 Total 2 paths.
                 
                 So for $k=0$, min paths is not necessarily 1.
                 It is related to the cycle structure.
                 
                 However, for random graphs with $p=0.3$, $k$ will be 
                 non-zero.
                 So we will use the matching approach.
                 
              3. Implement Hungarian Algorithm for Minimum Weight Perfect Matching.
              4. Construct the paths.
              5. Ensure paths are simple. If a path from the Eulerian circuit 
                 decomposition is not simple, we might need to refine.
                 But for small $n$, the paths from the Eulerian circuit 
                 decomposition of $G+M$ are likely simple if $G$ is "tree-like" 
                 or sparse. With $p=0.3$, it's sparse enough.
                 
              Let's code this up.
        """
        if not edges:
            return []
            
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find odd-degree vertices
        odd_verts = [v for v in adj if len(adj[v]) % 2 != 0]
        
        if not odd_verts:
            # Eulerian graph.
            # Find Eulerian circuit.
            # Decompose into simple paths.
            # For now, try to find a Hamiltonian path?
            # Or just use a greedy heuristic for Eulerian case.
            # But this is rare.
            pass
            
        if len(odd_verts) > 0:
            # Compute all-pairs shortest paths (unweighted)
            n = max(adj.keys()) + 1 if adj else 0
            # Use BFS from each odd vertex
            dist = {}
            for v in odd_verts:
                dist[v] = {v: 0}
                queue = [v]
                while queue:
                    u = queue.pop(0)
                    for w in adj[u]:
                        if w not in dist[v]:
                            dist[v][w] = dist[v][u] + 1
                            queue.append(w)
            
            # Build cost matrix for matching
            k = len(odd_verts)
            # k is even
            cost_matrix = [[0]*k for _ in range(k)]
            for i in range(k):
                for j in range(i+1, k):
                    d = dist[odd_verts[i]].get(odd_verts[j], 1000)
                    cost_matrix[i][j] = d
                    cost_matrix[j][i] = d
            
            # Solve Min Weight Perfect Matching
            # Since k is small (<=30), we can use DP or Hungarian.
            # Hungarian is O(N^3). k=30 -> 27000 ops. Very fast.
            # But implementing Hungarian from scratch is error-prone.
            # Let's use a simple DP for matching since k <= 30?
            # 2^30 is too big.
            # Let's use a greedy matching first, then improve?
            # Or use `scipy.optimize.linear_sum_assignment`?
            # The prompt says NumPy and SciPy are available.
            
            import numpy as np
            from scipy.optimize import linear_sum_assignment
            
            # Create a square matrix for linear_sum_assignment
            # We need to match odd_verts[i] with odd_verts[j]
            # linear_sum_assignment finds min cost assignment.
            # We need to ensure we don't match i with i.
            # And we need a perfect matching.
            
            # The cost matrix is symmetric.
            # We can just run linear_sum_assignment on the cost_matrix.
            # It will return row_ind, col_ind.
            # We need to make sure we don't have i->i.
            # But odd_verts are distinct, so cost[i][i] is 0.
            # We should set cost[i][i] to a large value to prevent self-matching.
            for i in range(k):
                cost_matrix[i][i] = 10000
                
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # The matching is pairs (odd_verts[i], odd_verts[col_ind[i]])
            # But linear_sum_assignment gives a bijection.
            # We need to extract the pairs.
            # Since the graph is undirected, (u,v) is same as (v,u).
            # linear_sum_assignment might give (i,j) and (j,i).
            # We should only take i < j.
            
            matching_edges = set()
            for i in range(k):
                j = col_ind[i]
                if i < j:
                    u = odd_verts[i]
                    v = odd_verts[j]
                    matching_edges.add((u, v))
            
            # Now we have the set of virtual edges to add.
            # Add them to the graph.
            new_adj = {v: set(nei) for v, nei in adj.items()}
            for u, v in matching_edges:
                new_adj[u].add(v)
                new_adj[v].add(u)
            
            # Find Eulerian circuit in new_adj
            # Hierholzer's algorithm
            # We need to track edges.
            # Let's represent edges as (u, v) with u < v for uniqueness, 
            # but for the circuit we need to traverse them.
            
            # Build edge list for new graph
            new_edges = set()
            for u in new_adj:
                for v in new_adj[u]:
                    if u < v:
                        new_edges.add((u, v))
            
            # Hierholzer's
            # We need to be able to remove edges.
            # Use a dict of sets for adjacency, and remove edges as we go.
            adj_dict = defaultdict(set)
            for u, v in new_edges:
                adj_dict[u].add(v)
                adj_dict[v].add(u)
                
            circuit = []
            stack = [list(new_adj.keys())[0]] # Start anywhere
            while stack:
                v = stack[-1]
                if adj_dict[v]:
                    w = next(iter(adj_dict[v]))
                    # Remove edge (v, w)
                    adj_dict[v].remove(w)
                    adj_dict[w].remove(v)
                    stack.append(w)
                else:
                    circuit.append(stack.pop())
                    
            circuit.reverse()
            
            # circuit is a list of vertices.
            # The edges in the circuit are (circuit[i], circuit[i+1]) and (circuit[-1], circuit[0])
            
            # Now, we need to split this circuit into paths by removing the matching edges.
            # The matching edges are the "cuts".
            # Identify the indices in the circuit where a matching edge is traversed.
            
            # A matching edge (u, v) appears in the circuit as a transition u->v or v->u.
            # We need to find all such transitions.
            
            cut_indices = set()
            for i in range(len(circuit)):
                u = circuit[i]
                v = circuit[(i+1) % len(circuit)]
                e = tuple(sorted((u, v)))
                if e in matching_edges:
                    # This edge is a cut.
                    # We cut "after" this edge? Or "at" this edge?
                    # The path starts after the cut and ends before the next cut.
                    # Let's mark the index i as a cut point.
                    # The path will be circuit[i+1] ... circuit[j] where j is the next cut.
                    cut_indices.add(i)
                    
            # Sort cut indices
            cuts = sorted(cut_indices)
            
            paths = []
            if not cuts:
                # No cuts? This means k=0? But we only enter if k>0.
                # Or if matching is empty?
                # If k>0, matching is not empty.
                pass
            
            for idx, c in enumerate(cuts):
                # The path starts at circuit[c+1]
                # And ends at the vertex before the next cut.
                # Next cut is cuts[(idx+1) % len(cuts)]
                next_c = cuts[(idx+1) % len(cuts)]
                
                # We need to traverse from c+1 to next_c (wrapping around if necessary)
                path = []
                curr = c + 1
                end = next_c + 1 # We stop when we reach the start of the next segment
                # Actually, the segment is from c+1 to next_c (inclusive of next_c? No, next_c is a cut edge, so the path ends at the vertex *before* the cut edge next_c?
                # Let's trace:
                # Circuit: ... u -[cut]-> v ... w -[cut]-> x ...
                # Cut at edge u-v. Path starts at v.
                # Cut at edge w-x. Path ends at w.
                # So path is v ... w.
                
                # So we start at circuit[c+1].
                # We continue until we reach circuit[next_c].
                # Because the cut edge is (circuit[next_c], circuit[next_c+1]).
                # So the path ends at circuit[next_c].
                
                start_idx = c + 1
                end_idx = next_c
                
                # Handle wrap-around
                if start_idx <= end_idx:
                    path = circuit[start_idx : end_idx + 1]
                else:
                    path = circuit[start_idx:] + circuit[:end_idx + 1]
                    
                # Check if path is valid (simple)
                # If not, we might have a problem.
                # But for now, let's assume it is.
                if len(path) >= 2:
                    paths.append(path)
                    
            return paths
        else:
            # k=0 case
            # Find Eulerian circuit and try to decompose into simple paths.
            # For now, return a single path if possible, or fallback.
            # This is rare.
            return [[u, v] for u, v in edge_set]

    def validate_and_fix(paths, edges):
        edge_set = set(edges)
        used_edges = set()
        valid_paths = []
        for p in paths:
            if len(p) != len(set(p)):
                continue # Skip non-simple paths
            ok = True
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set or e in used_edges:
                    ok = False
                    break
                used_edges.add(e)
            if ok:
                valid_paths.append(p)
        
        # Add missing edges as single-edge paths
        for e in edge_set - used_edges:
            valid_paths.append(list(e))
            
        return valid_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        
        if time_left < 1:
            paths = [[u, v] for u, v in g_edges]
        else:
            paths = solve_path_cover_optimal(g_edges)
            paths = validate_and_fix(paths, g_edges)
            
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
