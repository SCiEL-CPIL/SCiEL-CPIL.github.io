"""
Simulated Annealing for C6-free 4-coloring of complete graphs.
Starts from a random or heuristic coloring, then uses SA to minimize C6 conflicts.
Conflict-focused restarts help escape local minima.
"""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def count_c6_in_color(adj, n, color):
        """Count all 6-cycles in the subgraph of edges with given color."""
        # Build adjacency list for this color
        adj_list = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == color:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
        
        count = 0
        # For each 6-tuple of distinct vertices, check if they form a C6
        # To avoid overcounting, fix vertex 0 as start and require increasing order for other vertices
        # Actually, let's count each C6 exactly once by fixing the smallest vertex and requiring
        # the cycle to be in a canonical form.
        
        # Simpler: iterate over all combinations of 6 vertices, and for each, check all 60 distinct cycles
        # This is expensive but n <= 19 so C(19,6) = 27132, times 60 = ~1.6M checks per color
        # But we can be smarter: for each 6-set, check if the induced subgraph contains a C6
        
        from itertools import combinations
        
        # Precompute the 60 edge sets for C6 on 6 labeled vertices
        # A C6 on vertices (v0,v1,v2,v3,v4,v5) has edges:
        # (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
        # We need to check all ways to map 6 distinct vertices to these positions
        # But many are equivalent. The number of distinct 6-cycles on 6 labeled vertices is 60.
        # Actually, the number of distinct Hamiltonian cycles on 6 labeled vertices is (6-1)!/2 = 60.
        
        # For efficiency, for each 6-set S, check if the induced subgraph on S (in the color-c graph) 
        # contains any C6. We can do this by checking all 60 permutations.
        
        # But this is too slow. Let's use a different approach:
        # For each 6-set, build the 6x6 adjacency matrix and check if it contains a C6.
        
        # Actually, let's just do a DFS-based count for each 6-set.
        # For a 6-set {a,b,c,d,e,f}, a C6 exists if there's a Hamiltonian cycle in the induced subgraph.
        
        # Better approach: for each pair of non-adjacent vertices in the 6-set, check if there's a path of length 4 between them.
        # No, that's also complex.
        
        # Let's just brute force with the 60 cycles.
        
        # Precompute the 60 distinct C6 edge patterns on vertices 0-5
        if not hasattr(count_c6_in_color, '_cycle_edges'):
            from itertools import permutations
            cycle_edges = []
            seen = set()
            for perm in permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycle_edges.append(edges)
            count_c6_in_color._cycle_edges = cycle_edges
        
        cycle_edges = count_c6_in_color._cycle_edges
        
        for subset in combinations(range(n), 6):
            # Build mapping from 0-5 to actual vertices
            mapping = list(subset)
            # Check each cycle pattern
            for edges in cycle_edges:
                all_present = True
                for (u, v) in edges:
                    if mapping[u] not in adj_list[mapping[v]]:
                        all_present = False
                        break
                if all_present:
                    count += 1
                    break  # Count each 6-set at most once
        
        return count

    def total_conflicts(adj, n):
        """Total number of C6s across all 4 colors."""
        return sum(count_c6_in_color(adj, n, c) for c in range(4))

    def get_conflict_edges(adj, n):
        """Return list of edges (i,j) that are part of at least one C6."""
        conflict_edges = set()
        for c in range(4):
            adj_list = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j] == c:
                        adj_list[i].add(j)
                        adj_list[j].add(i)
            
            from itertools import combinations
            if not hasattr(get_conflict_edges, '_cycle_edges'):
                from itertools import permutations
                cycle_edges = []
                seen = set()
                for perm in permutations(range(1, 6)):
                    cycle = (0,) + perm
                    edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                    if edges not in seen:
                        seen.add(edges)
                        cycle_edges.append(edges)
                get_conflict_edges._cycle_edges = cycle_edges
            
            cycle_edges = get_conflict_edges._cycle_edges
            
            for subset in combinations(range(n), 6):
                mapping = list(subset)
                for edges in cycle_edges:
                    all_present = True
                    for (u, v) in edges:
                        if mapping[u] not in adj_list[mapping[v]]:
                            all_present = False
                            break
                    if all_present:
                        # All 6 edges of this C6 are in conflict
                        for (u, v) in edges:
                            conflict_edges.add((min(mapping[u], mapping[v]), max(mapping[u], mapping[v])))
                        break
        
        return conflict_edges

    def simulated_annealing(n, initial_adj, max_iter, start_temp, cool_rate, rng, deadline):
        """Run SA to minimize C6 conflicts."""
        adj = [row[:] for row in initial_adj]
        current_conflicts = total_conflicts(adj, n)
        best_adj = [row[:] for row in adj]
        best_conflicts = current_conflicts
        
        temp = start_temp
        iter_count = 0
        
        while temp > 0.01 and iter_count < max_iter:
            if time.time() > deadline:
                break
            
            # Pick a random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
            
            old_color = adj[i][j]
            new_color = rng.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Compute delta conflicts
            # This is expensive, so we do it incrementally
            # For now, compute full conflicts before and after
            adj[i][j] = new_color
            adj[j][i] = new_color
            new_conflicts = total_conflicts(adj, n)
            delta = new_conflicts - current_conflicts
            
            if delta < 0 or rng.random() < (2.718281828 ** (-delta / temp)):
                current_conflicts = new_conflicts
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_adj = [row[:] for row in adj]
            else:
                adj[i][j] = old_color
                adj[j][i] = old_color
            
            temp *= cool_rate
            iter_count += 1
        
        return best_adj, best_conflicts

    def heuristic_init(n, rng):
        """Create a heuristic initial coloring."""
        adj = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                # Use distance-based coloring
                adj[i][j] = (j - i) % 4
                adj[j][i] = adj[i][j]
        return adj

    def random_init(n, rng):
        """Create a random initial coloring."""
        adj = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = rng.randint(0, 3)
                adj[j][i] = adj[i][j]
        return adj

    best_n = 0
    best_colors = None

    # Try n from 18 down to 10
    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        found = False
        max_sa_time = min(20, deadline - 10 - (time.time() - start_time))
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(attempt * 1000 + n)
            
            # Alternate between heuristic and random init
            if attempt % 2 == 0:
                adj = heuristic_init(n, rng)
            else:
                adj = random_init(n, rng)
            
            initial_conflicts = total_conflicts(adj, n)
            if initial_conflicts == 0:
                best_n = n
                best_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        best_colors.append(adj[i][j])
                found = True
                break
            
            # Run SA with conflict-focused restarts
            start_temp = max(1.0, initial_conflicts * 0.1)
            cool_rate = 0.99
            max_iter = min(5000, int(max_sa_time * 100))
            
            best_adj, best_conf = simulated_annealing(n, adj, max_iter, start_temp, cool_rate, rng, deadline - 10)
            
            if best_conf == 0:
                best_n = n
                best_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        best_colors.append(best_adj[i][j])
                found = True
                break
            
            # Conflict-focused restart: perturb edges in conflict
            if best_conf > 0:
                conflict_edges = get_conflict_edges(best_adj, n)
                if conflict_edges:
                    for (i, j) in list(conflict_edges)[:min(10, len(conflict_edges))]:
                        best_adj[i][j] = rng.randint(0, 3)
                        best_adj[j][i] = best_adj[i][j]
                    # Re-run SA from perturbed state
                    start_temp2 = max(0.5, best_conf * 0.05)
                    best_adj2, best_conf2 = simulated_annealing(n, best_adj, max_iter // 2, start_temp2, cool_rate, rng, deadline - 10)
                    if best_conf2 < best_conf:
                        best_adj, best_conf = best_adj2, best_conf2
                    if best_conf == 0:
                        best_n = n
                        best_colors = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                best_colors.append(best_adj[i][j])
                        found = True
                        break
        
        if found:
            break

    if best_n == 0:
        # Fallback: small valid coloring
        n = 7
        adj = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = (j - i) % 4
                adj[j][i] = adj[i][j]
        best_n = n
        best_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                best_colors.append(adj[i][j])

    return {'n': best_n, 'colors': best_colors}
