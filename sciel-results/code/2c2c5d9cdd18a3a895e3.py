"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import heapq

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s budget
    N = 180
    
    # 1. Precompute the conflict graph (2-uniform)
    # Edge (u, v) exists if there exists w such that {u, v, w} is a reciprocal triple.
    # To ensure the resulting set is valid for the original problem, we define the graph
    # such that any independent set in this graph is a valid set for the original problem.
    # As analyzed, if we add an edge between the two larger elements (b, c) of every triple {a, b, c},
    # then any independent set cannot contain both b and c, thus cannot contain the triple.
    # This makes the problem a standard Maximum Independent Set (MIS) on a graph.
    
    # Build adjacency list
    adj = [set() for _ in range(N + 1)]
    
    # Generate all triples (a, b, c) with 1/a = 1/b + 1/c, a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, N + 1):
        a_sq = a * a
        # Iterate divisors d of a^2
        # d < e implies b < c
        for d in range(1, int(a_sq**0.5) + 1):
            if a_sq % d == 0:
                e = a_sq // d
                if d == e:
                    continue # b=c, not distinct
                b = a + d
                c = a + e
                if b <= N and c <= N:
                    triples.append((a, b, c))
                    # Add edge between b and c
                    adj[b].add(c)
                    adj[c].add(b)
    
    # 2. Solve Maximum Independent Set on the graph
    # Strategy: Greedy coloring + Local Search (1-swap, 2-swap)
    # Or a simple iterative greedy with restarts.
    
    def greedy_mis(order):
        """Greedy MIS based on vertex order."""
        selected = set()
        for v in order:
            # Check if v can be added
            can_add = True
            for u in adj[v]:
                if u in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(v)
        return selected

    def local_search_mis(initial_set, time_budget):
        """Local search on the graph to improve the independent set."""
        current = set(initial_set)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        start = time.time()
        
        # Precompute neighbors for faster access? adj is already a list of sets.
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            
            # 1-swap: Remove x, add y
            # To maximize size, we should only swap if it allows adding more later?
            # Or just do random swaps to escape local optima?
            # For MIS, a 1-swap (remove x, add y) keeps size same.
            # It only helps if it enables a subsequent addition.
            # So we do: Remove x, Add y, then try to add as many as possible (Greedy fill).
            
            # Iterate over all x in current
            for x in list(current):
                if time.time() - start > time_budget:
                    return best, best_size
                
                # Try removing x
                trial = current - {x}
                
                # Try adding each y not in trial
                # To be efficient, only try y that are not neighbors of trial?
                # No, just try all y.
                for y in range(1, N + 1):
                    if y in trial:
                        continue
                    
                    # Check if y can be added to trial
                    can_add = True
                    for u in adj[y]:
                        if u in trial:
                            can_add = False
                            break
                    
                    if can_add:
                        new_set = trial | {y}
                        # Greedy fill: try to add any other vertex
                        added_any = False
                        for z in range(1, N + 1):
                            if z in new_set:
                                continue
                            # Check if z can be added to new_set
                            can_add_z = True
                            for u in adj[z]:
                                if u in new_set:
                                    can_add_z = False
                                    break
                            if can_add_z:
                                new_set.add(z)
                                added_any = True
                        
                        new_size = len(new_set)
                        if new_size > best_size:
                            best = set(new_set)
                            best_size = new_size
                            current = new_set
                            current_size = new_size
                            improved = True
                            break # Restart outer loop with new current
                if improved:
                    break
            
            # If no improvement via 1-swap + fill, try a random kick?
            # Or just break if no improvement found in full pass.
            
        return best, best_size

    # Initial solution: Greedy on random order
    best_set = set()
    best_size = 0
    
    # Run multiple random restarts of greedy + local search
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.seed(i)
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy
        current = greedy_mis(order)
        
        # Local Search
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            sa_result, sa_size = local_search_mis(current, remaining / 2.0)
            if sa_size > best_size:
                best_set = sa_result
                best_size = sa_size

    # Final safety check: ensure best_set is not empty
    if not best_set:
        # Fallback: just take all large numbers (which have few conflicts)
        # In the graph, large numbers have fewer edges?
        # Edges are (b,c). b,c > a.
        # Large numbers are likely to be b or c.
        # But they might be connected to many small a's? No, a is not in the edge.
        # The edge is between b and c.
        # So large numbers are connected to other large numbers.
        # Small numbers (a) are isolated in this graph?
        # Wait. The edge is (b,c). a is not part of the edge.
        # So a is an isolated vertex in the graph?
        # No, a might be b or c in another triple.
        # Example: a=2, b=3, c=6. Edge (3,6).
        # In triple a=3, b=4, c=12. Edge (4,12).
        # So 3 is in edge (3,6) and (4,12)? No, 3 is b in first, a in second.
        # In the second triple, the edge is (4,12). 3 is not in the edge.
        # So 3 is connected to 6.
        # Is 3 connected to 4? No.
        # So the graph only connects the "larger" elements of triples.
        # Small numbers (like 1, 2, 3) might have few edges if they are rarely the larger element.
        # But 3 is the larger element in (2,3,6).
        # 1 is never the larger element (since a>=1, b>a, c>a).
        # So 1 is isolated.
        # 2 is never the larger element.
        # 3 is larger in (2,3,6).
        # So 1, 2 are isolated. They can always be picked.
        
        # Fallback: Pick all isolated vertices + greedy on rest
        isolated = [v for v in range(1, N+1) if len(adj[v]) == 0]
        best_set = set(isolated)
        # Greedy add rest
        remaining_nodes = [v for v in range(1, N+1) if v not in best_set]
        random.shuffle(remaining_nodes)
        for v in remaining_nodes:
            can_add = True
            for u in adj[v]:
                if u in best_set:
                    can_add = False
                    break
            if can_add:
                best_set.add(v)
        best_size = len(best_set)

    return sorted(best_set)

# EVOLVE-BLOCK-END
