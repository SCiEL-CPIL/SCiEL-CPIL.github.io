"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # 1. Pre-compute the Conflict Graph
    # A conflict exists between edge e1=(u1, v1) and e2=(u2, v2) if they cannot coexist.
    # They cannot coexist if adding both to a valid graph creates a violation.
    # Actually, the violation is defined on the FULL graph.
    # Two edges e1, e2 are in conflict if there is NO valid graph containing both e1 and e2?
    # No, that's too strong.
    # The standard reduction to MIS for "find max subset of items with no conflicts" works if the property is "pairwise compatible".
    # Is "validity" pairwise?
    # If G is valid, and we add e, it becomes invalid. The violation involves a path s-u-v-t.
    # The edges involved in the violation are (s,u), (u,v), (v,t).
    # If we just have {e1, e2}, is it valid?
    # A path of length 3 requires 3 edges.
    # So any graph with < 3 edges is valid.
    # Thus, any pair of edges is compatible (forms a valid graph of size 2).
    # Wait, the problem says "Reject if ANY simple three-edge path...".
    # If the graph has only 2 edges, there are no 3-edge paths, so it is valid.
    # Therefore, the constraint is NOT pairwise. It is a hypergraph constraint (size 3 or more).
    # Specifically, a set of 3 edges {e1, e2, e3} forming a path P3 is invalid if the endpoints have equal degree.
    # But the degree is in the FULL graph.
    # This makes the "static conflict graph" approach tricky because "conflict" depends on the rest of the graph (degrees).
    
    # However, we can still use a Branch and Bound approach on the graph edges directly,
    # but the "bound" needs to be smart.
    
    # Alternative: Use the property that degrees must be distinct for endpoints of any P3.
    # Let's stick to a robust local search / simulated annealing hybrid or a better B&B on the graph structure itself, 
    # but the prompt asks for a "conflict graph" strategy.
    
    # Let's re-read the inspiration: "Model the problem as finding a maximum clique in a conflict graph... using a branch-and-bound".
    # If pairwise compatibility is not sufficient (because constraints are higher-order), a simple MIS on a static conflict graph won't work directly 
    # unless we define conflicts conservatively.
    
    # Let's define a conflict between edge e1 and e2 if they CANNOT be in the same valid graph.
    # As established, any 2 edges are valid. So no pairwise conflicts?
    # Then the conflict graph is empty, MIS is all edges, which is wrong.
    
    # So the "Conflict Graph" strategy must be interpreted differently or I must use a different structural change.
    
    # Let's look at the "Maximum Clique in a conflict graph" inspiration again.
    # Maybe the vertices are not edges, but something else?
    # Or maybe the "conflict" is dynamic?
    
    # Let's try a different structural change: **Genetic Algorithm / Evolutionary Algorithm** on the edge set.
    # Or **Simulated Annealing** (Inspiration 2) but implemented more efficiently with bitsets.
    
    # Actually, let's look at the current program's weakness: it's greedy.
    # A **Simulated Annealing** approach that allows "invalid" states but penalizes them heavily, 
    # using fast bitset-based violation counting, is a strong candidate.
    
    # Let's implement Simulated Annealing with a very fast validity check.
    
    def get_degrees_and_adj(edges):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def count_violations_fast(edges):
        """
        Fast violation count.
        Violation: path s-u-v-t where deg(s) == deg(t).
        For each edge (u,v), let S = {s in N(u)\{v}}, T = {t in N(v)\{u}}.
        Violations += sum_{d} count_S[d] * count_T[d].
        """
        if not edges:
            return 0
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        violations = 0
        # To speed up, we can pre-compute degree counts for neighbors?
        # For each node u, we need counts of degrees of its neighbors.
        # But degrees change.
        
        # Iterate over edges
        for u, v in edges:
            # Get degree counts of neighbors of u (excluding v)
            deg_counts_u = {}
            for s in adj[u]:
                if s != v:
                    d = deg[s]
                    deg_counts_u[d] = deg_counts_u.get(d, 0) + 1
            
            # Check neighbors of v (excluding u)
            for t in adj[v]:
                if t != u:
                    dt = deg[t]
                    if dt in deg_counts_u:
                        violations += deg_counts_u[dt]
                        
        return violations

    def simulated_annealing():
        # Initial state: Random valid graph or empty?
        # Start with a valid graph to ensure we can always return something.
        # Greedy construction for initial state.
        
        def greedy_init():
            current_edges = []
            random.shuffle(all_edges)
            for e in all_edges:
                test = current_edges + [e]
                if count_violations_fast(test) == 0:
                    current_edges = test
            return current_edges

        current_edges = greedy_init()
        current_violations = count_violations_fast(current_edges)
        best_edges = current_edges[:]
        best_score = len(current_edges) # Maximize edges, so score = count if valid, else -inf?
        # We want to maximize edge count among VALID graphs.
        # So fitness = len(edges) if violations == 0 else -len(edges) - violations * 100
        
        def fitness(edges, viol):
            if viol == 0:
                return len(edges)
            return -len(edges) - viol * 100

        best_fitness = fitness(current_edges, current_violations)
        
        T = 100.0
        T_min = 0.01
        cooling = 0.999
        
        # Pre-compute edge indices for faster manipulation?
        # Using lists of tuples is okay for n=15, m=105.
        
        while T > T_min:
            if time.time() > deadline:
                break
            
            # Generate neighbor: Add, Remove, or Swap an edge
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove random edge
                if not current_edges:
                    continue
                idx = random.randint(0, len(current_edges) - 1)
                new_edges = current_edges[:]
                removed = new_edges.pop(idx)
                delta_viol = None # Recalculate
                new_viol = count_violations_fast(new_edges)
            elif move_type < 0.6:
                # Add random edge not in current
                missing = [e for e in all_edges if e not in set(current_edges)]
                if not missing:
                    continue
                new_edge = random.choice(missing)
                new_edges = current_edges + [new_edge]
                new_viol = count_violations_fast(new_edges)
            else:
                # Swap: Remove one, add one
                if len(current_edges) == 0:
                    continue
                idx = random.randint(0, len(current_edges) - 1)
                new_edges = current_edges[:]
                removed = new_edges.pop(idx)
                missing = [e for e in all_edges if e not in set(new_edges)]
                if not missing:
                    continue
                new_edge = random.choice(missing)
                new_edges.append(new_edge)
                new_viol = count_violations_fast(new_edges)
            
            new_fitness = fitness(new_edges, new_viol)
            delta_f = new_fitness - fitness(current_edges, current_violations)
            
            if delta_f > 0 or random.random() < pow(2.71828, delta_f / T):
                current_edges = new_edges
                current_violations = new_viol
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_edges = new_edges[:]
            
            T *= cooling
            
        return best_edges

    best_edges = simulated_annealing()
    
    # Final Polish: Try to add any remaining edge that keeps it valid
    if best_edges:
        best_set = set(best_edges)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for e in all_edges:
                if e in best_set:
                    continue
                new_edges = best_edges + [e]
                if count_violations_fast(new_edges) == 0:
                    best_edges = new_edges
                    best_set.add(e)
                    improved = True
                    if time.time() > deadline:
                        return [list(x) for x in best_edges]
    
    if not best_edges:
        best_edges = [(0, i) for i in range(1, n)]

    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
