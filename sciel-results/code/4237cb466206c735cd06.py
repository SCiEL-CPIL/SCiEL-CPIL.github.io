import time
from random import Random
from collections import deque

# scipy is available in the environment
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def find_2_factor(edges, n):
    """
    Finds a 2-factor (max number of edge-disjoint cycles covering max vertices)
    using Minimum-Cost Maximum Flow.
    Returns:
        cycles: list of cycles (each is a list of vertices)
        remaining_edges: set of edges not in the 2-factor
    """
    if not HAS_SCIPY:
        return [], set(tuple(sorted(e)) for e in edges)
        
    # Map original nodes to flow network nodes
    # Source: 0
    # In-nodes: 1 .. n
    # Out-nodes: n+1 .. 2n
    # Sink: 2n+1
    total_nodes = 2 * n + 2
    source = 0
    sink = 2 * n + 1
    
    # Build sparse matrix for linear_sum_assignment
    # We construct the cost matrix for the bipartite matching between "in" and "out"
    # plus source/sink connections.
    
    # The standard formulation for 2-factor via flow:
    # Minimize sum(cost_e * f_e) subject to flow conservation.
    # Since all edge costs are 1, we want to maximize flow (number of edges in cycles).
    
    # Nodes in the cost matrix:
    # 0: Source
    # 1..n: In-nodes
    # n+1..2n: Out-nodes
    # 2n+1: Sink
    
    # Edges in flow network:
    # 1. Source -> In[u] (cap 2, cost 0)
    # 2. In[u] -> Out[v] (cap 1, cost 1) for each edge (u,v) in graph
    # 3. Out[u] -> Sink (cap 2, cost 0)
    
    # linear_sum_assignment solves: min C @ x where x is permutation matrix (assignment)
    # This corresponds to finding a perfect matching in a bipartite graph.
    # We can model the flow as a matching problem by duplicating nodes?
    # Actually, linear_sum_assignment on a (2n+2) x (2n+2) matrix is not directly
    # the standard flow formulation unless we transform it.
    
    # Better: Use the fact that a 2-factor is a perfect matching in the directed line graph?
    # Or simpler: Just use a standard max flow min cost algorithm if available, 
    # or implement a simple Hungarian-like approach for this specific structure.
    
    # Since we only have standard library + scipy, and scipy.optimize doesn't have
    # a direct max-flow-min-cost that is easy to use for this specific directed graph
    # without constructing a large LP, let's try a different approach.
    
    # Alternative: Randomized Greedy 2-Factor Heuristic
    # This is often effective for sparse graphs.
    
    adj = {u: set() for u in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    remaining = set(tuple(sorted(e)) for e in edges)
    selected_edges = []
    
    # We want to select edges such that every vertex has degree 2 in the selected subgraph.
    # This is a b-matching problem with b=2.
    
    # Greedy approach:
    # 1. While there are edges:
    #    a. Find a vertex with the most available edges.
    #    b. Pick two edges incident to it that don't violate degree constraints of neighbors.
    #    c. Remove them.
    
    # This greedy might get stuck. A better heuristic:
    # Try to find long cycles first.
    
    cycles = []
    # Use a randomized walk to find cycles
    rng = Random(42)
    
    while remaining:
        # Find a vertex with degree >= 2 in remaining
        start = None
        for u in range(n):
            if len(adj[u] & {v for (a,b) in remaining if a==u or b==u}) >= 2:
                start = u
                break
        if start is None:
            break
            
        # DFS to find a cycle
        visited = {start}
        path = [start]
        stack = [(start, iter(adj[start]))]
        found_cycle = None
        
        while stack:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                e = (node, nb) if node < nb else (nb, node)
                if e not in remaining:
                    continue
                if nb == start and len(path) >= 3:
                    found_cycle = path[:]
                    break
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(adj[nb])))
                    advanced = True
                    break
            if found_cycle:
                break
            if not advanced:
                stack.pop()
                if path:
                    visited.discard(path.pop())
        
        if found_cycle:
            cycles.append(found_cycle)
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i+1)%len(found_cycle)]
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
        else:
            break
            
    return cycles, remaining

def merge_cycles(cycles, edges_set):
    """
    Try to merge cycles if they share vertices? No, cycles in a 2-factor are vertex-disjoint.
    Wait, the 2-factor found by flow/matching is vertex-disjoint.
    The "greedy" cycle finding above is NOT guaranteed to produce a 2-factor.
    It produces a set of vertex-disjoint cycles.
    The remaining edges form a forest (if we extracted a 2-factor) or something else.
    
    If we use the greedy method, the remaining graph might still have cycles.
    We should just keep extracting cycles until none exist.
    """
    return cycles

def decompose_graph(edges, n):
    """
    Decompose edges into cycles and single edges.
    """
    if not edges:
        return []
        
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Phase 1: Extract cycles
    # We will repeatedly find a cycle in the remaining graph.
    # To minimize pieces, we want long cycles.
    
    adj = {u: set() for u in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        
    # Sort vertices by degree to start from high-degree nodes
    # This helps find longer cycles
    
    while remaining:
        # Check if any vertex has degree >= 2
        start = None
        for u in range(n):
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
            
        # Find a cycle starting from 'start'
        # Use a randomized DFS with a bias towards unvisited nodes
        # and a limit on depth to avoid getting stuck in small cycles if possible?
        # Actually, finding ANY cycle is good. Longer is better.
        
        # Let's try to find the longest cycle using a heuristic:
        # Randomized restarts of DFS.
        
        best_cycle = None
        rng = Random(123)
        
        for _ in range(10): # 10 restarts
            visited = {start}
            path = [start]
            stack = [(start, iter(list(adj[start])[:]))]
            # Shuffle neighbors for randomness
            for node, neighbors_iter in stack:
                pass # Just to be clear, we need to shuffle the list before iterating
            
            # Proper DFS with shuffling
            stack = [(start, iter(sorted(adj[start], key=lambda x: rng.random())))]
            visited = {start}
            path = [start]
            
            found = False
            while stack and not found:
                node, neighbors = stack[-1]
                advanced = False
                for nb in neighbors:
                    e = (node, nb) if node < nb else (nb, node)
                    if e not in remaining:
                        continue
                    if nb == start and len(path) >= 3:
                        if not best_cycle or len(path) > len(best_cycle):
                            best_cycle = path[:]
                        found = True
                        break
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb], key=lambda x: rng.random()))))
                        advanced = True
                        break
                if found:
                    break
                if not advanced:
                    stack.pop()
                    if path:
                        visited.discard(path.pop())
            
            if best_cycle and len(best_cycle) == n:
                break # Found Hamiltonian cycle
                
        if best_cycle:
            pieces.append(best_cycle)
            # Remove edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i+1)%len(best_cycle)]
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            # No cycle found (should not happen if we checked degree)
            # Or maybe the graph is a tree?
            break
            
    # Phase 2: Remaining edges are a forest. Add them as single edges.
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n)
        results.append(pieces)
        
        if time.time() - start_time > 140:
            # Fallback for remaining graphs
            for spec in specs[specs.index((n,p,seed))+1:]:
                e = instance(*spec)
                results.append([list(ed) for ed in e])
            break
            
    while len(results) < 3:
        results.append([])
        
    return results[:3]

if __name__ == "__main__":
    construct()
