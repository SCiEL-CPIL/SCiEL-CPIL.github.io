"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing piece count."""
import time
from random import Random
import sys

_DEADLINE = None

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle_heuristic(adj, start, max_depth, rng):
    """
    Find a simple cycle starting from 'start' using a bounded DFS.
    Returns the longest cycle found (list of vertices) or None.
    max_depth limits the search depth to avoid exponential time.
    """
    best_cycle = None
    best_len = 0
    
    # Stack items: (node, path, visited_set_in_path)
    # Using a set for visited is fast for membership, but we need to remove on backtrack.
    # For small graphs (N<=30), a list with 'in' check is O(N) which is fine for depth <= N.
    # Let's use a list for path and a set for quick lookup if needed, or just list since N is small.
    
    stack = [(start, [start])]
    visited_global = {start} # Not strictly needed for simple cycle if we check path, but helps prune if we revisit nodes in different branches? 
                             # Actually, for simple cycle, we just need to not revisit nodes in the *current path*.
    
    while stack:
        node, path = stack.pop()
        current_len = len(path)
        
        # If we have a valid cycle, update best
        # We check neighbors to see if they close the cycle
        # Note: The cycle is closed if a neighbor is in the path (and not the immediate parent to avoid 2-cycles, 
        # but in a simple graph, parent is the only one we came from, so any other match is a cycle).
        # Wait, the standard DFS cycle detection:
        # If neighbor is in path, we found a cycle.
        
        # Optimization: If current path length is already > best_len, we might find a longer one by extending?
        # No, extending doesn't close the cycle. Closing happens when we hit a node in path.
        # So we should check for closure at each step.
        
        # Check for closure
        for nb in adj.get(node, []):
            if nb in path:
                # Found a cycle
                idx = path.index(nb)
                cycle = path[idx:]
                if len(cycle) >= 3:
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
            else:
                # Expand
                if current_len < max_depth:
                    # Randomize neighbors for diversity
                    neighbors = list(adj.get(node, []))
                    rng.shuffle(neighbors)
                    for nb in neighbors:
                        if nb not in path:
                            stack.append((nb, path + [nb]))
                            break # Only expand one branch at a time? No, standard DFS expands all.
                            # But pushing all neighbors creates a huge stack.
                            # Let's do proper DFS: push all unvisited neighbors.
        
        # The above logic is slightly flawed for standard DFS because it pops and then iterates.
        # Let's rewrite with a more standard iterative DFS that tracks parent to avoid backtracking immediately,
        # but allows other paths.
        
    # The previous implementation was a bit messy. Let's do a cleaner bounded DFS.
    # Since N is small (<=30), we can use recursion with a depth limit if we set recursion limit high enough,
    # or iterative. Iterative is safer.
    
    # Revised Iterative DFS for Longest Cycle
    best_cycle = None
    best_len = 0
    max_d = min(max_depth, len(adj) + 5) # Don't go deeper than N
    
    # Stack: (node, path_list)
    # We need to ensure we don't revisit nodes in the current path.
    # We also want to explore different branches.
    
    # A simple backtracking search
    def dfs(node, path, visited):
        nonlocal best_cycle, best_len
        if len(path) > max_d:
            return
            
        # Check if any neighbor closes the cycle
        for nb in adj.get(node, []):
            if nb in visited:
                idx = path.index(nb)
                cycle = path[idx:]
                if len(cycle) >= 3 and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle[:]
            else:
                # Expand
                visited.add(nb)
                path.append(nb)
                dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                
    # Start DFS
    visited = {start}
    dfs(start, [start], visited)
    
    return best_cycle

def _decompose_once(edges, time_limit=None):
    """
    Improved Greedy: Repeatedly find the longest cycle among several candidates, remove it.
    """
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    start_time = time.time()
    rng = Random(42) # Fixed seed for reproducibility of the heuristic
    
    def build_adj():
        adj = {}
        for u, v in remaining:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj
    
    # Estimate max possible cycle length based on edge count
    # A cycle of length L uses L edges.
    # We can't have a cycle longer than the number of vertices involved, or edges available.
    # Let's just cap depth at N.
    
    while remaining:
        if time_limit and time.time() - start_time > time_limit:
            break
            
        adj = build_adj()
        if not adj:
            break
            
        # Find candidates: vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
            
        # Sort candidates by degree descending (heuristic: high degree nodes are likely in larger structures)
        candidates.sort(key=lambda v: len(adj[v]), reverse=True)
        
        best_cycle = None
        best_len = 0
        
        # Try a subset of candidates to find a long cycle
        # Limit attempts to keep it fast
        max_attempts = 20
        attempts = 0
        
        # Also, we can try different random seeds/orders for the DFS
        # The DFS itself uses random shuffle of neighbors.
        
        for cand in candidates[:max_attempts]:
            # Set a depth limit. 
            # A cycle cannot be longer than the number of edges in the connected component, 
            # but practically N is the upper bound.
            # Let's try depth up to len(adj)
            max_depth = len(adj)
            
            # To get variety, we can run the heuristic a few times for the same candidate?
            # The DFS shuffles neighbors, so one run explores a random subset of the tree.
            # Let's run it once per candidate first.
            
            cycle = _find_longest_cycle_heuristic(adj, cand, max_depth, rng)
            
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
                
            # Early exit if we found a "Hamiltonian" like cycle (using all vertices)?
            # Or if we found a very long one?
            if best_len >= len(adj):
                break
                
            attempts += 1
            if attempts >= max_attempts:
                break
        
        if best_cycle is None or len(best_cycle) < 3:
            # No more cycles found?
            # It's possible we missed some due to depth limit or randomness.
            # Let's try a simpler greedy for the rest if we are stuck?
            # Or just break and leave as edges.
            # Given the small size, if we can't find a cycle with degree>=2 nodes, there are no cycles.
            break
            
        # Remove best_cycle edges
        for i in range(len(best_cycle)):
            u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = tuple(sorted((u, v)))
            if e in remaining:
                remaining.discard(e)
                
        pieces.append(best_cycle)
        
    # Remaining edges as single pieces
    for u, v in remaining:
        pieces.append([u, v])
        
    return pieces

def _decompose_improved(edges, time_limit=None):
    """
    Runs the improved decomposition.
    """
    start_time = time.time()
    if time_limit is None:
        time_limit = 10
        
    # The main decomposition
    pieces = _decompose_once(edges, time_limit=time_limit)
    
    # Optional: Post-processing to merge edges into cycles?
    # If we have 3 single edges forming a triangle, we can merge them.
    # This can be done in a loop:
    # 1. Build adj from remaining single edges (if we kept track).
    # But _decompose_once returns all pieces (cycles and edges).
    # We can try to find cycles in the subgraph of single-edge pieces.
    
    # Let's separate single edges and cycles
    single_edges = [p for p in pieces if len(p) == 2]
    cycles = [p for p in pieces if len(p) >= 3]
    
    if single_edges:
        # Build graph from single edges
        adj = {}
        edge_set = set()
        for u, v in single_edges:
            e = tuple(sorted((u, v)))
            edge_set.add(e)
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            
        # Try to find cycles in this residual graph
        # This is the same problem, but on a sparser graph.
        # We can use a simple greedy here.
        residual_pieces = []
        remaining_edges = set(edge_set)
        
        # Simple greedy for residual
        while remaining_edges:
            # Find any cycle
            found = False
            for u in list(adj.keys()):
                if not adj.get(u): continue
                # Quick DFS for any cycle
                # Since graph is sparse, this is fast
                # We can just call _find_longest_cycle_heuristic with small depth
                # Or just a simple DFS
                pass # Skip complex logic here to save time, the main loop should have caught most.
            # If we don't implement this, we just keep them as single edges.
            # Given the main loop runs until no cycles are found in the *full* remaining graph,
            # and we remove cycles, the residual graph should be acyclic (a forest) if the algorithm was perfect.
            # However, the heuristic might miss cycles.
            # Let's do one more pass with a simpler, exhaustive DFS for small components.
            
            # Identify components
            visited = set()
            for start_node in list(adj.keys()):
                if start_node in visited: continue
                # BFS/DFS to get component
                comp_nodes = set()
                stack = [start_node]
                while stack:
                    n = stack.pop()
                    if n in comp_nodes: continue
                    comp_nodes.add(n)
                    visited.add(n)
                    for nb in adj.get(n, []):
                        if nb not in comp_nodes:
                            stack.append(nb)
                
                # Extract edges in this component
                comp_edges = [e for e in remaining_edges if e[0] in comp_nodes]
                if len(comp_edges) < 3:
                    # Too small for a cycle
                    for e in comp_edges:
                        residual_pieces.append(list(e))
                    remaining_edges.difference_update(comp_edges)
                    continue
                    
                # Try to decompose this component
                # It's small. We can use the same heuristic.
                comp_adj = {u: set() for u in comp_nodes}
                for u, v in comp_edges:
                    comp_adj[u].add(v)
                    comp_adj[v].add(u)
                
                # Run a few iterations of cycle finding on this component
                local_remaining = set(comp_edges)
                local_pieces = []
                local_adj = comp_adj
                # Deep copy adj for local use
                local_adj = {k: set(v) for k, v in comp_adj.items()}
                
                while local_remaining:
                    # Find cycle
                    cands = [v for v in local_adj if len(local_adj[v]) >= 2]
                    if not cands:
                        break
                    best_c = None
                    for c in cands[:5]:
                        cyc = _find_longest_cycle_heuristic(local_adj, c, 20, rng)
                        if cyc and (best_c is None or len(cyc) > len(best_c)):
                            best_c = cyc
                    if not best_c:
                        break
                    for i in range(len(best_c)):
                        u, v = best_c[i], best_c[(i+1)%len(best_c)]
                        e = tuple(sorted((u,v)))
                        local_remaining.discard(e)
                    local_pieces.append(best_c)
                    # Update local_adj
                    local_adj = {}
                    for u, v in local_remaining:
                        local_adj.setdefault(u, set()).add(v)
                        local_adj.setdefault(v, set()).add(u)
                        
                for p in local_pieces:
                    residual_pieces.append(p)
                for e in local_remaining:
                    residual_pieces.append(list(e))
                remaining_edges.difference_update(comp_edges) # All handled

        # Replace single_edges with residual_pieces
        pieces = cycles + residual_pieces
        
    return pieces

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        if time.time() > _DEADLINE:
            while len(results) < len(specs):
                results.append([])
            return results
            
        edges = _instance(n, p, seed)
        
        graphs_left = len(specs) - i
        time_left = _DEADLINE - time.time()
        if time_left <= 0:
            while len(results) < len(specs):
                results.append([])
            return results
            
        per_graph = time_left / graphs_left
        budget = min(per_graph, 40) # Give more budget to allow the heuristic to work
        
        pieces = _decompose_improved(edges, time_limit=budget)
        results.append(pieces)
        
    return results
