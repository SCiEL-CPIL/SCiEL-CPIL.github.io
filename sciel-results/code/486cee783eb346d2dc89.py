"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # Safety margin under 150s wall-clock
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def greedy_decompose(edges, n, trial_seed):
        """
        Decompose graph into cycles and edges using a randomized greedy DFS approach.
        """
        if not edges:
            return []
            
        rng = Random(trial_seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        def find_cycle(start, target, max_depth=n):
            """
            Find a simple cycle from start to target using DFS.
            Returns the cycle as a list of vertices or None.
            """
            if start == target:
                return None # Need at least 3 vertices for a cycle in this context (simple cycle)
            
            stack = [(start, [start])]
            visited = {start}
            
            while stack:
                node, path = stack.pop()
                
                if node == target and len(path) >= 3:
                    return path
                
                if len(path) > max_depth:
                    continue
                
                # Get neighbors, sorted by degree descending to encourage longer cycles
                nbrs = list(adj[node])
                # Filter to only neighbors in remaining graph (adj is updated, so this is implicit)
                # Sort by degree
                nbrs.sort(key=lambda x: len(adj[x]), reverse=True)
                # Add randomness to break ties and explore different paths
                rng.shuffle(nbrs)
                
                for nb in nbrs:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
            
            return None

        while remaining:
            if time.time() > deadline - 10:
                break
                
            # Pick a random edge to start
            u, v = rng.choice(list(remaining))
            
            # Try to find a cycle containing edge (u,v)
            # We can try start=u, target=v or start=v, target=u
            cycle = find_cycle(u, v)
            if not cycle:
                cycle = find_cycle(v, u)
                
            if cycle:
                # Remove cycle edges from remaining and adj
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    e = (a, b) if a < b else (b, a)
                    remaining.discard(e)
                    adj[a].discard(b)
                    adj[b].discard(a)
                pieces.append(cycle)
            else:
                # No cycle found involving this edge
                pieces.append([u, v])
                e = (u, v) if u < v else (v, u)
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
                
        # Add any remaining edges as singletons (should be empty if loop finished, but safety)
        for e in remaining:
            pieces.append(list(e))
            
        return pieces

    def refine_pieces(pieces, n):
        """
        Local search: Find cycles among the single edges and replace them with cycle pieces.
        """
        single_edges = []
        cycle_pieces = []
        
        for p in pieces:
            if len(p) == 2:
                single_edges.append(tuple(sorted(p)))
            else:
                cycle_pieces.append(p)
                
        if not single_edges:
            return pieces
            
        # Build graph of single edges
        se_adj = defaultdict(set)
        se_set = set(single_edges)
        for u, v in single_edges:
            se_adj[u].add(v)
            se_adj[v].add(u)
            
        # Find cycles in this subgraph
        # We will repeatedly find a cycle, remove its edges, and add the cycle to pieces
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            if not se_set:
                break
                
            # Pick an edge to start cycle search
            u, v = next(iter(se_set))
            
            # Check if u, v are still in se_adj
            if v not in se_adj[u]:
                # Stale edge, remove
                se_set.discard((u,v) if u<v else (v,u))
                continue
                
            # DFS to find a cycle from u to v in se_graph
            def find_se_cycle(start, target):
                stack = [(start, [start])]
                visited = {start}
                while stack:
                    node, path = stack.pop()
                    if node == target and len(path) >= 3:
                        return path
                    nbrs = list(se_adj[node])
                    # Sort by degree
                    nbrs.sort(key=lambda x: len(se_adj[x]), reverse=True)
                    for nb in nbrs:
                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, path + [nb]))
                return None
                
            cycle = find_se_cycle(u, v)
            if not cycle:
                cycle = find_se_cycle(v, u)
                
            if cycle:
                # Replace edges in cycle
                cycle_edges = set()
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    e = (a, b) if a < b else (b, a)
                    cycle_edges.add(e)
                    
                # Remove from se_set and se_adj
                for e in cycle_edges:
                    se_set.discard(e)
                    a, b = e
                    se_adj[a].discard(b)
                    se_adj[b].discard(a)
                    
                cycle_pieces.append(cycle)
                improved = True
            else:
                # No cycle found with this edge. 
                # To avoid infinite loop, we can't just leave it.
                # If no cycle is found, this edge is part of no cycle in the current subgraph.
                # We can remove it from the "candidate" pool for cycle finding, but it remains a single edge piece.
                # For efficiency, we just break out of this specific search attempt?
                # Actually, if find_se_cycle fails, it means u and v are not in any common cycle in the current subgraph.
                # We can mark this edge as "dead" for cycle formation.
                # Simplest: just remove it from se_set? No, it's still a piece.
                # We just need to stop trying to form a cycle with it.
                # Let's remove it from se_adj? No, that breaks other potential cycles.
                # Let's just break the inner loop for this specific edge selection strategy?
                # Better: Shuffle the edge selection. If we always pick the first edge, we might get stuck.
                # Let's just remove this edge from the "active" search set but keep it in the final result.
                # Since we are modifying se_set, and se_set represents edges *available* for cycle formation,
                # if we can't form a cycle with (u,v), we can't form a cycle with *any* edge?
                # No, other edges might form cycles.
                # So we should just skip this edge.
                # To do that, we can remove it from se_set? No, that removes it from the graph.
                # We need a separate "active" set.
                
                # For simplicity and safety, if we fail to find a cycle for the first edge, 
                # we might be in a bipartite or tree-like component. 
                # Let's just break out of the while loop for this refinement pass 
                # and rely on the fact that if there were cycles, we'd likely find one 
                # with a random edge. 
                # Actually, let's just remove this edge from the *candidate* list for the next iteration 
                # by not putting it back? 
                # The current structure re-iterates from the start of se_set.
                # Let's just remove (u,v) from se_set to prevent re-checking it immediately?
                # No, that's dangerous.
                
                # Let's just break out of the refinement loop if we fail on the first pick?
                # No, that's weak.
                
                # Let's try a different edge.
                # We'll just pick a random edge from se_set in the next iteration.
                # To avoid infinite loop on the same edge, we can shuffle se_set?
                # Let's just break out of the inner while loop if we fail?
                # No, let's just remove this specific edge from the consideration for *this* specific cycle search, 
                # but since we restart the loop, we need to ensure progress.
                
                # Simplest robust fix: 
                # If no cycle found, remove this edge from se_adj? 
                # No.
                
                # Let's just break out of the refinement if we fail to find a cycle for the first edge?
                # This might miss cycles.
                
                # Alternative: 
                # Find ALL cycles in the subgraph at once? 
                # Given small size, we can do a full cycle enumeration.
                # But that's complex.
                
                # Let's just break out of the refinement loop. 
                # The greedy step usually handles most cycles. 
                # This is just a cleanup.
                break

        # Reconstruct pieces
        # We need to keep the single edges that were NOT part of any found cycle.
        # The edges in se_set are the ones remaining.
        # The edges removed from se_set are in cycle_pieces.
        
        final_pieces = list(cycle_pieces)
        for e in se_set:
            final_pieces.append(list(e))
            
        return final_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        best_pieces = None
        best_count = float('inf')
        
        # Run multiple trials
        trial_seed = 0
        while time.time() < deadline - 10:
            pieces = greedy_decompose(edges, n, trial_seed)
            # Refine
            refined = refine_pieces(pieces, n)
            count = len(refined)
            
            if count < best_count:
                best_count = count
                best_pieces = refined
                
            trial_seed += 1
            # If we hit a very good score (e.g., close to lower bound), we could stop earlier,
            # but let's just run until time.
            
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)
        
    return results
# EVOLVE-BLOCK-END
