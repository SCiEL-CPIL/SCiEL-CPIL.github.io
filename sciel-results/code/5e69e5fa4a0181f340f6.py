"""Exact Cover via Maximum Cycle Packing and Matching."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random

DEADLINE_MARGIN = 10  # seconds before 150s hard limit

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges):
    """
    Decompose edges into cycles and single edges.
    Strategy:
    1. Iteratively remove degree-1 vertices (leaves) and their edges (these must be single edges).
    2. The remaining graph has min degree >= 2.
    3. Use a greedy heuristic to extract edge-disjoint cycles:
       - Find a long cycle using a modified DFS/BFS that prefers high-degree nodes.
       - Remove those edges.
       - Repeat until no edges remain.
    4. Any remaining edges (if min degree drops below 2) are added as single edges.
    
    To improve cycle length, we prioritize finding cycles that cover high-degree vertices.
    """
    if not edges:
        return []

    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining_edges = set(edges)

    # Helper to get current degree
    def get_deg(v):
        return len(adj[v])

    # Phase 1: Peel off leaves (degree 1) and isolated vertices (degree 0)
    # These edges cannot be part of any cycle.
    changed = True
    while changed:
        changed = False
        to_remove = []
        for v in list(adj.keys()):
            if len(adj[v]) == 1:
                nb = list(adj[v])[0]
                to_remove.append((v, nb))
            elif len(adj[v]) == 0:
                pass # nothing to remove
            # We don't remove degree 0 here as they have no edges, but we should clean up dict
            # Actually, if degree 0, no edges to remove.
        
        if to_remove:
            changed = True
            for u, v in to_remove:
                if u in adj and v in adj[u]:
                    e = (u, v) if u < v else (v, u)
                    if e in remaining_edges:
                        pieces.append([u, v])
                        remaining_edges.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
            # Clean up empty nodes
            for v in list(adj.keys()):
                if not adj[v]:
                    del adj[v]

    # Phase 2: Extract cycles from the remaining 2-core (or higher)
    # We want to minimize pieces, so we want LONG cycles.
    # A greedy approach: repeatedly find a "good" cycle.
    # "Good" could mean long, or covering high-degree nodes.
    
    # To find a long cycle, we can use a heuristic:
    # Start from a high-degree node, do DFS, try to close the loop.
    # Or, use a simple heuristic: 
    # 1. Pick a vertex v with max degree.
    # 2. Do a DFS from v, keeping a path.
    # 3. If we hit a vertex in the path, we have a cycle.
    # 4. To maximize length, we might prefer to continue if possible, but that's complex.
    # 
    # Alternative: Use a simple "greedy cycle extraction" that works well on random graphs:
    # While edges remain:
    #   Find a cycle using a randomized DFS that tries to go deep.
    #   Remove it.
    
    # Let's implement a robust cycle finder that tries to find long cycles.
    # We will use a randomized approach with multiple restarts if it fails quickly?
    # No, let's stick to a single pass with a good heuristic to save time.
    
    # Heuristic: 
    # 1. Identify vertices with degree >= 2.
    # 2. Pick a random one (or max degree).
    # 3. Perform a DFS, but with a twist: 
    #    We maintain a stack. 
    #    We prefer neighbors that are NOT in the current path (to grow).
    #    If we hit a neighbor IN the path, we found a cycle.
    #    We want the cycle to be as long as possible.
    #    Simple strategy: Just take the first cycle found? It might be short (triangle).
    #    Better: Try to find a cycle that includes a specific "target" high-degree node?
    
    # Let's try a different angle: 
    # The number of pieces = (Total Edges - Edges in Cycles) + (Number of Cycles)
    # Minimize pieces = Maximize (Edges in Cycles - Number of Cycles)
    # Note: A cycle of length L contributes L edges and 1 piece. Net reduction in pieces vs all edges is L-1.
    # So we want to maximize sum(L_i - 1) over all cycles.
    # This is equivalent to maximizing sum(L_i) - count(cycles).
    # Since sum(L_i) is fixed (total edges in cycles), we want to MINIMIZE the number of cycles?
    # Wait.
    # If we have a fixed set of edges covered by cycles, say E_c.
    # Pieces = E_c (if all single) - (sum(L_i - 1)) = E_c - (E_c - k) = k.
    # No, that's wrong.
    # Total Pieces = (Edges not in cycles) + (Number of cycles).
    # Let E_total be total edges.
    # Let E_c be edges in cycles.
    # Edges not in cycles = E_total - E_c.
    # Pieces = (E_total - E_c) + k, where k is number of cycles.
    # We want to minimize this.
    # Since E_total is fixed, we want to minimize (k - E_c).
    # Or maximize (E_c - k).
    # For a cycle of length L, contribution to E_c is L, contribution to k is 1.
    # So we want to maximize sum(L_i - 1).
    # This means we prefer LONG cycles. A cycle of length 10 is better than two cycles of length 5 (10-1=9 vs 5-1+5-1=8).
    # So the goal is to pack the graph into as FEW, as LONG cycles as possible.
    
    # Algorithm:
    # While there are edges with degree >= 2:
    #   Find a cycle that maximizes length? (NP-hard generally, but heuristics work).
    #   Simple heuristic: 
    #     1. Start at a random vertex with deg >= 2.
    #     2. Do a DFS, preferring to go to unvisited nodes.
    #     3. If we close a cycle, check its length.
    #     4. If we can't find a "long" one, maybe try a different start?
    
    # Let's implement a "Greedy Long Cycle" finder:
    # 1. Collect all vertices with deg >= 2.
    # 2. Pick one randomly.
    # 3. DFS with backtracking limit? No, just simple DFS.
    #    To encourage long cycles, we can shuffle neighbors and try to visit many nodes.
    
    found_cycle = True
    while found_cycle and remaining_edges:
        # Check if any vertex has degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            # All remaining edges are bridges or isolated? 
            # If no vertex has deg >= 2, all remaining edges are disjoint or form a path?
            # Actually, if min deg < 2, we should peel again.
            break
        
        found_cycle = False
        # Try a few random starts to find a decently long cycle
        rng = Random(len(remaining_edges)) # Simple seed based on state
        rng.shuffle(candidates)
        
        best_cycle = None
        best_len = 0
        
        for start in candidates[:5]: # Try top 5 candidates
            # DFS to find a cycle
            visited = {start: 0} # node: index in path
            path = [start]
            stack = [(start, iter(sorted(adj[start])))]
            
            while stack:
                node, it = stack[-1]
                advanced = False
                for nb in it:
                    if (node, nb) if node < nb else (nb, node) not in remaining_edges:
                        continue
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cyc_len = len(path)
                        if cyc_len > best_len:
                            best_len = cyc_len
                            best_cycle = path[:]
                        # We could break here to take the first one, but we want longer
                        # However, DFS might find a short one first.
                        # Let's continue searching for a longer one? 
                        # That's expensive. 
                        # Let's just take the first valid cycle found in this DFS?
                        # Or break out of this DFS?
                        # If we break, we might miss a longer cycle in the same DFS tree.
                        # Given time constraints, let's take the first cycle found in this attempt.
                        # But we try multiple starts.
                        break
                    if nb not in visited:
                        visited[nb] = len(path)
                        path.append(nb)
                        stack.append((nb, iter(sorted(adj[nb]))))
                        advanced = True
                        break
                if not advanced:
                    # Backtrack
                    stack.pop()
                    if path:
                        path.pop()
                        # Remove from visited? 
                        # In standard DFS cycle detection, we keep visited to avoid cycles in the tree,
                        # but we need to detect cycles back to start.
                        # This implementation is slightly flawed for finding ALL cycles or the LONGEST.
                        # It finds A cycle.
                    if not path:
                        break
            
            if best_cycle and len(best_cycle) >= 3:
                found_cycle = True
                break # Take the first good start that yields a cycle? 
                # Actually, let's finish the loop to see if another start gives a LONGER cycle?
                # That might be too slow. Let's just take the first one found.
                # To improve, we can prioritize starts with higher degree.
        
        if best_cycle and len(best_cycle) >= 3:
            # Add cycle to pieces
            pieces.append(best_cycle[:])
            # Remove edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)
                if u in adj and v in adj[u]:
                    adj[u].discard(v)
                if v in adj and u in adj[v]:
                    adj[v].discard(u)
            
            # After removing a cycle, degrees change.
            # We might create new leaves.
            # The outer while loop will handle peeling leaves in the next iteration?
            # No, the outer while loop only peels leaves at the very beginning.
            # We need to peel leaves inside the loop or re-check.
            # Let's add a quick peel step here.
            changed = True
            while changed:
                changed = False
                for v in list(adj.keys()):
                    if len(adj[v]) == 1:
                        nb = list(adj[v])[0]
                        e = (v, nb) if v < nb else (nb, v)
                        if e in remaining_edges:
                            pieces.append([v, nb])
                            remaining_edges.discard(e)
                            adj[v].discard(nb)
                            adj[nb].discard(v)
                            changed = True
                for v in list(adj.keys()):
                    if not adj[v]:
                        del adj[v]

    # Phase 3: Handle remaining edges
    # If there are edges left, they are either:
    # 1. Isolated edges (u-v) where u and v are no longer connected to others in a cycle-able way.
    # 2. Part of a path?
    # Since we peeled all leaves, if edges remain, they must be in components with min degree >= 2?
    # But if we just finished a cycle extraction, we might have left edges that are now leaves.
    # The peel step inside the loop handles that.
    # If the loop terminates because `candidates` is empty (no deg >= 2),
    # then all remaining edges are in components with max degree 1? i.e. isolated edges.
    # So we just add them as single edges.
    
    for e in remaining_edges:
        pieces.append(list(e))

    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, spec in enumerate(specs):
        if time.time() - start > DEADLINE_MARGIN - 1:
            # Emergency fallback: just return edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)

    return results[:3]

# EVOLVE-BLOCK-END
