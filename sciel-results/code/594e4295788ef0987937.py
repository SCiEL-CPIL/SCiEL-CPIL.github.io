"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        # Helper to find a cycle starting from a node in the remaining graph
        def find_cycle(start_node):
            # DFS to find a cycle
            # We want to find any cycle in the current remaining graph
            # Let's do a BFS/DFS from a random node with degree > 0
            visited = {start_node: None}
            stack = [start_node]
            parent = {start_node: None}
            
            while stack:
                if time.time() > deadline:
                    return None
                node = stack.pop()
                for neighbor in list(adj[node]):
                    if (min(node, neighbor), max(node, neighbor)) not in remaining_edges:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = node
                        parent[neighbor] = node
                        stack.append(neighbor)
                    elif neighbor != parent[node]:
                        # Found a cycle
                        # Reconstruct path
                        cycle_nodes = [node]
                        curr = node
                        while curr != neighbor:
                            curr = parent[curr]
                            cycle_nodes.append(curr)
                        # The cycle is cycle_nodes + [neighbor] (start of the back edge)
                        # Actually, the cycle is the path from neighbor to node via parents, plus the edge node-neighbor
                        # Let's trace back from node to neighbor
                        path = []
                        curr = node
                        while curr != neighbor:
                            path.append(curr)
                            curr = parent[curr]
                        path.append(neighbor)
                        return path
            return None

        # Greedy strategy:
        # 1. While there are edges left:
        #    a. Find a node with degree >= 2 in the remaining graph.
        #    b. Try to find a cycle starting from it (or anywhere).
        #    c. If a cycle is found, add it to pieces and remove its edges.
        #    d. If no cycle is found, the remaining graph is a forest. Add all edges as single pieces.
        
        # To speed up, we can just run a global DFS/BFS on the remaining graph to find ANY cycle.
        
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find any cycle in the remaining graph
            # We can just pick an arbitrary edge and try to find a cycle containing it?
            # Or simpler: iterate through nodes, run DFS.
            cycle_found = False
            # To make it deterministic and fast, let's just run DFS from the smallest node with degree > 0
            # But DFS might not find a cycle if we don't check cross edges properly.
            # Standard DFS cycle detection:
            
            visited_global = set()
            cycle = None
            
            for u in sorted(adj.keys()):
                if u in visited_global:
                    continue
                if not any((min(u, v), max(u, v)) in remaining_edges for v in adj[u]):
                    continue
                
                # DFS
                stack = [(u, iter(sorted(adj[u])))]
                visited_global.add(u)
                parent = {u: None}
                path = [u]
                
                while stack:
                    node, neighbors = stack[-1]
                    found_next = False
                    for neighbor in neighbors:
                        if (min(node, neighbor), max(node, neighbor)) not in remaining_edges:
                            continue
                        if neighbor not in visited_global:
                            visited_global.add(neighbor)
                            parent[neighbor] = node
                            path.append(neighbor)
                            stack.append((neighbor, iter(sorted(adj[neighbor]))))
                            found_next = True
                            break
                        elif neighbor != parent[node]:
                            # Cycle found
                            # Path from neighbor to node in current stack
                            # Find index of neighbor in path
                            idx = path.index(neighbor)
                            cycle = path[idx:]
                            # Note: path contains nodes, we need to close the loop
                            # The cycle is cycle + [neighbor] (implied edge)
                            # Wait, the path is the tree path. The back edge is node-neighbor.
                            # So the cycle is path[idx:] + [neighbor]
                            cycle = path[idx:]
                            # Actually, the cycle vertices are path[idx:] and the edge (node, neighbor) closes it.
                            # So the list of vertices for the cycle piece is path[idx:]
                            # Let's verify: path[idx] is neighbor. path[-1] is node.
                            # Edges: (path[i], path[i+1]) for i in idx..len-2, and (node, neighbor).
                            # This is a valid cycle.
                            break
                    if cycle:
                        break
                    if not found_next:
                        stack.pop()
                        if len(path) > 0:
                            path.pop()
                
                if cycle:
                    break
            
            if cycle:
                # Add cycle to pieces
                pieces.append(cycle)
                # Remove edges from remaining_edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    remaining_edges.discard((min(u, v), max(u, v)))
                # Update adj? Not strictly necessary if we check remaining_edges,
                # but updating adj speeds up neighbor iteration.
                # For simplicity and correctness with the 'visited_global' logic above,
                # we should probably just restart the search if we modify the graph significantly?
                # No, the while loop restarts the outer while, so visited_global is reset.
                # But we need to ensure adj reflects remaining edges for the next iteration?
                # The inner DFS checks `if (min(node, neighbor), max(node, neighbor)) not in remaining_edges: continue`
                # So it's safe to keep adj static (full graph) and filter by remaining_edges.
                pass
            else:
                # No cycle found in remaining graph. It's a forest.
                # Add all remaining edges as single pieces
                for u, v in remaining_edges:
                    pieces.append([u, v])
                remaining_edges = set()
                break
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback: just return trivial decomposition for remaining specs to avoid crash?
            # Or break and return what we have?
            # The task expects 3 lists. If we run out of time, we should probably
            # return the trivial decomposition for the ones we haven't processed to ensure validity,
            # or just return empty lists? No, must cover all edges.
            # Let's just do the trivial one for the rest if time is tight.
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
