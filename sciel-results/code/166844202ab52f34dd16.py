"""Improved program for cycle decomposition."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_cycles_decomposition(edges):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        remaining = set(edges)
        pieces = []
        
        # Build adjacency with dynamic updates
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Find the best cycle: prefer longer cycles to minimize piece count
            best_cycle = None
            best_len = 0
            
            # Try each edge as a starting point
            for start_edge in sorted(remaining):
                u0, v0 = start_edge
                
                # BFS to find shortest path from v0 to u0 (gives shortest cycle through this edge)
                # But we want LONGER cycles, so use a different strategy:
                # Try to find a long cycle using DFS with backtracking, but limit depth
                visited = {u0}
                stack = [(v0, [u0, v0], 0)]
                found = False
                while stack and not found:
                    node, cur_path, depth = stack.pop()
                    if depth > 25:  # limit search depth
                        continue
                    if node == u0 and len(cur_path) >= 3:
                        # Found a cycle
                        if len(cur_path) > best_len:
                            best_cycle = cur_path
                            best_len = len(cur_path)
                        found = True
                        break
                    for nb in sorted(adj.get(node, []), reverse=True):
                        if (min(node, nb), max(node, nb)) not in remaining:
                            continue
                        if (min(node, nb), max(node, nb)) == (u0, v0):
                            continue
                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, cur_path + [nb], depth + 1))
                
                if found and best_len >= 3:
                    break  # Found a good cycle, stop searching
                
                # Early termination if we've spent too much time on this edge
                if time.time() > deadline:
                    break
            
            if best_cycle and len(best_cycle) >= 3:
                # Remove cycle edges from remaining
                for i in range(len(best_cycle)):
                    a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                    e = (min(a, b), max(a, b))
                    if e in remaining:
                        remaining.discard(e)
                        adj[a].discard(b)
                        adj[b].discard(a)
                pieces.append(best_cycle)
            else:
                # No more cycles found; add all remaining as single edges
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining = set()
                break
        
        # If we hit deadline, add remaining as single edges
        if remaining:
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            pieces = find_cycles_decomposition(edges)
            results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
