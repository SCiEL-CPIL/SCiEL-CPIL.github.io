"""Rewritten program using maximum matching to find longest path."""
import random
import heapq
from collections import deque

# Generate the pinned graph using the same seed as specified
def _build_graph():
    rng = random.Random(748)
    n = 100
    p = 1.8 / 100
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
                adj[j].add(i)
    return adj

# Pre-compute the graph once
_ADJ = _build_graph()
_N = 100

def construct():
    """
    Find a simple path in the graph maximizing the number of vertices.
    Uses a combination of:
    1. Longest path via DFS with pruning for small components
    2. Heuristic path extension
    3. Random restarts with different strategies
    """
    n = _N
    adj = _ADJ
    
    # First, try to find a long path using multiple strategies
    
    best_path = []
    
    # Strategy 1: Greedy path extension from every vertex with different tie-breaking
    for start in range(n):
        # Try greedy with minimum degree first
        path = [start]
        seen = {start}
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            # Pick the candidate with minimum degree among unvisited
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        if len(path) > len(best_path):
            best_path = path[:]
        
        # Try greedy with maximum degree first
        path = [start]
        seen = {start}
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            path.append(v)
            seen.add(v)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Strategy 2: DFS-based longest path with memoization for connected components
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # For each component, try to find the longest path
    for comp in components:
        if len(comp) <= len(best_path):
            continue
        
        comp_set = set(comp)
        comp_adj = {v: [u for u in adj[v] if u in comp_set] for v in comp}
        
        # If component is small enough, do exhaustive search
        if len(comp) <= 15:
            # DFS to find longest path
            def dfs_longest(start, visited_set, current_path):
                best_local = current_path[:]
                for nb in comp_adj[start]:
                    if nb not in visited_set:
                        new_path = dfs_longest(nb, visited_set | {nb}, current_path + [nb])
                        if len(new_path) > len(best_local):
                            best_local = new_path
                return best_local
            
            for v in comp:
                p = dfs_longest(v, {v}, [v])
                if len(p) > len(best_path):
                    best_path = p[:]
        else:
            # For larger components, use heuristic approaches
            # Try multiple random walks with backtracking
            for attempt in range(50):
                # Random start
                start = comp[attempt % len(comp)]
                path = [start]
                seen = {start}
                
                # Use a priority queue to explore promising branches
                # We'll do a beam search
                beam = [(path, seen)]
                best_beam_path = path[:]
                
                for _ in range(30):  # Max depth
                    next_beam = []
                    for p, s in beam:
                        current = p[-1]
                        candidates = [v for v in comp_adj[current] if v not in s]
                        if not candidates:
                            if len(p) > len(best_beam_path):
                                best_beam_path = p[:]
                            continue
                        # Sort candidates by heuristic
                        candidates.sort(key=lambda w: len(comp_adj[w] - s))
                        # Take top few
                        for v in candidates[:3]:
                            new_p = p + [v]
                            new_s = s | {v}
                            next_beam.append((new_p, new_s))
                            if len(new_p) > len(best_beam_path):
                                best_beam_path = new_p[:]
                    if not next_beam:
                        break
                    beam = next_beam
                    if len(beam) > 100:
                        # Prune beam
                        beam.sort(key=lambda x: -len(x[0]))
                        beam = beam[:50]
                
                if len(best_beam_path) > len(best_path):
                    best_path = best_beam_path[:]
    
    # Strategy 3: Try to extend the best path found so far
    if best_path:
        # Try extending from both ends
        extended = True
        while extended:
            extended = False
            # Try extending from start
            start = best_path[0]
            for v in adj[start]:
                if v not in set(best_path):
                    best_path = [v] + best_path
                    extended = True
                    break
            # Try extending from end
            if not extended:
                end = best_path[-1]
                for v in adj[end]:
                    if v not in set(best_path):
                        best_path = best_path + [v]
                        extended = True
                        break
    
    # Strategy 4: Try to find a path that covers more vertices using a different approach
    # Use the fact that in sparse graphs, long paths can be found by following
    # low-degree vertices first
    
    for start in range(n):
        path = [start]
        seen = {start}
        # Use a queue-based approach: always go to the unvisited neighbor with fewest options
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            # Sort by number of unvisited neighbors (ascending)
            candidates.sort(key=lambda w: len(adj[w] - seen))
            v = candidates[0]
            path.append(v)
            seen.add(v)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Strategy 5: Try starting from leaves (degree 1 vertices) if any exist
    leaves = [v for v in range(n) if len(adj[v]) == 1]
    for leaf in leaves:
        path = [leaf]
        seen = {leaf}
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            v = candidates[0]
            path.append(v)
            seen.add(v)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Ensure we return a valid non-empty path
    if not best_path:
        # Fallback: return a single vertex
        best_path = [0]
    
    # Final verification
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # This shouldn't happen, but just in case
            return best_path[:i+1] if i > 0 else [best_path[0]]
    
    return best_path
