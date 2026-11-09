"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        adj[i].sort()
    return adj

def _find_ham_cycle(adj, n, deadline, start=0):
    """Find a Hamiltonian cycle starting and ending at 'start' using backtracking."""
    # Fix start to avoid rotation symmetry
    path = [start]
    visited = [False] * n
    visited[start] = True
    remaining = set(range(n))
    remaining.discard(start)
    
    def backtrack():
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last vertex connects to start
            if path[-1] in adj[start]:
                return list(path)
            return None
        
        last = path[-1]
        # Try neighbors of last, ordered by degree in remaining graph
        candidates = [v for v in adj[last] if v in remaining]
        # Heuristic: prefer vertices with fewer remaining options (MRV-like)
        candidates.sort(key=lambda v: len([w for w in adj[v] if w in remaining and w != last]))
        
        for v in candidates:
            # Pruning: if v is not start, check that v still has a path back
            # Simple pruning: if remaining has vertices that would become disconnected
            if len(remaining) == 1:
                # Only one vertex left, it must connect to start
                if v != start and v not in adj[start]:
                    continue
                # Also v must connect to last
                if v not in adj[last]:
                    continue
            
            path.append(v)
            visited[v] = True
            remaining.discard(v)
            
            # Additional pruning: check if any unvisited vertex (except start) has degree 0 in remaining
            # This is expensive, so only do it for larger remaining sets
            if len(remaining) > 2:
                valid = True
                for u in remaining:
                    if u == start:
                        continue
                    deg = 0
                    for w in adj[u]:
                        if w in remaining and w != start:
                            deg += 1
                        elif w == start:
                            # start is the end, so this edge is only usable at the end
                            pass
                    # In the remaining subgraph (excluding start), u needs at least 1 neighbor
                    # unless it's the last vertex to be visited
                    if deg == 0 and len(remaining) > 1:
                        valid = False
                        break
                if not valid:
                    path.pop()
                    visited[v] = False
                    remaining.add(v)
                    continue
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            visited[v] = False
            remaining.add(v)
        
        return None
    
    return backtrack()

def _extract_cycles_from_2regular(adj, n):
    """Extract all cycles from a 2-regular graph. Returns list of cycles."""
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = [i]
            visited[i] = True
            cur = i
            while True:
                nxt = None
                for v in adj[cur]:
                    if not visited[v]:
                        nxt = v
                        break
                if nxt is None:
                    # All neighbors visited, check if we can close the cycle
                    for v in adj[cur]:
                        if v == i:
                            break
                    break
                cycle.append(nxt)
                visited[nxt] = True
                cur = nxt
            if len(cycle) >= 3:
                cycles.append(cycle)
    return cycles

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = _build_adj(edges, n)
    
    # Edge set for tracking
    all_edges = set()
    for u, v in edges:
        all_edges.add(_edge_key(u, v))
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    # Strategy: try different starting vertices and orderings
    
    best_result = None
    
    # Try with start=0 first
    for start in [0, 1, 2]:
        if time.time() > deadline:
            break
        
        # Find first Hamiltonian cycle
        adj1 = [list(x) for x in adj]
        cycle1 = _find_ham_cycle(adj1, n, deadline, start=start)
        if cycle1 is None:
            continue
        
        # Remove cycle1 edges
        used_edges = set()
        for i in range(len(cycle1)):
            u = cycle1[i]
            v = cycle1[(i + 1) % len(cycle1)]
            used_edges.add(_edge_key(u, v))
        
        # Build adj for remaining graph
        adj2 = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if _edge_key(u, v) not in used_edges:
                    adj2[u].append(v)
        
        # Find second Hamiltonian cycle
        cycle2 = _find_ham_cycle(adj2, n, deadline, start=start)
        if cycle2 is None:
            continue
        
        # Remove cycle2 edges
        used_edges2 = set(used_edges)
        for i in range(len(cycle2)):
            u = cycle2[i]
            v = cycle2[(i + 1) % len(cycle2)]
            used_edges2.add(_edge_key(u, v))
        
        # Check remaining graph is a single Hamiltonian cycle
        adj3 = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if _edge_key(u, v) not in used_edges2:
                    adj3[u].append(v)
        
        # Verify each vertex has degree 2
        valid = True
        for u in range(n):
            if len(adj3[u]) != 2:
                valid = False
                break
        
        if not valid:
            continue
        
        # Extract the cycle from the 2-regular graph
        cycles3 = _extract_cycles_from_2regular(adj3, n)
        if len(cycles3) == 1 and len(cycles3[0]) == n:
            cycle3 = cycles3[0]
            # Verify it's a valid cycle
            valid3 = True
            for i in range(len(cycle3)):
                u = cycle3[i]
                v = cycle3[(i + 1) % len(cycle3)]
                if _edge_key(u, v) not in all_edges:
                    valid3 = False
                    break
            if valid3:
                result = [list(cycle1), list(cycle2), list(cycle3)]
                # Convert to ensure pure Python ints
                result = [[int(v) for v in c] for c in result]
                best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: try to find just 2 cycles
    for start in [0, 1]:
        if time.time() > deadline:
            break
        adj1 = [list(x) for x in adj]
        cycle1 = _find_ham_cycle(adj1, n, deadline, start=start)
        if cycle1 is None:
            continue
        used_edges = set()
        for i in range(len(cycle1)):
            u = cycle1[i]
            v = cycle1[(i + 1) % len(cycle1)]
            used_edges.add(_edge_key(u, v))
        
        adj2 = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                if _edge_key(u, v) not in used_edges:
                    adj2[u].append(v)
        
        cycle2 = _find_ham_cycle(adj2, n, deadline, start=start)
        if cycle2 is not None:
            result = [list(cycle1), list(cycle2)]
            result = [[int(v) for v in c] for c in result]
            return result
    
    # Last resort: return one Hamiltonian cycle
    adj1 = [list(x) for x in adj]
    cycle1 = _find_ham_cycle(adj1, n, deadline, start=0)
    if cycle1 is not None:
        return [[int(v) for v in cycle1]]
    
    return [[0, 1, 2]]  # Should not reach here

# EVOLVE-BLOCK-END
