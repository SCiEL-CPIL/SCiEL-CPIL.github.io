"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 140.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycle(adj, n):
    """
    Find any simple cycle in the graph represented by adj.
    Returns a list of vertices forming the cycle, or None if no cycle exists.
    """
    visited = [False] * n
    parent = [-1] * n
    
    def dfs(u):
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                parent[v] = u
                if dfs(v):
                    return True
            elif v != parent[u]:
                # Found a back edge
                # Reconstruct cycle
                cycle = [v]
                curr = u
                while curr != v:
                    cycle.append(curr)
                    curr = parent[curr]
                cycle.append(v)
                # The cycle is v -> ... -> u -> v
                # We need to reverse the part from u back to v to get correct order?
                # Actually, the path is v <- ... <- u.
                # So the cycle is v, ..., u, v.
                # Let's trace:
                # parent chain: u -> ... -> v
                # So the path in the tree is v to u.
                # The back edge is u-v.
                # So the cycle is v, (nodes from v to u in tree), u.
                # The list `cycle` currently has [v, u, ..., v] if we just appended.
                # Let's refine reconstruction.
                
                # Correct reconstruction:
                # We have back edge (u, v).
                # Path in tree from v to u.
                # Let's collect nodes from u back to v.
                path = [u]
                curr = u
                while curr != v:
                    curr = parent[curr]
                    path.append(curr)
                # path is [u, ..., v]
                # So cycle is [v, ..., u]
                # Wait, the edge is (u, v).
                # So the cycle is v, ..., u, v.
                # The vertices are v, ..., u.
                # The list `path` is [u, ..., v].
                # Reversing it gives [v, ..., u].
                # This is the cycle vertices.
                return list(reversed(path))
        return False

    for i in range(n):
        if not visited[i] and adj[i]:
            if dfs(i):
                return True # Actually we need to return the cycle
    return None

# The above DFS is a bit messy with return types. Let's write a cleaner one.

def _find_cycle_clean(adj, n):
    visited = [False] * n
    parent = [-1] * n
    cycle = []

    def dfs(u):
        nonlocal cycle
        visited[u] = True
        for v in adj[u]:
            if not visited[v]:
                parent[v] = u
                if dfs(v):
                    return True
            elif v != parent[u] and parent[v] != -1: # v is visited and not parent
                # Found a cycle
                # Reconstruct
                path = [u]
                curr = u
                while curr != v:
                    curr = parent[curr]
                    path.append(curr)
                # path is [u, ..., v]
                # Cycle is [v, ..., u]
                cycle = list(reversed(path))
                return True
        return False

    for i in range(n):
        if not visited[i] and adj[i]:
            if dfs(i):
                return cycle
    return None


def _decompose(edges, n):
    """
    Decompose the graph into cycles and edges.
    """
    if not edges:
        return []

    # Build initial adjacency list with sets for easy removal
    adj = defaultdict(set)
    remaining_edges = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        remaining_edges.add((min(u, v), max(u, v)))

    pieces = []

    while remaining_edges:
        if time.time() - _start_time > _DEADLINE_SECONDS:
            # Emergency fallback: return all remaining edges as single pieces
            pieces.extend([list(e) for e in remaining_edges])
            break

        # Find a cycle
        cyc = _find_cycle_clean(adj, n)
        
        if cyc:
            # Validate cycle
            # cyc is a list of vertices [v0, v1, ..., vk]
            # Edges are (v0,v1), (v1,v2), ..., (vk, v0)
            # Note: _find_cycle_clean returns [v, ..., u] for back edge (u,v)
            # So the cycle is v -> ... -> u -> v
            # The list is [v, ..., u].
            # The edges are between consecutive elements and the last to first.
            
            # Check if all edges exist in remaining_edges
            valid = True
            cyc_edges = []
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                e = (min(u, v), max(u, v))
                if e not in remaining_edges:
                    valid = False
                    break
                cyc_edges.append(e)
            
            if valid:
                pieces.append(cyc)
                for e in cyc_edges:
                    remaining_edges.remove(e)
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                # This shouldn't happen if adj is consistent, but just in case
                break
        else:
            # No cycle found, remaining graph is a forest
            # All remaining edges are single pieces
            for e in remaining_edges:
                pieces.append(list(e))
            break

    return pieces


_start_time = None

def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    global _start_time
    _start_time = time.time()
    
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - _start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        
        # Check if we have time
        if time.time() - _start_time > _DEADLINE_SECONDS - 10:
            results.append([[u, v] for u, v in edges])
            continue

        pieces = _decompose(edges, n)
        
        # Verification
        if not _verify(edges, pieces, n):
            # Fallback to all single edges
            pieces = [[u, v] for u, v in edges]
        
        results.append(pieces)

    return results

def _verify(edges, pieces, n):
    """Verify that pieces are edge-disjoint and cover all edges."""
    used_edges = set()
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    for piece in pieces:
        if len(piece) == 2:
            u, v = piece
            e = (min(u, v), max(u, v))
            if e in used_edges:
                return False
            used_edges.add(e)
        elif len(piece) >= 3:
            # Check distinct vertices
            if len(set(piece)) != len(piece):
                return False
            for i in range(len(piece)):
                u = piece[i]
                v = piece[(i + 1) % len(piece)]
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    return False
                used_edges.add(e)
        else:
            return False
            
    return used_edges == edge_set
