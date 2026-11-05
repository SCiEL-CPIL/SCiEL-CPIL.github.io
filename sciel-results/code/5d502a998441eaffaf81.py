import time
import random
from collections import defaultdict, deque

_T0 = time.time()
_DEADLINE = 148.0
_MARGIN = 5.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set(tuple(sorted(e)) for e in zip(order, order[1:]))
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose_eulerian(adj):
    """
    Decompose an even-degree graph into simple cycles.
    Returns list of cycles (each cycle is a list of vertices [v0, v1, ..., vk]
    where edges (v0,v1), (v1,v2), ..., (vk,v0) are in the graph).
    """
    adj = {u: set(vs) for u, vs in adj.items()}
    cycles = []
    
    # Remove isolated vertices
    adj = {u: vs for u, vs in adj.items() if vs}
    
    while adj:
        # Find a vertex with degree >= 2
        start = None
        for u in adj:
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
        
        # Find a cycle containing start using DFS
        parent = {start: None}
        stack = [(start, iter(sorted(adj[start])))]
        order = [start]
        found_cycle = None
        
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if v == parent.get(u):
                    continue
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    stack.append((v, iter(sorted(adj[v]))))
                    advanced = True
                    break
                else:
                    # Back edge found - extract cycle
                    idx = order.index(v)
                    cyc = order[idx:]
                    if len(cyc) >= 3:
                        found_cycle = cyc
                        break
            if found_cycle:
                break
            if not advanced:
                stack.pop()
        
        if found_cycle is None:
            # Should not happen in even-degree graph, but safety break
            break
        
        # Remove cycle edges from adj
        for i in range(len(found_cycle)):
            a = found_cycle[i]
            b = found_cycle[(i + 1) % len(found_cycle)]
            adj[a].discard(b)
            adj[b].discard(a)
        
        # Remove vertices with no edges
        adj = {u: vs for u, vs in adj.items() if vs}
        
        cycles.append(found_cycle)
    
    return cycles

def _decompose(edges):
    """
    Return list of pieces: cycles (vertex lists len>=3) and single edges.
    Strategy: find a large even-degree subgraph, decompose it into cycles,
    remaining edges become singles.
    """
    n = max((max(u, v) for u, v in edges), default=0) + 1
    present = set((min(u, v), max(u, v)) for u, v in edges)
    if not present:
        return []
    
    best_pieces = None
    best_score = len(present)  # all singles
    
    rng = random.Random(42)
    
    for restart in range(300):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Build a subgraph with all even degrees
        # Start with all edges, then remove edges to fix odd degrees
        adj = defaultdict(set)
        for u, v in present:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find odd-degree vertices
        odd = [u for u in adj if len(adj[u]) % 2 == 1]
        
        # Try to fix parity by removing edges between odd vertices
        # Greedy: repeatedly find two odd vertices connected by an edge, remove it
        fixed = True
        while fixed:
            fixed = False
            odd = [u for u in adj if len(adj[u]) % 2 == 1]
            if not odd:
                break
            for i in range(len(odd)):
                for j in range(i + 1, len(odd)):
                    a, b = odd[i], odd[j]
                    if b in adj[a]:
                        adj[a].discard(b)
                        adj[b].discard(a)
                        fixed = True
                        break
                if fixed:
                    break
        
        # If we still have odd vertices, remove random incident edges
        odd = [u for u in adj if len(adj[u]) % 2 == 1]
        for u in odd:
            if adj[u]:
                v = next(iter(adj[u]))
                adj[u].discard(v)
                adj[v].discard(u)
        
        # Remove isolated vertices
        adj = {u: vs for u, vs in adj.items() if vs}
        
        # Check if all degrees are even
        all_even = all(len(vs) % 2 == 0 for vs in adj.values())
        if not all_even:
            # Skip this restart if we couldn't fix parity
            continue
        
        # Decompose the even-degree subgraph into cycles
        cycles = _decompose_eulerian(adj)
        
        # Build pieces: cycles + leftover singles
        used = set()
        pieces = []
        
        for c in cycles:
            # Verify cycle edges are in present
            ok = True
            for i in range(len(c)):
                a = c[i]
                b = c[(i + 1) % len(c)]
                e = (min(a, b), max(a, b))
                if e not in present:
                    ok = False
                    break
            if ok and len(c) >= 3:
                pieces.append(c)
                for i in range(len(c)):
                    a = c[i]
                    b = c[(i + 1) % len(c)]
                    used.add((min(a, b), max(a, b)))
        
        # Leftover edges as singles
        for e in present:
            if e not in used:
                pieces.append([e[0], e[1]])
                used.add(e)
        
        score = len(pieces)
        if score < best_score:
            best_score = score
            best_pieces = pieces
        
        if best_score == len(present):
            break
    
    if best_pieces is None:
        return [[u, v] for u, v in edges]
    return best_pieces

def construct():
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
        if time.time() > _DEADLINE - _MARGIN:
            # Fill remaining with all-singles to keep interface valid
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                e2 = _instance(*spec2)
                results.append([[u, v] for u, v in e2])
            break
    return results
