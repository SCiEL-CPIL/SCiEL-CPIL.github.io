import time
import random
from collections import defaultdict, deque

_T0 = time.time()
_DEADLINE = 140.0
_MARGIN = 10.0

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set(tuple(sorted(e)) for e in zip(order, order[1:]))
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_eulerian(adj):
    """Decompose an even-degree graph into cycles using Hierholzer's algorithm."""
    # Work with a copy
    adj = {u: set(vs) for u, vs in adj.items()}
    cycles = []
    
    while adj:
        # Find a vertex with edges
        start = next(iter(adj))
        while adj[start]:
            start = next(iter(adj))
        if not adj[start]:
            continue
        
        # Hierholzer's
        stack = [start]
        path = []
        while stack:
            u = stack[-1]
            if adj[u]:
                v = next(iter(adj[u]))
                adj[u].discard(v)
                adj[v].discard(u)
                stack.append(v)
            else:
                path.append(stack.pop())
        
        path.reverse()
        # path is a closed walk; extract cycles
        if len(path) >= 3:
            # path[0] == path[-1] for a simple cycle, but Hierholzer gives
            # a closed walk. We need to extract simple cycles.
            # Actually, for an even-degree graph, Hierholzer gives us cycles.
            # The path is a sequence where path[0] = path[-1] if we close it.
            # Let's just use the path as a cycle if it's valid.
            # Actually the path from Hierholzer is a closed walk.
            # Let's trace it properly.
            pass
        
        # Let me redo this more carefully
        # Hierholzer's: we get a closed walk. For an Eulerian circuit,
        # the walk visits every edge exactly once and returns to start.
        # For a general even-degree graph (not necessarily connected),
        # we decompose each connected component.
        
        # Actually, let me just do it differently: repeatedly find cycles.
        break
    
    # Simpler approach: repeatedly find a cycle, remove it, repeat.
    adj = {u: set(vs) for u, vs in adj.items()}
    cycles = []
    
    while True:
        # Find any vertex with degree >= 2
        start = None
        for u in adj:
            if len(adj[u]) >= 2:
                start = u
                break
        if start is None:
            break
        
        # DFS to find a cycle containing start
        # Use iterative DFS with parent tracking
        parent = {start: -1}
        stack = [start]
        found_cycle = None
        
        while stack:
            u = stack[-1]
            if not adj[u]:
                stack.pop()
                continue
            # Find a neighbor to visit
            v = None
            for nb in sorted(adj[u]):
                if nb == parent.get(u, -1):
                    continue
                if nb not in parent:
                    v = nb
                    break
                else:
                    # Back edge - found a cycle
                    # Cycle is from nb to u in the DFS tree
                    cyc = [u]
                    cur = u
                    while cur != nb:
                        cur = parent[cur]
                        cyc.append(cur)
                    if len(cyc) >= 3:
                        found_cycle = cyc
                        break
            if found_cycle:
                break
            if v is not None:
                parent[v] = u
                stack.append(v)
            else:
                stack.pop()
        
        if found_cycle is None:
            # Try a different approach: just walk around
            # This shouldn't happen in an even-degree graph
            break
        
        # Remove the cycle edges
        for i in range(len(found_cycle) - 1):
            a, b = found_cycle[i], found_cycle[i + 1]
            adj[a].discard(b)
            adj[b].discard(a)
        # Remove empty vertices
        adj = {u: vs for u, vs in adj.items() if vs}
        
        cycles.append(found_cycle)
    
    return cycles

def _decompose_eulerian(present, n):
    """
    Decompose graph into cycles + singles.
    Strategy: find a large even-degree subgraph, decompose it into cycles,
    remaining edges become singles.
    """
    if not present:
        return []
    
    best_pieces = None
    best_score = len(present)  # all singles
    
    rng = random.Random(42)
    
    for restart in range(50):
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
        
        # Try to pair up odd vertices by removing edges between them
        # or by adding/removing edges to make degrees even
        # Strategy: for each pair of odd vertices, try to find a path
        # and remove alternating edges to fix parity.
        # Simpler: just remove edges incident to odd vertices to make them even.
        
        # Alternative: find a maximum matching-like pairing of odd vertices
        # where the path between them has odd length (to fix both parities)
        
        # Simplest: greedily remove edges to fix odd degrees
        # For each odd vertex, remove one incident edge (if the other endpoint
        # is also odd, great; otherwise it becomes odd too)
        
        # Better approach: use the fact that we can toggle edges.
        # Find a set of edges to remove such that all degrees become even.
        # This is equivalent to finding a T-join where T is the set of odd vertices.
        
        # For now, let's just try: remove random edges to fix parity
        adj_copy = {u: set(vs) for u, vs in adj.items()}
        
        # Greedy parity fixing: repeatedly find two odd vertices connected by an edge,
        # remove that edge
        fixed = True
        while fixed:
            fixed = False
            odd = [u for u in adj_copy if len(adj_copy[u]) % 2 == 1]
            if not odd:
                break
            for i in range(len(odd)):
                for j in range(i + 1, len(odd)):
                    a, b = odd[i], odd[j]
                    if b in adj_copy[a]:
                        adj_copy[a].discard(b)
                        adj_copy[b].discard(a)
                        fixed = True
                        break
                if fixed:
                    break
        
        # If we still have odd vertices, remove edges from them
        odd = [u for u in adj_copy if len(adj_copy[u]) % 2 == 1]
        for u in odd:
            if adj_copy[u]:
                v = next(iter(adj_copy[u]))
                adj_copy[u].discard(v)
                adj_copy[v].discard(u)
        
        # Now adj_copy should have all even degrees
        # Verify and decompose into cycles
        for u in adj_copy:
            if len(adj_copy[u]) % 2 != 0:
                # Shouldn't happen, but just in case
                adj_copy[u].clear()
        
        adj_copy = {u: vs for u, vs in adj_copy.items() if vs}
        
        # Decompose into cycles
        cycles = _find_cycles_eulerian(adj_copy)
        
        # Build pieces
        used = set()
        pieces = []
        for c in cycles:
            valid = True
            for i in range(len(c) - 1):
                e = (min(c[i], c[i+1]), max(c[i], c[i+1]))
                if e not in present:
                    valid = False
                    break
            if valid and len(c) >= 3:
                pieces.append(c)
                for i in range(len(c) - 1):
                    used.add((min(c[i], c[i+1]), max(c[i], c[i+1])))
        
        # Remaining edges as singles
        for e in present:
            if e not in used:
                pieces.append([e[0], e[1]])
                used.add(e)
        
        score = len(pieces)
        if score < best_score:
            best_score = score
            best_pieces = pieces
        
        if best_score <= len(present) // 2:  # Can't do much better than E/2
            break
    
    if best_pieces is None:
        return [[u, v] for u, v in present]
    return best_pieces

def _local_search_improve(pieces, present, n):
    """
    Try to improve the decomposition by replacing pairs of single edges
    with cycles, or by merging cycles.
    """
    if not pieces:
        return pieces
    
    # Identify single edges (pieces of length 2)
    singles = {}  # edge -> index in pieces
    cycles = []   # (index, cycle)
    
    for i, p in enumerate(pieces):
        if len(p) == 2:
            e = (min(p[0], p[1]), max(p[0], p[1]))
            singles[e] = i
        else:
            cycles.append((i, p))
    
    # Try to find cycles that can replace pairs of single edges
    # For each pair of single edges (u,v) and (w,x), check if there's a
    # path from v to w (or similar) using existing cycle edges or other singles
    # that would form a cycle.
    
    # Simpler: try to find a cycle in the graph formed by single edges + 
    # cycle edges, and if it uses at least 2 single edges, replace them.
    
    # Build adjacency from all pieces
    adj = defaultdict(set)
    for p in pieces:
        for i in range(len(p) - 1):
            u, v = p[i], p[i+1]
            adj[u].add(v)
            adj[v].add(u)
    
    # Find cycles in the full piece graph that use single edges
    # and see if we can merge them
    
    # Actually, let's try a different approach:
    # For each pair of single edges sharing a vertex, try to find a cycle
    # that includes both.
    
    new_pieces = list(pieces)
    improved = True
    max_iter = 200
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        
        # Get current single edges
        singles = {}
        for i, p in enumerate(new_pieces):
            if len(p) == 2:
                e = (min(p[0], p[1]), max(p[0], p[1]))
                singles[e] = i
        
        # For each pair of single edges, try to find a cycle
        single_list = list(singles.keys())
        for i in range(len(single_list)):
            for j in range(i + 1, len(single_list)):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                e1, e2 = single_list[i], single_list[j]
                u1, v1 = e1
                u2, v2 = e2
                
                # Try combinations: (u1-v1) + (u2-v2) + path
                # Possible connections:
                # 1. u1==u2: need path v1 to v2
                # 2. u1==v2: need path v1 to u2
                # 3. v1==u2: need path u1 to v2
                # 4. v1==v2: need path u1 to u2
                # 5. u1==u2 and v1==v2: same edge, skip
                # 6. u1==v2 and v1==u2: same edge reversed, skip
                
                pairs_to_check = []
                if u1 == u2 and v1 != v2:
                    pairs_to_check.append((v1, v2))
                if u1 == v2 and v1 != u2:
                    pairs_to_check.append((v1, u2))
                if v1 == u2 and u1 != v2:
                    pairs_to_check.append((u1, v2))
                if v1 == v2 and u1 != u2:
                    pairs_to_check.append((u1, u2))
                
                if not pairs_to_check:
                    continue
                
                for (a, b) in pairs_to_check:
                    if a == b:
                        continue
                    # Find a path from a to b in the graph excluding e1 and e2
                    # The path should use at least one edge (to form a cycle of length >= 3)
                    # Cycle would be: a - ... - b - (one of the single edges) - a
                    # Actually, the cycle is: a - path - b, then b to a via the two singles
                    
                    # Let's think: if u1==u2, the cycle is u1-v1-...-v2-u2=u1
                    # So we need a path from v1 to v2 not using e1 or e2
                    
                    # BFS/DFS from a to b avoiding the two single edges
                    forbidden = {e1, e2}
                    visited = {a}
                    queue = deque([(a, [a])])
                    found_path = None
                    
                    while queue:
                        node, path = queue.popleft()
                        if node == b:
                            if len(path) >= 2:  # At least one edge
                                found_path = path
                            break
                        for nb in sorted(adj[node]):
                            e = (min(node, nb), max(node, nb))
                            if e in forbidden:
                                continue
                            if nb not in visited:
                                visited.add(nb)
                                queue.append((nb, path + [nb]))
                    
                    if found_path and len(found_path) >= 2:
                        # We found a path from a to b
                        # Construct the cycle
                        # The cycle includes: the two single edges + the path
                        # Total vertices in cycle: len(path) + 1 (the shared vertex)
                        # Wait, let me think more carefully.
                        
                        # If u1 == u2 = s, e1 = (s, v1), e2 = (s, v2)
                        # Path from v1 to v2: v1 - p1 - p2 - ... - v2
                        # Cycle: s - v1 - p1 - ... - v2 - s
                        # That's a valid cycle if all vertices are distinct
                        
                        # Determine which case we're in
                        if u1 == u2:
                            s = u1
                            # Cycle: s, v1, path[1:-1], v2
                            # path is [v1, ..., v2]
                            cycle = [s] + found_path
                            # Check all distinct
                            if len(set(cycle)) == len(cycle) and len(cycle) >= 3:
                                # Replace the two singles with this cycle
                                # Remove the two single edges from new_pieces
                                idx1 = singles[e1]
                                idx2 = singles[e2]
                                # Remove them (mark as None)
                                new_pieces[idx1] = None
                                new_pieces[idx2] = None
                                # Add the cycle
                                new_pieces.append(cycle)
                                improved = True
                                break
                        elif u1 == v2:
                            # e1 = (u1, v1), e2 = (u2, v2) with u1 == v2
                            # Path from v1 to u2
                            # Cycle: u1 - v1 - path - u2 - v2=u1
                            # Wait, u1 == v2, so cycle is: u1 - v1 - ... - u2 - u1
                            # path is [v1, ..., u2]
                            s = u1  # = v2
                            cycle = [s] + found_path
                            if len(set(cycle)) == len(cycle) and len(cycle) >= 3:
                                idx1 = singles[e1]
                                idx2 = singles[e2]
                                new_pieces[idx1] = None
                                new_pieces[idx2] = None
                                new_pieces.append(cycle)
                                improved = True
                                break
                        elif v1 == u2:
                            # e1 = (u1, v1), e2 = (u2, v2) with v1 == u2
                            # Path from u1 to v2
                            # Cycle: u1 - v1=u2 - ... - v2 - u1? No.
                            # e1 connects u1 to v1, e2 connects u2 to v2.
                            # v1 == u2, so: u1 - v1 - (path) - v2, and v1-u2 is the same.
                            # Cycle: u1 - v1 - path[1:] - v2 - u1? 
                            # We need an edge from v2 to u1. Do we have one?
                            # Not necessarily. Let me reconsider.
                            # Actually the cycle should be: u1 - v1 - ... - v2 - ? - u1
                            # We need v2 connected to u1. That's not guaranteed.
                            # Skip this case for now.
                            pass
                        elif v1 == v2:
                            # e1 = (u1, v1), e2 = (u2, v2) with v1 == v2
                            # Path from u1 to u2
                            # Cycle: u1 - v1 - u2 - path - u1? 
                            # Again need u2 to u1 edge. Skip.
                            pass
                    
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Remove None entries
    result = [p for p in new_pieces if p is not None]
    return result

def _decompose(edges):
    """Return list of pieces: cycles (vertex lists len>=3) and single edges."""
    if not edges:
        return []
    
    n = max((max(u, v) for u, v in edges), default=0) + 1
    present = set((min(u, v), max(u, v)) for u, v in edges)
    
    # Phase 1: Initial decomposition
    pieces = _decompose_eulerian(present, n)
    
    # Phase 2: Local search improvement
    if time.time() < _DEADLINE - _MARGIN:
        pieces = _local_search_improve(pieces, present, n)
    
    return pieces

def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
        if time.time() > _DEADLINE - _MARGIN:
            for spec2 in specs[len(results):]:
                e2 = _instance(*spec2)
                results.append([[u, v] for u, v in e2])
            break
    return results
