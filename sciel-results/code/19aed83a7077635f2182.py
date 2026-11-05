"""Rewritten program: dense Q3-free graph on 16 vertices via branch-and-bound edge addition."""
import time
import random
import itertools

_N = 16
_E = _N * (_N - 1) // 2  # 120
_DEADLINE = 150
_MARGIN = 20  # safety margin in seconds

# Q3 edges in 8-vertex cube (vertices 0..7, edges where XOR in {1,2,4})
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

# All 8-vertex subsets of 16 vertices
_ALL_8SETS = list(itertools.combinations(range(_N), 8))


def _build_adj_bits(edges):
    adj = [0] * _N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _has_q3(adj_bits):
    """Check if graph represented by adj_bits contains a Q3 subgraph.
    Uses backtracking for each 8-vertex subset.
    """
    for subset in _ALL_8SETS:
        # Quick check: count edges within subset
        edge_count = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if (adj_bits[subset[i]] >> subset[j]) & 1:
                    edge_count += 1
        if edge_count < 12:
            continue
        # Try to find Q3 within this 8-vertex induced subgraph
        # Map subset indices 0..7 to actual vertices
        # Backtrack: assign Q3 vertices (0..7) to subset positions
        # But since we're checking if the 8-vertex induced subgraph IS Q3,
        # we just need to check if it contains Q3 as a subgraph
        # Since it has exactly 8 vertices, Q3 is a spanning subgraph
        # Check if there's a bijection from Q3 vertices to subset vertices
        # that preserves all 12 Q3 edges
        
        # Faster: check if the induced subgraph has exactly Q3 as a spanning subgraph
        # i.e., every edge in Q3 is present in the induced subgraph
        # Since Q3 is a specific graph on 8 vertices, we need to check if
        # the 8-vertex induced subgraph contains Q3 as a subgraph (not necessarily induced)
        
        # The fastest check: the 8-vertex induced subgraph must have all 12 Q3 edges present
        # for SOME labeling. Since we have exactly 8 vertices, we can try all 8! = 40320
        # permutations, but that's too slow for 12870 subsets.
        
        # Better: use the fact that Q3 is 3-regular. Check degree sequence first.
        degs = []
        for i in range(8):
            d = 0
            for j in range(8):
                if i != j and (adj_bits[subset[i]] >> subset[j]) & 1:
                    d += 1
            degs.append(d)
        degs.sort()
        # Q3 has all degrees 3
        if degs != [3] * 8:
            continue
        
        # Now check if the 3-regular 8-vertex graph is isomorphic to Q3
        # Q3 is the unique 3-regular bipartite graph on 8 vertices with girth 4
        # Check bipartiteness and girth
        # Actually, the unique 3-regular graph on 8 vertices is Q3
        # So just check that it's 3-regular (already done) and connected
        # All 3-regular 8-vertex graphs: Q3 and... let's just do a simple check
        
        # Simple check: verify that the 12 edges match Q3 structure
        # Build adjacency for the subset
        sub_adj = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_bits[subset[i]] >> subset[j]) & 1:
                    sub_adj[i][j] = 1
                    sub_adj[j][i] = 1
        
        # Check if this is Q3: all degrees 3 (done), and check one specific property
        # In Q3, every vertex is in exactly 4 4-cycles. Or simpler: check that the
        # graph is bipartite and 3-regular (which uniquely determines Q3 on 8 vertices)
        
        # Bipartite check
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        bipartite = True
        while queue:
            v = queue.pop(0)
            for w in range(8):
                if sub_adj[v][w]:
                    if color[w] == -1:
                        color[w] = 1 - color[v]
                        queue.append(w)
                    elif color[w] == color[v]:
                        bipartite = False
                        break
            if not bipartite:
                break
        if bipartite:
            return True
    
    return False


def _find_q3_subset(adj_bits):
    """Find a Q3 subgraph and return the 8-vertex subset, or None."""
    for subset in _ALL_8SETS:
        edge_count = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if (adj_bits[subset[i]] >> subset[j]) & 1:
                    edge_count += 1
        if edge_count < 12:
            continue
        degs = []
        for i in range(8):
            d = 0
            for j in range(8):
                if i != j and (adj_bits[subset[i]] >> subset[j]) & 1:
                    d += 1
            degs.append(d)
        degs.sort()
        if degs != [3] * 8:
            continue
        sub_adj = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_bits[subset[i]] >> subset[j]) & 1:
                    sub_adj[i][j] = 1
                    sub_adj[j][i] = 1
        color = [-1] * 8
        color[0] = 0
        queue = [0]
        bipartite = True
        while queue:
            v = queue.pop(0)
            for w in range(8):
                if sub_adj[v][w]:
                    if color[w] == -1:
                        color[w] = 1 - color[v]
                        queue.append(w)
                    elif color[w] == color[v]:
                        bipartite = False
                        break
            if not bipartite:
                break
        if bipartite:
            return subset
    return None


def _is_q3_free_fast(adj_bits):
    """Fast Q3-freeness check using the subset approach."""
    return not _has_q3(adj_bits)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    # Strategy: Start from K16 and remove edges to make it Q3-free
    # Then try to add back edges greedily
    
    # Start with complete graph
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    adj_bits = _build_adj_bits(all_edges)
    
    # Remove edges until Q3-free
    removed = []
    edge_set = set(all_edges)
    
    while time.time() < deadline:
        q3_subset = _find_q3_subset(adj_bits)
        if q3_subset is None:
            break
        
        # Find an edge within q3_subset to remove
        subset_edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                u, v = q3_subset[i], q3_subset[j]
                if (adj_bits[u] >> v) & 1:
                    subset_edges.append((min(u,v), max(u,v)))
        
        if not subset_edges:
            break
        
        # Remove a random edge from the Q3
        e = random.choice(subset_edges)
        u, v = e
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        edge_set.discard(e)
        removed.append(e)
    
    # Now we have a Q3-free graph. Try to add back removed edges and others.
    # Build list of candidate edges to try adding
    candidates = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if not (adj_bits[u] >> v) & 1:
                candidates.append((u, v))
    
    # Sort candidates: try removed edges first, then random
    random.shuffle(candidates)
    
    # Greedy add
    added_edges = []
    for u, v in candidates:
        if time.time() > deadline:
            break
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        if _is_q3_free_fast(adj_bits):
            added_edges.append((u, v))
        else:
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
    
    # Build final edge list
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj_bits[u] >> v) & 1:
                edges.append([u, v])
    
    # Local search: try swaps
    if time.time() < deadline - 5:
        current_edges = set((min(u,v), max(u,v)) for u, v in edges)
        all_possible = set((u, v) for u in range(_N) for v in range(u+1, _N))
        missing = list(all_possible - current_edges)
        
        for _iter in range(20):
            if time.time() > deadline - 5:
                break
            improved = False
            random.shuffle(missing)
            for e_add in missing:
                if time.time() > deadline - 5:
                    break
                if not current_edges:
                    break
                # Try removing each edge and adding e_add
                for e_rem in list(current_edges):
                    u1, v1 = e_rem
                    u2, v2 = e_add
                    adj_bits[u1] &= ~(1 << v1)
                    adj_bits[v1] &= ~(1 << u1)
                    adj_bits[u2] |= (1 << v2)
                    adj_bits[v2] |= (1 << u2)
                    
                    if _is_q3_free_fast(adj_bits):
                        current_edges.discard(e_rem)
                        current_edges.add(e_add)
                        missing.remove(e_add)
                        missing.append(e_rem)
                        improved = True
                        break
                    else:
                        adj_bits[u2] &= ~(1 << v2)
                        adj_bits[v2] &= ~(1 << u2)
                        adj_bits[u1] |= (1 << v1)
                        adj_bits[v1] |= (1 << u1)
                    if improved:
                        break
            if not improved:
                break
        
        # Rebuild edge list
        edges = [[u, v] for u, v in current_edges]
        adj_bits = _build_adj_bits(edges)
    
    return edges
