"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _build_edges():
    d = 7
    n = 1 << d
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    return edges

def _build_c4_map(edges):
    """Map each edge index to the list of pairs of other edge indices that,
    together with this edge, complete a 4-cycle."""
    n = len(edges)
    # For each edge (u,v), find all 4-cycles containing it.
    # A 4-cycle containing edge (u,v) is: u-v-w-x-u where x = u^a, w = v^a
    # for some a != (u^v bit). Actually: if edge is (u,v) with v = u ^ (1<<i),
    # then for each j != i, the cycle is u, v, v^(1<<j), u^(1<<j).
    # The other three edges are (v, v^(1<<j)), (u^(1<<j), v^(1<<j)), (u, u^(1<<j)).
    # We need to find indices of those edges.
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
    
    # For each edge, find the two "parallel" edges that form 4-cycles with it.
    # Edge (u,v) where v = u ^ (1<<i). For each j != i:
    #   c1 = (u ^ (1<<j), v ^ (1<<j))  -- parallel edge
    #   c2 = (u, u ^ (1<<j))           -- one connecting edge
    #   c3 = (v, v ^ (1<<j))           -- other connecting edge
    # The 4-cycle is: u - v - (v^(1<<j)) - (u^(1<<j)) - u
    # Edges: (u,v), (v, v^(1<<j)), (v^(1<<j), u^(1<<j)), (u^(1<<j), u)
    
    # For a given edge e=(u,v), a 4-cycle is determined by choosing j != i.
    # The three other edges are:
    #   e1 = (v, v ^ (1<<j))
    #   e2 = (u ^ (1<<j), v ^ (1<<j))
    #   e3 = (u, u ^ (1<<j))
    # All four edges form a C4.
    
    # For each edge, store the set of "conflict pairs": two edge indices such that
    # if both are in the chosen set, adding this edge would complete a C4.
    # Actually, for greedy: when considering edge e, it's blocked if there exist
    # j1 != i and j2 != i such that... no.
    # 
    # A C4 containing edge e=(u,v) [v=u^(1<<i)] and using direction j:
    # vertices: u, v, v^(1<<j), u^(1<<j)
    # edges: (u,v), (v, v^(1<<j)), (v^(1<<j), u^(1<<j)), (u^(1<<j), u)
    # So edge e is in a C4 with edges (v, v^(1<<j)) and (u^(1<<j), v^(1<<j)) and (u, u^(1<<j)).
    # For e to complete a C4, the other 3 edges must all be present.
    # But actually, we just need: for some j, all three other edges are present.
    #
    # For efficiency in greedy: precompute for each edge e, the list of "blocking triples".
    # A triple is (a, b, c) of edge indices such that {e, a, b, c} forms a C4.
    # e is blocked if any triple has all three in chosen.
    
    # Better: for each edge e, precompute the list of "conflict pairs".
    # Edge e is in C4s with various other edges. Specifically, for each j != i,
    # the C4 has edges e, e1, e2, e3.
    # e is blocked if e1, e2, e3 are all chosen.
    # But also, e could be blocked via different C4s.
    #
    # For local search, we need: given a set S, which edges can be added?
    # Edge e can be added if no C4 containing e has all other 3 edges in S.
    
    # Let's precompute for each edge e: list of (e1, e2, e3) triples.
    c4_triples = [[] for _ in range(n)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1  # the bit that differs
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            v2 = v ^ bit_j
            u2 = u ^ bit_j
            # C4: u - v - v2 - u2 - u
            e1 = (min(v, v2), max(v, v2))
            e2 = (min(v2, u2), max(v2, u2))
            e3 = (min(u, u2), max(u, u2))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                c4_triples[idx].append((idx1, idx2, idx3))
    
    return c4_triples

def _can_add(e_idx, chosen, c4_triples):
    """Check if edge e_idx can be added to chosen without creating a C4."""
    for (a, b, c) in c4_triples[e_idx]:
        if a in chosen and b in chosen and c in chosen:
            return False
    return True

def _run_greedy(edges, c4_triples, rng):
    """Run greedy with random edge order."""
    order = list(range(len(edges)))
    rng.shuffle(order)
    chosen = set()
    for e_idx in order:
        if _can_add(e_idx, chosen, c4_triples):
            chosen.add(e_idx)
    return chosen

def _local_improve(chosen, edges, c4_triples, rng, time_limit):
    """Try to improve the chosen set via local search.
    Returns improved set."""
    chosen = set(chosen)
    n = len(edges)
    
    # First pass: try to add any edge that can be added
    improved = True
    while improved:
        improved = False
        for e_idx in range(n):
            if e_idx not in chosen and _can_add(e_idx, chosen, c4_triples):
                chosen.add(e_idx)
                improved = True
    
    # Second pass: try 1-edge swaps: remove one edge, try to add two others
    # This is expensive, so do a limited number of random attempts
    for _ in range(200):
        if time.time() > time_limit:
            break
        # Pick a random edge to remove
        if len(chosen) == 0:
            break
        e_rm = rng.choice(list(chosen))
        chosen.discard(e_rm)
        
        # Try to add back edges that were previously blocked
        added_any = False
        for e_idx in range(n):
            if e_idx not in chosen and _can_add(e_idx, chosen, c4_triples):
                chosen.add(e_idx)
                added_any = True
            if time.time() > time_limit:
                break
        
        # If we didn't improve, consider restoring
        # Actually, just keep the result; the next iterations will try to fix
        # But we should track best
        # For simplicity, accept if size didn't decrease
        # Actually, let's do proper hill climbing: keep best
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    edges = _build_edges()
    c4_triples = _build_c4_triples(edges)
    n = len(edges)
    
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        chosen = _run_greedy(edges, c4_triples, rng)
        size = len(chosen)
        
        if size > best_size:
            best_set = set(chosen)
            best_size = size
        
        # Local improvement
        if time.time() < deadline:
            improved = _local_improve(chosen, edges, c4_triples, rng, deadline)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
    
    # Convert edge indices back to [u, v] pairs
    result = []
    for e_idx in best_set:
        u, v = edges[e_idx]
        result.append([u, v])
    
    return result

def _build_c4_triples(edges):
    d = 7
    n = len(edges)
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
    
    c4_triples = [[] for _ in range(n)]
    
    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            v2 = v ^ bit_j
            u2 = u ^ bit_j
            e1 = (min(v, v2), max(v, v2))
            e2 = (min(v2, u2), max(v2, u2))
            e3 = (min(u, u2), max(u, u2))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                c4_triples[idx].append((idx1, idx2, idx3))
    
    return c4_triples
# EVOLVE-BLOCK-END
