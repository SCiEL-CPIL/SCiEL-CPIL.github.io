"""Dense odd-order graphs avoiding equal-degree path endpoints."""
import random
import time

# EVOLVE-BLOCK-START

N = 15
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_EDGES = len(ALL_EDGES)  # 105

def _edge_bad(u, v, degrees):
    """Check if edge (u,v) is bad: exists s,t not in {u,v}, s!=t, deg[s]==deg[t]."""
    seen = set()
    for w in range(N):
        if w != u and w != v:
            d = degrees[w]
            if d in seen:
                return True
            seen.add(d)
    return False

def _all_edges_valid(degrees):
    """Check all edges are valid given current degrees."""
    for u, v in ALL_EDGES:
        if _edge_bad(u, v, degrees):
            return False
    return True

def _count_bad_edges(degrees):
    """Count how many edges are bad given current degrees."""
    count = 0
    for u, v in ALL_EDGES:
        if _edge_bad(u, v, degrees):
            count += 1
    return count

def construct():
    deadline = time.time() + 85
    best_edge_set = set()
    best_count = 0
    rng = random.Random(42)
    
    # We represent state as a set of edges and degrees array.
    # An edge is "bad" if deg[s]==deg[t] for some s,t not incident to it.
    # The constraint: for EVERY edge (u,v) in the graph, there must NOT exist
    # s,t not in {u,v}, s!=t, with deg[s]==deg[t].
    
    # Key insight: The badness of an edge depends ONLY on the degree sequence,
    # not on which edges are present (as long as the edge is present).
    # So we need: for every edge (u,v) in our graph, the degree sequence (excluding u,v)
    # has all distinct values.
    
    # Strategy: Start with empty graph, greedily add edges.
    # When we add an edge, degrees change, which may make other edges bad.
    # We need to ensure after each addition, all present edges are valid.
    
    def try_add(edge_set, degrees, e):
        """Try adding edge e. Returns success bool."""
        u, v = e
        degrees[u] += 1
        degrees[v] += 1
        edge_set.add(e)
        
        # Check if all edges in edge_set are still valid
        for (a, b) in edge_set:
            if _edge_bad(a, b, degrees):
                # Roll back
                degrees[a] -= 1  # wait, this is wrong, we need to roll back u,v
                # Let me redo
                pass
        
        # Redo properly
        degrees[u] -= 1
        degrees[v] -= 1
        edge_set.discard(e)
        
        # Check: after adding e, are all edges valid?
        degrees[u] += 1
        degrees[v] += 1
        for (a, b) in edge_set:
            if (a, b) != (u, v) and _edge_bad(a, b, degrees):
                # Bad
                degrees[u] -= 1
                degrees[v] -= 1
                return False
        # All good
        edge_set.add(e)
        return True
    
    # Simpler: just check after tentative add
    def can_add(edge_set, degrees, e):
        u, v = e
        new_deg = degrees[:]
        new_deg[u] += 1
        new_deg[v] += 1
        for (a, b) in edge_set:
            if _edge_bad(a, b, new_deg):
                return False
        return True
    
    # Greedy with multiple orderings
    while time.time() < deadline:
        edge_set = set()
        degrees = [0] * N
        remaining = list(ALL_EDGES)
        rng.shuffle(remaining)
        
        for e in remaining:
            if can_add(edge_set, degrees, e):
                edge_set.add(e)
                degrees[e[0]] += 1
                degrees[e[1]] += 1
        
        # Local search: try to add more edges by removing one and adding two
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edge_set = set(edge_set)
        
        # Try local improvement
        time_left = deadline - time.time()
        if time_left < 2:
            break
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding a single missing edge
            missing = [e for e in ALL_EDGES if e not in edge_set]
            rng.shuffle(missing)
            for e in missing:
                if can_add(edge_set, degrees, e):
                    edge_set.add(e)
                    degrees[e[0]] += 1
                    degrees[e[1]] += 1
                    improved = True
                    if len(edge_set) > best_count:
                        best_count = len(edge_set)
                        best_edge_set = set(edge_set)
                    break
            if improved:
                continue
            # Try swap: remove one edge, add two
            edges_list = list(edge_set)
            rng.shuffle(edges_list)
            for e_rem in edges_list:
                # Temporarily remove
                old_deg = degrees[:]
                edge_set.discard(e_rem)
                degrees[e_rem[0]] -= 1
                degrees[e_rem[1]] -= 1
                
                # Try to add two edges
                added = []
                missing = [e for e in ALL_EDGES if e not in edge_set]
                rng.shuffle(missing)
                for e_add in missing:
                    if can_add(edge_set, degrees, e_add):
                        edge_set.add(e_add)
                        degrees[e_add[0]] += 1
                        degrees[e_add[1]] += 1
                        added.append(e_add)
                        break
                if len(added) == 1:
                    # Try to add one more
                    missing = [e for e in ALL_EDGES if e not in edge_set]
                    rng.shuffle(missing)
                    for e_add2 in missing:
                        if can_add(edge_set, degrees, e_add2):
                            edge_set.add(e_add2)
                            degrees[e_add2[0]] += 1
                            degrees[e_add2[1]] += 1
                            added.append(e_add2)
                            break
                if len(added) >= 2:
                    improved = True
                    if len(edge_set) > best_count:
                        best_count = len(edge_set)
                        best_edge_set = set(edge_set)
                else:
                    # Roll back
                    for e_a in added:
                        edge_set.discard(e_a)
                        degrees[e_a[0]] -= 1
                        degrees[e_a[1]] -= 1
                    edge_set.add(e_rem)
                    degrees = old_deg[:]
    
    result = [[int(u), int(v)] for u, v in best_edge_set]
    return result

# EVOLVE-BLOCK-END
