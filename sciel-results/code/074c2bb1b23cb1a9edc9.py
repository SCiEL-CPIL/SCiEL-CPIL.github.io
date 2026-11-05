import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)

    # Build conflict graph: two edges conflict if they form K_{2,3}
    # Edges e1=(u1,v1) and e2=(u2,v2) conflict if:
    # - They share exactly one vertex, say u, and there exist 3 vertices w such that
    #   w is connected to both the other endpoints in some K_{2,3} configuration
    # More precisely: e1 and e2 are in the same K_{2,3} if there exist 3 vertices
    # that are common neighbors of the two "top" vertices, or similar configurations.
    
    # Actually, let me think about this differently. A K_{2,3} has 2 vertices on one side
    # (call them A, B) and 3 on the other (C, D, E). The edges are: A-C, A-D, A-E, B-C, B-D, B-E.
    # So two edges conflict if they could be part of the same K_{2,3}.
    
    # Two edges e1=(a,b) and e2=(c,d) conflict if:
    # Case 1: They share a vertex. Say e1=(x,y), e2=(x,z). Then they conflict if there exist
    #         2 more vertices w1, w2 such that (y,w1), (y,w2), (z,w1), (z,w2) are all edges.
    #         This means y and z have at least 2 common neighbors.
    # Case 2: They are disjoint. e1=(a,b), e2=(c,d) with all distinct. They conflict if there exist
    #         2 vertices w1, w2 such that a-w1, b-w1, c-w2, d-w2 and a-w2, b-w2, c-w1, d-w1... 
    #         Actually this is getting complex. Let me use a simpler approach.
    
    # Simpler: two edges conflict if adding both to any set would create a K_{2,3}.
    # But that depends on the context. Let me use the standard definition:
    # Two edges are in conflict if there exists a K_{2,3} that contains both edges.
    
    # For edges (u,v) and (w,x):
    # They are in a K_{2,3} if:
    # 1. u=w: then v and x need to have >= 2 common neighbors
    # 2. v=x: then u and w need to have >= 2 common neighbors  
    # 3. u=x: then v and w need to have >= 2 common neighbors
    # 4. v=w: then u and x need to have >= 2 common neighbors
    # 5. All four distinct: then we need a K_{2,3} containing both. This happens if
    #    there exist vertices a,b such that {a,b} are the "top" and {u,v,w,x} contain
    #    3 of the "bottom" vertices. This is complex.
    
    # Let me use a different approach: precompute for each pair of vertices how many
    # common neighbors they have. Then two edges conflict if they share a vertex and
    # the other endpoints have >= 2 common neighbors.
    
    # Actually, the cleanest approach: build the conflict graph where edge i conflicts
    # with edge j if there exists a K_{2,3} containing both edges i and j.
    
    # For efficiency, let me use bitsets.
    
    # First, let's precompute adjacency
    adj = [0] * n
    
    # For the conflict graph, I'll use a different approach:
    # Two edges (u,v) and (w,x) are in conflict if they can be part of the same K_{2,3}.
    # 
    # A K_{2,3} is determined by choosing 2 vertices for the "left" side and 3 for the "right" side,
    # with all 6 cross-edges present.
    #
    # Two edges are in the same K_{2,3} if:
    # - They share a vertex: say edges (a,b) and (a,c). Then b and c must have at least 2 common neighbors.
    # - They don't share a vertex: edges (a,b) and (c,d). Then there must exist a K_{2,3} containing both.
    #   This happens if there are vertices p,q such that {p,q} are on one side and the 3 on the other
    #   include 2 from {a,b} and 1 from {c,d} or vice versa, with all cross edges present.
    
    # This is getting complex. Let me use a simpler heuristic-based approach instead.
    
    # Actually, let me reconsider. The constraint is: no two vertices have 3 common neighbors.
    # So the graph is K_{2,3}-free.
    #
    # Two edges e1=(u1,v1) and e2=(u2,v2) are "compatible" if there exists a K_{2,3}-free graph
    # containing both edges. But that's not quite right either.
    
    # Let me think about it differently. The constraint is on pairs of vertices: for any pair (a,b),
    # |N(a) ∩ N(b)| <= 2.
    #
    # When we add an edge (u,v), it affects the common neighbor counts for all pairs involving u or v.
    # Specifically, for any w that is a neighbor of both u and v, adding (u,v) increases the common
    # neighbor count for pairs (u,w) and (v,w) by 1.
    
    # So the conflict between two edges is context-dependent. This makes the independent set formulation
    # tricky.
    
    # Let me use a different approach: simulated annealing with a better objective function.
    
    def get_edges_from_mask(mask):
        edges = []
        for i in range(num_edges_total):
            if (mask >> i) & 1:
                edges.append(list(all_edges[i]))
        return edges

    def init_adj(mask):
        adj = [0] * n
        for i in range(num_edges_total):
            if (mask >> i) & 1:
                u, v = all_edges[i]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def count_violations(adj):
        v = 0
        for u in range(n):
            for w in range(u + 1, n):
                if (adj[u] & adj[w]).bit_count() >= 3:
                    v += 1
        return v

    def can_add(u, v, adj):
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> v) & 1:
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            if (adj[w] >> u) & 1:
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
        return True

    def greedy_fill(adj):
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True

    def adj_to_mask(adj):
        mask = 0
        for i in range(num_edges_total):
            u, v = all_edges[i]
            if (adj[u] >> v) & 1:
                mask |= (1 << i)
        return mask

    best_mask = 0
    best_edge_count = 0

    # Use multiple random restarts with simulated annealing
    num_runs = 30
    for run in range(num_runs):
        if time.time() > deadline - 5.0:
            break

        # Random initialization with varying density
        p = random.uniform(0.20, 0.45)
        current_mask = 0
        for i in range(num_edges_total):
            if random.random() < p:
                current_mask |= (1 << i)

        adj = init_adj(current_mask)
        violations = count_violations(adj)
        edge_count = bin(current_mask).count('1')

        if violations == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_mask = current_mask

        # Simulated annealing with better parameters
        T = 3.0
        cooling = 0.998
        min_T = 0.01
        steps = 0
        max_steps = 50000

        while T > min_T and steps < max_steps:
            if time.time() > deadline - 5.0:
                break

            idx = random.randint(0, num_edges_total - 1)
            u, v = all_edges[idx]
            is_present = (current_mask >> idx) & 1

            if is_present:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_violations = count_violations(adj)
                new_edge_count = edge_count - 1
                old_energy = violations * 1000 - edge_count
                new_energy = new_violations * 1000 - new_edge_count
                delta_energy = new_energy - old_energy
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_mask ^= (1 << idx)
                    violations = new_violations
                    edge_count = new_edge_count
                    if violations == 0 and edge_count > best_edge_count:
                        best_edge_count = edge_count
                        best_mask = current_mask
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations
                    new_edge_count = edge_count + 1
                    old_energy = violations * 1000 - edge_count
                    new_energy = new_violations * 1000 - new_edge_count
                    delta_energy = new_energy - old_energy
                    if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                        current_mask |= (1 << idx)
                        violations = new_violations
                        edge_count = new_edge_count
                        if violations == 0 and edge_count > best_edge_count:
                            best_edge_count = edge_count
                            best_mask = current_mask
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations + 1
                    new_edge_count = edge_count + 1
                    old_energy = violations * 1000 - edge_count
                    new_energy = new_violations * 1000 - new_edge_count
                    delta_energy = new_energy - old_energy
                    if random.random() < math.exp(-delta_energy / T):
                        current_mask |= (1 << idx)
                        violations = new_violations
                        edge_count = new_edge_count
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

            T *= cooling
            steps += 1

        # Try to fix violations and then greedy fill
        if violations > 0:
            # Remove edges to fix violations
            for _ in range(500):
                if time.time() > deadline - 3.0:
                    break
                found = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if (adj[a] & adj[b]).bit_count() >= 3:
                            # Find an edge to remove that involves a or b
                            candidates = []
                            for i in range(num_edges_total):
                                eu, ev = all_edges[i]
                                if (current_mask >> i) & 1:
                                    if eu == a or eu == b or ev == a or ev == b:
                                        candidates.append(i)
                            if candidates:
                                # Prefer removing edges that help most
                                best_rem = None
                                best_score = float('inf')
                                for rem_idx in candidates:
                                    ru, rv = all_edges[rem_idx]
                                    # Try removing this edge
                                    adj[ru] &= ~(1 << rv)
                                    adj[rv] &= ~(1 << ru)
                                    test_violations = count_violations(adj)
                                    adj[ru] |= (1 << rv)
                                    adj[rv] |= (1 << ru)
                                    if test_violations < best_score:
                                        best_score = test_violations
                                        best_rem = rem_idx
                                if best_rem is not None:
                                    ru, rv = all_edges[best_rem]
                                    adj[ru] &= ~(1 << rv)
                                    adj[rv] &= ~(1 << ru)
                                    current_mask ^= (1 << best_rem)
                                    violations = count_violations(adj)
                                    edge_count = bin(current_mask).count('1')
                                    found = True
                                    break
                    if found:
                        break
                if violations == 0:
                    break

        # Greedy fill if valid
        if violations == 0:
            greedy_fill(adj)
            fill_mask = adj_to_mask(adj)
            fill_count = bin(fill_mask).count('1')
            if fill_count > best_edge_count:
                best_edge_count = fill_count
                best_mask = fill_mask

    # Final greedy fill on best solution
    if best_mask != 0:
        adj = init_adj(best_mask)
        greedy_fill(adj)
        best_mask = adj_to_mask(adj)

    return get_edges_from_mask(best_mask)

def _construct_greedy():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.seed(42)
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    adj = [0] * n
    edges = []
    for u, v in shuffled:
        if (adj[u] & adj[v]).bit_count() < 3:
            ok = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[u] & adj[w]).bit_count() + (1 if (adj[w] & (1 << v)) else 0) >= 3:
                    ok = False
                    break
                if ok:
                    if (adj[v] & adj[w]).bit_count() + (1 if (adj[w] & (1 << u)) else 0) >= 3:
                        ok = False
                        break
            if ok:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    return edges

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass
