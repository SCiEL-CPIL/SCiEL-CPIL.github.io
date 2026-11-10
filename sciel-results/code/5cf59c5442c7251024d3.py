"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from random import Random

def _generate_edges(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    return adj

def _bron_kerbosch_max_cliques(adj, n, time_limit, max_cliques=200):
    """
    Find large cliques using a modified Bron-Kerbosch with pivot.
    Returns a list of cliques (each a frozenset of vertices).
    We focus on finding cliques of size >= 3 to reduce edge count per clique.
    """
    start_time = time.time()
    adj_bool = adj  # bool matrix
    n = adj_bool.shape[0]
    
    # Pre-compute neighbor sets as bitmasks for speed
    # Since n <= 64, we can use Python integers as bitmasks
    neighbors = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj_bool[u, v]:
                mask |= (1 << v)
        neighbors[u] = mask
    
    found_cliques = []
    
    def bk(R, P, X):
        nonlocal found_cliques
        if time.time() - start_time > time_limit:
            return
        if P == 0 and X == 0:
            if len(R) >= 3:  # Only keep cliques of size at least 3
                found_cliques.append(frozenset(R))
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        # We want to minimize the number of recursive calls
        # Standard: pick u in P ∪ X that maximizes |P ∩ N(u)|
        # Actually, to minimize recursion, we pick u such that P \ N(u) is minimized
        # Let's pick u from P ∪ X with max |P & neighbors[u]|
        
        U = P | X
        if U == 0:
            return
            
        # Find pivot u
        max_inter = -1
        pivot = -1
        # Iterate through bits in U
        temp = U
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            inter = P & neighbors[u]
            cnt = bin(inter).count('1')
            if cnt > max_inter:
                max_inter = cnt
                pivot = u
            temp ^= lsb
        
        if pivot == -1:
            return
            
        # Candidates: P \ N(pivot)
        candidates = P & ~neighbors[pivot]
        
        temp = candidates
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            new_R = R | (1 << v)
            new_P = P & neighbors[v]
            new_X = X & neighbors[v]
            
            bk(new_R, new_P, new_X)
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            if len(found_cliques) > max_cliques:
                return

    # Initial call
    # R = 0, P = all vertices, X = 0
    bk(0, (1 << n) - 1, 0)
    
    return found_cliques

def _partition_edges(n, edges, adj, time_budget):
    """
    Partition edges into cliques.
    Strategy:
    1. Find a set of large cliques (size >= 3) greedily.
    2. Assign edges to the first clique that contains them.
    3. Remaining edges (not covered by any clique) become individual cliques (size 2).
    """
    start_time = time.time()
    
    # Find candidate cliques
    # We run Bron-Kerbosch to find maximal cliques of size >= 3
    # But standard BK finds all maximal cliques. We might have too many.
    # Let's limit the search or just take the first few large ones.
    
    # Actually, for dense graphs, finding ALL maximal cliques can be expensive.
    # Let's use a greedy approach instead:
    # 1. Sort edges by some heuristic? Or just iterate?
    # 2. For each edge, try to extend it to a maximal clique?
    
    # Let's try a hybrid:
    # Use BK to find some large cliques, but stop early if too many found or time runs out.
    
    cliques = _bron_kerbosch_max_cliques(adj, n, time_budget * 0.5, max_cliques=100)
    
    # Sort cliques by size descending
    cliques.sort(key=lambda c: len(c), reverse=True)
    
    # Map edges to cliques
    # We need to assign each edge to exactly one clique.
    # If an edge is in multiple cliques, we assign it to the largest one (or the first one found).
    # Then we remove that edge from consideration for other cliques.
    # If a clique loses edges, it might no longer be a clique?
    # NO. A clique is defined by vertices. If we select a set of vertices that form a clique,
    # ALL edges between them must be assigned to this clique.
    # So we cannot just pick a subset of edges.
    
    # Therefore, the strategy is:
    # 1. Identify a set of vertex sets that are cliques.
    # 2. These vertex sets must be edge-disjoint.
    # 3. Assign all edges within each vertex set to that clique.
    # 4. Any edge not covered by any of these cliques becomes a singleton edge-clique.
    
    # So we need to select a collection of edge-disjoint cliques (size >= 3) to maximize coverage.
    # This is a packing problem.
    
    # Greedy Packing:
    # 1. Sort candidate cliques by size (descending).
    # 2. Maintain a set of "used" edges.
    # 3. For each candidate clique, check if all its edges are unused.
    # 4. If yes, accept the clique and mark its edges as used.
    
    used_edges = set()
    selected_cliques = []
    
    # Convert cliques (frozensets of vertices) to list of edges
    clique_edges_list = []
    for c in cliques:
        vlist = list(c)
        c_edges = []
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                u, v = vlist[i], vlist[j]
                if u > v:
                    u, v = v, u
                c_edges.append((u, v))
        clique_edges_list.append((c, c_edges))
    
    # Sort by size descending
    clique_edges_list.sort(key=lambda x: len(x[0]), reverse=True)
    
    for c, c_edges in clique_edges_list:
        # Check if any edge in c_edges is already used
        conflict = False
        for e in c_edges:
            if e in used_edges:
                conflict = True
                break
        if not conflict:
            # Accept this clique
            selected_cliques.append(list(c))
            for e in c_edges:
                used_edges.add(e)
        
        if time.time() - start_time > time_budget:
            break
    
    # Now, construct the final partition
    # 1. Add selected cliques
    # 2. Add all remaining edges as individual cliques
    
    final_partition = []
    for c in selected_cliques:
        final_partition.append(c)
    
    # Find remaining edges
    remaining_edges = [e for e in edges if e not in used_edges]
    for e in remaining_edges:
        final_partition.append(list(e))
    
    return final_partition

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _generate_edges(n, p, seed)
        adj = _build_adj(n, edges)
        
        # Allocate time budget based on remaining time
        # Assume 50s per instance
        time_budget = min(50, deadline - time.time())
        if time_budget < 5:
            # Not enough time, just return edges as cliques
            results.append([list(e) for e in edges])
            continue
        
        partition = _partition_edges(n, edges, adj, time_budget)
        results.append(partition)
        
        if time.time() > deadline:
            # If we're running out of time, fill remaining with trivial partition
            for _ in range(2 - len(results)):
                results.append([])
            break
    
    return results
# EVOLVE-BLOCK-END
