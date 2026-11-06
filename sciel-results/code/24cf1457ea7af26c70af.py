# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 150s kill
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2  # 448

    # Build list of all edges as (u, v) with u < v, and track dimension
    all_edges = []
    edge_dim = {}
    for x in range(nverts):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
                edge_dim[(x, y)] = i

    # Precompute for each edge the list of "witness" triples
    # Edge (u,v) along dim i: for each j != i, C4 uses edges (u,u^j),(v,v^j),(u^j,v^j)
    edge_witnesses = {}
    for (u, v) in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        witnesses = []
        for j in range(d):
            if j != i:
                uj = u ^ (1 << j)
                vj = v ^ (1 << j)
                e1 = (min(u, uj), max(u, uj))
                e2 = (min(v, vj), max(v, vj))
                e3 = (min(uj, vj), max(uj, vj))
                witnesses.append((e1, e2, e3))
        edge_witnesses[(u, v)] = witnesses

    # Build dimension-stratified order: for each dimension i, list all edges in that dim
    dim_edges = [[] for _ in range(d)]
    for (u, v) in all_edges:
        dim_edges[edge_dim[(u, v)]].append((u, v))

    # Helper: greedy add with a given order of edges
    def greedy_order(order):
        chosen = set()
        for edge in order:
            blocked = False
            for (e1, e2, e3) in edge_witnesses[edge]:
                if e1 in chosen and e2 in chosen and e3 in chosen:
                    blocked = True
                    break
            if not blocked:
                chosen.add(edge)
        return chosen

    # Helper: local search - try to improve by removing one edge and adding others
    def local_search(chosen, max_iter=200):
        chosen = set(chosen)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            # Try each edge in chosen: remove it, then greedily add back as many as possible
            # Focus on edges that are "bottlenecks" - part of many potential C4s
            for edge in list(chosen):
                # Remove this edge
                chosen.discard(edge)
                # Now try to add back edges that were blocked
                added = 0
                for (e1, e2, e3) in edge_witnesses[edge]:
                    # If this edge was blocking, the witnesses might now be addable
                    pass
                # Greedily try to add all edges not in chosen
                for cand in all_edges:
                    if cand in chosen:
                        continue
                    blocked = False
                    for (a, b, c) in edge_witnesses[cand]:
                        if a in chosen and b in chosen and c in chosen:
                            blocked = True
                            break
                    if not blocked:
                        chosen.add(cand)
                        added += 1
                # If we added more than 1, we improved (removed 1, added >1)
                if added <= 1:
                    # Revert: put edge back
                    chosen.add(edge)
                else:
                    improved = True
                    break  # restart scan
        return chosen

    best_count = 0
    best_edges = []
    rng = np.random.RandomState(42)

    # Strategy 1: Dimension-stratified with random dimension order
    # This keeps edges of the same dimension together (locality) but randomizes dim order
    while time.time() < deadline:
        # Random permutation of dimensions
        dim_order = list(range(d))
        rng.shuffle(dim_order)
        
        order = []
        for dim in dim_order:
            dim_list = list(dim_edges[dim])
            rng.shuffle(dim_list)
            order.extend(dim_list)
        
        chosen = greedy_order(order)
        count = len(chosen)
        
        # Local search polish
        if time.time() < deadline - 5:
            chosen = local_search(chosen, max_iter=100)
            count = len(chosen)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in chosen]
        
        if best_count >= 308:
            break

    # Strategy 2: Reverse-stratified (try from end)
    if time.time() < deadline - 10:
        dim_order = list(range(d))
        rng.shuffle(dim_order)
        order = []
        for dim in dim_order:
            dim_list = list(dim_edges[dim])
            rng.shuffle(dim_list)
            order.extend(reversed(dim_list))
        
        chosen = greedy_order(order)
        count = len(chosen)
        
        if time.time() < deadline - 5:
            chosen = local_search(chosen, max_iter=100)
            count = len(chosen)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in chosen]

    # Strategy 3: Pure random permutation (as before, for diversity)
    while time.time() < deadline - 5:
        order = list(all_edges)
        rng.shuffle(order)
        
        chosen = greedy_order(order)
        count = len(chosen)
        
        if time.time() < deadline - 5:
            chosen = local_search(chosen, max_iter=50)
            count = len(chosen)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in chosen]
        
        if best_count >= 308:
            break

    return best_edges
# EVOLVE-BLOCK-END
