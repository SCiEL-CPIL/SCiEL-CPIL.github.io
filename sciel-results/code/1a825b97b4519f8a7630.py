import time
import sys
from random import Random

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    Strategy:
    1. Generate the split graph.
    2. Maximize size-3 cliques {v,u,w} via greedy + local search.
    3. Cover remaining edges with size-2 cliques and one large internal clique.
    4. Deadline guard at ~140s.
    """
    
    deadline = time.time() + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency: for each v in I, list of K-neighbors
        nbrs_of_I = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            nbrs_of_I[v] = nbrs
        
        # All internal edges in K
        all_internal = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_internal.add((u, w))
        
        I_vertices = list(range(k, n))
        num_I = n - k
        num_K = k
        
        best_M = 0
        best_assign = None
        
        search_rng = Random(seed + 7777)
        
        def greedy_packing(order, local_rng):
            """Greedy: for each v in order, greedily pick disjoint K-pairs from N(v)."""
            used_internal = set()
            assign = {}
            M = 0
            
            for v in order:
                nbrs = nbrs_of_I[v]
                if len(nbrs) < 2:
                    continue
                
                local_nbrs = nbrs[:]
                local_rng.shuffle(local_nbrs)
                
                used_K_verts = set()
                pairs = []
                
                i = 0
                while i < len(local_nbrs):
                    u = local_nbrs[i]
                    if u in used_K_verts:
                        i += 1
                        continue
                    found = False
                    j = i + 1
                    while j < len(local_nbrs):
                        w = local_nbrs[j]
                        if w in used_K_verts:
                            j += 1
                            continue
                        ek = (u, w) if u < w else (w, u)
                        if ek not in used_internal:
                            pairs.append((u, w))
                            used_internal.add(ek)
                            used_K_verts.add(u)
                            used_K_verts.add(w)
                            M += 1
                            found = True
                            break
                        j += 1
                    i += 1
                
                if pairs:
                    assign[v] = pairs
            
            return M, assign
        
        def local_search(assign, M, local_rng):
            """Try to improve by re-assigning a random v's pairs."""
            if not assign:
                return M, assign
            
            # Build used_internal from current assign
            used_internal = set()
            for v, pairs in assign.items():
                for u, w in pairs:
                    ek = (u, w) if u < w else (w, u)
                    used_internal.add(ek)
            
            # Try a few random re-assignments
            for _ in range(50):
                v = local_rng.choice(list(assign.keys()))
                old_pairs = assign[v]
                old_M = len(old_pairs)
                
                # Remove old pairs from used_internal
                for u, w in old_pairs:
                    ek = (u, w) if u < w else (w, u)
                    used_internal.discard(ek)
                
                # Re-greedy for v
                nbrs = nbrs_of_I[v]
                local_nbrs = nbrs[:]
                local_rng.shuffle(local_nbrs)
                used_K = set()
                new_pairs = []
                
                i = 0
                while i < len(local_nbrs):
                    u = local_nbrs[i]
                    if u in used_K:
                        i += 1
                        continue
                    j = i + 1
                    while j < len(local_nbrs):
                        w = local_nbrs[j]
                        if w in used_K:
                            j += 1
                            continue
                        ek = (u, w) if u < w else (w, u)
                        if ek not in used_internal:
                            new_pairs.append((u, w))
                            used_internal.add(ek)
                            used_K.add(u)
                            used_K.add(w)
                            break
                        j += 1
                    i += 1
                
                new_M = len(new_pairs)
                
                if new_M > old_M:
                    if new_pairs:
                        assign[v] = new_pairs
                    else:
                        del assign[v]
                    M += (new_M - old_M)
                else:
                    # Revert
                    for u, w in new_pairs:
                        ek = (u, w) if u < w else (w, u)
                        used_internal.discard(ek)
                    for u, w in old_pairs:
                        ek = (u, w) if u < w else (w, u)
                        used_internal.add(ek)
                    assign[v] = old_pairs
            
            return M, assign
        
        # Multiple restarts with different strategies
        num_restarts = 60
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            if restart % 4 == 0:
                order = I_vertices[:]
                search_rng.shuffle(order)
            elif restart % 4 == 1:
                order = sorted(I_vertices, key=lambda v: -len(nbrs_of_I[v]))
            elif restart % 4 == 2:
                order = sorted(I_vertices, key=lambda v: len(nbrs_of_I[v]))
            else:
                # Weighted random: higher degree more likely to go first
                order = I_vertices[:]
                for i in range(len(order) - 1, 0, -1):
                    j = search_rng.randint(0, i)
                    order[i], order[j] = order[j], order[i]
            
            M, assign = greedy_packing(order, search_rng)
            
            # Local search improvement
            if M > 0:
                M, assign = local_search(assign, M, search_rng)
            
            if M > best_M:
                best_M = M
                best_assign = {v: pairs[:] for v, pairs in assign.items()}
        
        # Construct cliques from best assignment
        cliques = []
        covered_cross = set()
        
        if best_assign:
            for v, pairs in best_assign.items():
                for u, w in pairs:
                    cliques.append([v, u, w])
                    covered_cross.add((v, u))
                    covered_cross.add((v, w))
        
        # Remaining cross edges as size-2 cliques
        for v in range(k, n):
            for u in nbrs_of_I[v]:
                if (v, u) not in covered_cross:
                    cliques.append([v, u])
        
        # Remaining internal edges: one large clique on all involved K vertices
        used_int = set()
        if best_assign:
            for v, pairs in best_assign.items():
                for u, w in pairs:
                    ek = (u, w) if u < w else (w, u)
                    used_int.add(ek)
        
        remaining_verts = set()
        for e in all_internal:
            if e not in used_int:
                remaining_verts.add(e[0])
                remaining_verts.add(e[1])
        
        if len(remaining_verts) >= 2:
            cliques.append(list(remaining_verts))
        
        # Ensure all are Python int lists
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        return final_cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 2:
            # Emergency: return empty (invalid but avoids timeout)
            result.append([])
            continue
        result.append(instance(*spec))
    
    return result
