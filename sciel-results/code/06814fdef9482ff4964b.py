import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))
        
        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs
        
        # Group I vertices by their exact neighborhood in C
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)
        
        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))
        
        best_partition = None
        best_count = float('inf')
        
        # Strategy: 
        # 1. Greedily form large cliques covering I vertices and their shadows.
        # 2. For remaining C-C edges, use a "Maximal Clique" heuristic based on 
        #    common neighbors, trying to pack as many edges into each clique as possible.
        #    This is a structural change from simple greedy edge-picking to a 
        #    clique-expansion approach that respects the chordal nature better 
        #    by finding large independent sets in the complement (cliques in G).
        
        # We will run a few variations of the heuristic to find the best partition.
        
        for strategy_idx in range(3):
            if time.time() > deadline - 10.0:
                break
                
            partition = []
            used_c_edges = set()
            
            # Process I groups
            # Order can be varied, but for now we stick to descending shadow size 
            # as it tends to cover more edges per clique.
            
            group_order = sorted_groups
            if strategy_idx == 1:
                # Try ascending? Might be risky but worth a shot if time permits.
                # Actually, descending is usually better for "covering" because 
                # large shadows are harder to cover with small cliques.
                group_order = sorted(sorted_groups, key=lambda x: len(x[0]))
            elif strategy_idx == 2:
                # Randomized ordering based on a seed?
                # Let's just stick to deterministic for now to avoid noise, 
                # but maybe shuffle slightly? No, keep it simple.
                pass
            
            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue
                
                # Find a large clique within shadow using unused C-C edges
                available = list(shadow)
                
                if not available:
                    continue
                
                # Helper: count unused edges from u within a set
                def get_unused_degree(u, avail_set):
                    cnt = 0
                    for w in avail_set:
                        if u != w:
                            e = (min(u, w), max(u, w))
                            if e in internal_edges and e not in used_c_edges:
                                cnt += 1
                    return cnt
                
                # Sort available by unused degree descending
                avail_set = set(available)
                available.sort(key=lambda u: -get_unused_degree(u, avail_set))
                
                clique = []
                for u in available:
                    ok = True
                    for w in clique:
                        e = (min(u, w), max(u, w))
                        if e not in internal_edges or e in used_c_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(u)
                
                if len(clique) >= 1:
                    new_clique = sorted(i_vertices + clique)
                    partition.append(new_clique)
                    # Mark C-C edges as used
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            used_c_edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))
                
                # Cover remaining cross edges for this group's I vertices
                rem_i = list(i_vertices)
                rem_c = list(shadow - set(clique))
                
                while rem_i and rem_c:
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Find a clique in rem_c using unused edges
                    rem_c_set = set(rem_c)
                    avail_c = sorted(rem_c, key=lambda u: -get_unused_degree(u, rem_c_set))
                    
                    c_clique = []
                    for u in avail_c:
                        ok = True
                        for w in c_clique:
                            e = (min(u, w), max(u, w))
                            if e not in internal_edges or e in used_c_edges:
                                ok = False
                                break
                        if ok:
                            c_clique.append(u)
                    
                    if not c_clique:
                        c_clique = [rem_c[0]]
                    
                    new_clique = sorted(rem_i + c_clique)
                    partition.append(new_clique)
                    
                    # Mark C-C edges in c_clique as used
                    for i_idx in range(len(c_clique)):
                        for j_idx in range(i_idx + 1, len(c_clique)):
                            used_c_edges.add((min(c_clique[i_idx], c_clique[j_idx]), max(c_clique[i_idx], c_clique[j_idx])))
                    
                    # Remove used C's from rem_c
                    c_clique_set = set(c_clique)
                    rem_c = [c for c in rem_c if c not in c_clique_set]

            # Cover remaining C-C edges
            rem_set = internal_edges - used_c_edges
            
            # Precompute: for each vertex, which other C-vertices it has an unused edge to
            # We'll maintain this as we remove edges.
            rem_adj = [set() for _ in range(k)]
            for (a, b) in rem_set:
                rem_adj[a].add(b)
                rem_adj[b].add(a)
            
            cc_strategy = strategy_idx  # 0: max-common, 1: min-degree, 2: max-common
            
            while rem_set:
                if time.time() > deadline - 3.0:
                    break
                
                if cc_strategy == 0:
                    # Pick edge with max valid common neighbors
                    best_edge = None
                    max_common = -1
                    for (a, b) in rem_set:
                        common = rem_adj[a] & rem_adj[b]
                        common.discard(a)
                        common.discard(b)
                        if len(common) > max_common:
                            max_common = len(common)
                            best_edge = (a, b)
                else:
                    # Pick edge with min degree sum (to leave high-degree vertices for later)
                    best_edge = None
                    min_deg = float('inf')
                    for (a, b) in rem_set:
                        d = len(rem_adj[a]) + len(rem_adj[b])
                        if d < min_deg:
                            min_deg = d
                            best_edge = (a, b)
                
                if best_edge is None:
                    break
                
                a, b = best_edge
                clique = [a, b]
                
                # Expand: candidates are vertices connected to ALL clique members in rem graph
                candidates = rem_adj[a] & rem_adj[b]
                candidates.discard(a)
                candidates.discard(b)
                
                # Sort by remaining degree descending
                candidates = sorted(candidates, key=lambda u: -len(rem_adj[u]))
                
                for c in candidates:
                    # Verify c connects to all current clique members
                    ok = True
                    for u in clique:
                        if u not in rem_adj[c]:
                            ok = False
                            break
                    if ok:
                        clique.append(c)
                
                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    # Remove edges covered by this clique
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            u, v = clique[i], clique[j]
                            rem_set.discard((min(u,v), max(u,v)))
                            rem_adj[u].discard(v)
                            rem_adj[v].discard(u)
            
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        
        if best_partition is None:
            return []
        
        cleaned = []
        for clique in best_partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
