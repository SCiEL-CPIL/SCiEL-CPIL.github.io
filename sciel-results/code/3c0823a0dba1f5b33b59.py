import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            # Pivot selection: choose vertex u in P union X with max degree in P
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        # Precompute: for each vertex, which cliques does it hit?
        # This allows quick updates.
        n_cliques = len(maximal_cliques)
        
        # For each vertex, list of clique indices it hits
        vertex_cliques = [[] for _ in range(n)]
        for i, c in enumerate(maximal_cliques):
            for v in c:
                vertex_cliques[v].append(i)
        
        # Greedy initialization
        unhit_mask = (1 << n_cliques) - 1  # All cliques unhit
        transversal = set()
        
        while unhit_mask:
            if time.time() > deadline:
                break
            # Count frequency of each vertex in unhit cliques
            freq = [0] * n
            # Iterate through unhit cliques using bitmask
            m = unhit_mask
            while m:
                # Get lowest set bit
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                for v in maximal_cliques[idx]:
                    freq[v] += 1
            
            if not any(freq):
                break
                
            best_v = None
            best_score = None
            for v in range(n):
                f = freq[v]
                if f == 0:
                    continue
                # Score: frequency is primary. Degree is secondary.
                score = (f, len(adj[v]), -v)
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            
            transversal.add(best_v)
            
            # Remove all cliques containing best_v from unhit_mask
            for ci in vertex_cliques[best_v]:
                unhit_mask &= ~(1 << ci)

        # Ensure validity if broken by time
        if unhit_mask:
            m = unhit_mask
            while m:
                lowbit = m & (-m)
                idx = lowbit.bit_length() - 1
                m ^= lowbit
                transversal.add(next(iter(maximal_cliques[idx])))

        # Helper to check if a set is a valid transversal
        # Returns True if every maximal clique is hit by the set
        def is_valid_transversal(candidate_set):
            # For efficiency, we can check if any clique is disjoint from candidate_set
            for c in maximal_cliques:
                if not (c & candidate_set):
                    return False
            return True

        # Local Search Refinement
        # Strategy: 
        # 1. Greedy Removal: Remove any vertex if the set remains valid.
        # 2. k-for-1 Replacement: Try to replace a subset of k vertices with 1 new vertex w.
        #    This reduces size by k-1.
        
        improved = True
        # We limit the number of passes to avoid infinite loops or excessive time
        # But since size strictly decreases on success, it will terminate.
        max_passes = 50
        pass_count = 0
        
        while improved and time.time() < deadline and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # 1. Greedy Removal
            for v in list(transversal):
                if time.time() > deadline:
                    return sorted(transversal)
                # Check if removing v keeps it valid
                candidate = transversal - {v}
                if is_valid_transversal(candidate):
                    transversal = candidate
                    improved = True
            
            if not transversal:
                break

            # 2. k-for-1 Replacement
            # We try k=1 (redundant check again, though removal covers it), 
            # k=2 (replace 2 with 1), k=3 (replace 3 with 1)
            # For small transversals, we can check all subsets.
            # To be safe with time, we limit k to 3.
            
            trans_list = list(transversal)
            size = len(trans_list)
            
            # Check k=2 replacements
            for i in range(size):
                if time.time() > deadline:
                    return sorted(transversal)
                for j in range(i + 1, size):
                    v1 = trans_list[i]
                    v2 = trans_list[j]
                    
                    # Candidate set: transversal without v1, v2
                    base_candidate = transversal - {v1, v2}
                    
                    # Find cliques that are now unhit by base_candidate
                    # These are the "deficient" cliques that v1 or v2 were covering
                    # Actually, simpler: just check if adding any w to base_candidate makes it valid.
                    # But we want to find if there EXISTS a w such that (base_candidate | {w}) is valid.
                    # This is equivalent to: w must hit ALL cliques that are disjoint from base_candidate.
                    
                    # Let's find the set of cliques NOT hit by base_candidate.
                    # If this set is empty, we already handled it in removal (k=1 case essentially, 
                    # but here we removed 2, so if empty, we reduced size by 2, which is even better).
                    
                    unhit_by_base = []
                    for c in maximal_cliques:
                        if not (c & base_candidate):
                            unhit_by_base.append(c)
                    
                    if not unhit_by_base:
                        # We removed 2 vertices and the set is still valid!
                        transversal = base_candidate
                        improved = True
                        break
                    
                    # We need a single vertex w that hits ALL cliques in unhit_by_base.
                    # w must be in the intersection of all these cliques.
                    # Also w should not be v1 or v2 (if it were, we would have found it in removal? 
                    # No, removal checks if transversal-{v} is valid. 
                    # Here we check if transversal-{v1,v2}+{w} is valid.
                    # If w==v1, then transversal-{v2} is valid, which would have been caught by removal.
                    # So we can skip v1, v2 to save time, but strictly it's not necessary for correctness, 
                    # just for finding NEW solutions. However, if w==v1, size doesn't change.
                    # We want size reduction, so w must be distinct from v1, v2.
                    
                    potential_w = set(unhit_by_base[0])
                    for c in unhit_by_base[1:]:
                        potential_w &= c
                    
                    # Filter out v1, v2
                    potential_w.discard(v1)
                    potential_w.discard(v2)
                    
                    if potential_w:
                        # Any w in potential_w works. We pick one.
                        # Does it matter which one? Not for size, but maybe for future moves.
                        # Let's just pick the first one.
                        w = next(iter(potential_w))
                        transversal = base_candidate | {w}
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Check k=3 replacements (only if size >= 3)
            if len(transversal) >= 3:
                trans_list = list(transversal)
                size = len(trans_list)
                for i in range(size):
                    if time.time() > deadline:
                        return sorted(transversal)
                    for j in range(i + 1, size):
                        for k in range(j + 1, size):
                            v1, v2, v3 = trans_list[i], trans_list[j], trans_list[k]
                            base_candidate = transversal - {v1, v2, v3}
                            
                            unhit_by_base = []
                            for c in maximal_cliques:
                                if not (c & base_candidate):
                                    unhit_by_base.append(c)
                            
                            if not unhit_by_base:
                                transversal = base_candidate
                                improved = True
                                break
                            
                            potential_w = set(unhit_by_base[0])
                            for c in unhit_by_base[1:]:
                                potential_w &= c
                            
                            potential_w.discard(v1)
                            potential_w.discard(v2)
                            potential_w.discard(v3)
                            
                            if potential_w:
                                w = next(iter(potential_w))
                                transversal = base_candidate | {w}
                                improved = True
                                break
                            
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return sorted(transversal)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)
        if time.time() > deadline:
             # If we ran out of time, ensure we return what we have
             # The loop above checks deadline, but this is a safety net.
             pass

    return results
