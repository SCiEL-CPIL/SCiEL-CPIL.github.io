import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, edges):
        # Build adjacency masks
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            
            U = P | X
            if U:
                # Pivot selection
                best_u = -1
                max_deg = -1
                temp = U
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    deg = (P & adj_mask[u]).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_u = u
                    temp ^= lsb
                
                candidates = P & ~adj_mask[best_u]
            else:
                candidates = P
                
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                bron_kerbosch(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb

        # Start with all vertices that have degree > 0? 
        # BK handles isolated vertices, but starting with active is faster.
        active = 0
        for i in range(n):
            if adj_mask[i]:
                active |= (1 << i)
                
        bron_kerbosch(0, active, 0)
        return cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return set()
        
        # Map vertex to list of clique indices
        v_to_c = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            # c is a bitmask
            temp = c
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                v_to_c[v].append(i)
                temp ^= lsb
                
        uncovered = set(range(len(cliques)))
        S = set()
        
        while uncovered:
            best_v = -1
            max_covered = -1
            for v in range(n):
                if v in S:
                    continue
                covered = sum(1 for i in v_to_c[v] if i in uncovered)
                if covered > max_covered:
                    max_covered = covered
                    best_v = v
            
            if best_v == -1:
                break
                
            S.add(best_v)
            for i in v_to_c[best_v]:
                if i in uncovered:
                    uncovered.remove(i)
                    
        return S

    def local_search(S, cliques, n, time_limit):
        start_time = time.time()
        
        if not cliques:
            return S

        # Precompute v_to_c
        v_to_c = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            temp = c
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                v_to_c[v].append(i)
                temp ^= lsb
                
        # Initialize counts
        counts = [0] * len(cliques)
        for v in S:
            for i in v_to_c[v]:
                counts[i] += 1
                
        S = set(S)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Phase 1: Removal
            for v in list(S):
                can_remove = True
                for i in v_to_c[v]:
                    if counts[i] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for i in v_to_c[v]:
                        counts[i] -= 1
                    S.remove(v)
                    improved = True
            
            if improved:
                continue
            
            # Phase 2: 1-for-1 Swap
            # Try to swap v in S with w not in S
            # This maintains size but might enable removals
            found_swap = False
            for v in list(S):
                # Heuristic: try w that shares many cliques with v?
                # Just iterate all w not in S
                for w in range(n):
                    if w in S:
                        continue
                    # Check validity
                    valid = True
                    # Affected cliques are those containing v or w
                    # We can iterate v_to_c[v] and v_to_c[w]
                    # To avoid double counting, use a set of indices
                    affected = set(v_to_c[v]).union(v_to_c[w])
                    for i in affected:
                        cv = 1 if i in v_to_c[v] else 0
                        cw = 1 if i in v_to_c[w] else 0
                        new_c = counts[i] - cv + cw
                        if new_c < 1:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        for i in v_to_c[v]:
                            counts[i] -= 1
                        for i in v_to_c[w]:
                            counts[i] += 1
                        S.remove(v)
                        S.add(w)
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    break
                    
            if improved:
                continue

            # Phase 3: 2-for-1 Swap
            # Try to replace v1, v2 in S with w not in S
            # This reduces size by 1
            found_swap = False
            for w in range(n):
                if w in S:
                    continue
                
                S_list = list(S)
                for idx1 in range(len(S_list)):
                    v1 = S_list[idx1]
                    for idx2 in range(idx1 + 1, len(S_list)):
                        v2 = S_list[idx2]
                        
                        # Check validity
                        valid = True
                        affected = set(v_to_c[v1]).union(v_to_c[v2]).union(v_to_c[w])
                        for i in affected:
                            cv1 = 1 if i in v_to_c[v1] else 0
                            cv2 = 1 if i in v_to_c[v2] else 0
                            cw = 1 if i in v_to_c[w] else 0
                            new_c = counts[i] - cv1 - cv2 + cw
                            if new_c < 1:
                                valid = False
                                break
                                
                        if valid:
                            # Perform swap
                            for i in v_to_c[v1]:
                                counts[i] -= 1
                            for i in v_to_c[v2]:
                                counts[i] -= 1
                            for i in v_to_c[w]:
                                counts[i] += 1
                            
                            S.remove(v1)
                            S.remove(v2)
                            S.add(w)
                            
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break
                    
            if not improved:
                break
                
        return S

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Not enough time, return empty or greedy quickly?
            # Better to return empty to avoid penalty? No, must be valid.
            # Just break and fill with empty? No, must hit cliques.
            # We assume we have time.
            pass
            
        edges = random_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
            
        initial_S = greedy_hitting_set(cliques, n)
        
        # Allocate remaining time / 3
        time_left = deadline - time.time()
        per_instance_limit = max(5.0, time_left / 3.0)
        
        final_S = local_search(initial_S, cliques, n, per_instance_limit)
        results.append(sorted(final_S))
        
    return results
