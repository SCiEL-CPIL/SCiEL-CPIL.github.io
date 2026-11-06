"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 145  # 145s to leave safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed, time_budget):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency as bitmasks for fast intersection
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        remaining_mask = [0] * n  # bitmask of remaining edges per vertex
        for u, v in edges:
            remaining_mask[u] |= (1 << v)
            remaining_mask[v] |= (1 << u)
        
        # Count remaining edges
        remaining_count = len(edges)
        
        cliques = []
        
        def popcount(x):
            """Fast popcount using int.bit_count()."""
            return x.bit_count()
        
        def greedy_clique_from_start(start):
            """Greedy build a clique starting from 'start' using only remaining edges.
            Uses bitmask operations for speed."""
            clique = [start]
            # Candidates: neighbors of start via remaining edges
            candidates_mask = remaining_mask[start]
            
            while candidates_mask:
                # Pick candidate with max connections to other candidates
                best_cand = None
                best_score = -1
                cm = candidates_mask
                while cm:
                    lsb = cm & (-cm)
                    cand = lsb.bit_length() - 1
                    # Score: number of common remaining neighbors among candidates
                    score = popcount(remaining_mask[cand] & candidates_mask)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                    cm ^= lsb
                
                if best_cand is None:
                    break
                clique.append(best_cand)
                # Update candidates: must connect to best_cand via remaining edges
                candidates_mask &= remaining_mask[best_cand]
            
            return clique if len(clique) >= 2 else None
        
        def local_improve(clique, max_iter=10):
            """Try to extend the clique by checking if any remaining neighbor can be added."""
            clique_mask = 0
            for v in clique:
                clique_mask |= (1 << v)
            
            for _ in range(max_iter):
                # Find a vertex outside clique that connects to all clique members via remaining edges
                common_mask = remaining_mask[clique[0]]
                for v in clique[1:]:
                    common_mask &= remaining_mask[v]
                # Remove clique members themselves
                common_mask &= ~clique_mask
                
                if not common_mask:
                    break
                
                # Pick the one with highest remaining degree among candidates
                best_v = None
                best_deg = -1
                cm = common_mask
                while cm:
                    lsb = cm & (-cm)
                    v = lsb.bit_length() - 1
                    deg = popcount(remaining_mask[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cm ^= lsb
                
                if best_v is None:
                    break
                clique.append(best_v)
                clique_mask |= (1 << best_v)
            
            return clique
        
        def find_best_clique():
            """Try multiple starting vertices, return the largest clique found."""
            best_clique = None
            best_size = 1
            
            # Try vertices in order of remaining degree
            degrees = []
            for v in range(n):
                deg = popcount(remaining_mask[v])
                if deg >= 2:
                    degrees.append((deg, v))
            degrees.sort(reverse=True)
            
            # Try top vertices with more trials for better cliques
            max_trials = min(40, len(degrees))
            for i in range(max_trials):
                deg, v = degrees[i]
                clique = greedy_clique_from_start(v)
                if clique:
                    # Try to improve locally
                    clique = local_improve(clique, 5)
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique
                        if best_size >= 12:  # Good enough for dense graphs
                            break
            
            return best_clique
        
        def try_merge_small_cliques():
            """Try to merge small cliques (size 2-3) into larger ones if possible."""
            nonlocal cliques, remaining_mask, remaining_count
            
            # Find small cliques
            small_cliques = [i for i, c in enumerate(cliques) if len(c) <= 3]
            if len(small_cliques) < 2:
                return
            
            # Try pairs of small cliques
            for i_idx in range(len(small_cliques)):
                if time.time() > deadline - 15:
                    return
                ci = small_cliques[i_idx]
                clique_a = cliques[ci]
                if not clique_a:
                    continue
                
                clique_a_set = set(clique_a)
                clique_a_mask = 0
                for v in clique_a:
                    clique_a_mask |= (1 << v)
                
                for j_idx in range(i_idx + 1, len(small_cliques)):
                    cj = small_cliques[j_idx]
                    if cj <= ci:  # Already merged
                        continue
                    clique_b = cliques[cj]
                    if not clique_b:
                        continue
                    
                    # Check if all vertices in clique_b connect to all vertices in clique_a
                    can_merge = True
                    for vb in clique_b:
                        if vb in clique_a_set:
                            continue
                        # vb must connect to all in clique_a via remaining edges
                        for va in clique_a:
                            if not (remaining_mask[vb] & (1 << va)):
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    
                    if can_merge:
                        # Merge: create new clique
                        new_clique = clique_a + [v for v in clique_b if v not in clique_a_set]
                        if len(new_clique) > len(clique_a):
                            # Remove edges of merged clique from remaining
                            removed = 0
                            for i2, u in enumerate(new_clique):
                                for v2 in new_clique[i2+1:]:
                                    if remaining_mask[u] & (1 << v2):
                                        remaining_mask[u] &= ~(1 << v2)
                                        remaining_mask[v2] &= ~(1 << u)
                                        removed += 1
                            
                            # Replace cliques
                            cliques[ci] = new_clique
                            cliques[cj] = []
                            remaining_count -= removed
        
        while remaining_count > 0:
            if time.time() > deadline - 10:
                break
            
            # Find a good clique
            clique = find_best_clique()
            
            if clique is None or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = remaining_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            remaining_mask[u] &= ~(1 << v)
                            remaining_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue
            
            # Remove all edges within clique from remaining
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i+1:]:
                    if remaining_mask[u] & (1 << v):
                        remaining_mask[u] &= ~(1 << v)
                        remaining_mask[v] &= ~(1 << u)
                        removed += 1
            
            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = remaining_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            remaining_mask[u] &= ~(1 << v)
                            remaining_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue
            
            cliques.append(clique)
            remaining_count -= removed
        
        # Try to merge small cliques to reduce total count
        try_merge_small_cliques()
        
        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = remaining_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    remaining_mask[u] &= ~(1 << v)
                    remaining_mask[v] &= ~(1 << u)
                mask ^= lsb
        
        # Remove empty cliques
        cliques = [c for c in cliques if len(c) >= 2]
        
        return cliques
    
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed, deadline - time.time() - 5))
    
    return result
# EVOLVE-BLOCK-END
