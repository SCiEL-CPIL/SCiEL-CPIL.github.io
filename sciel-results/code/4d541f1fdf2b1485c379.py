"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

def construct():
    _deadline = time.time() + 135.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        rng = Random(seed)
        
        # Build cross edges using the same RNG sequence as instance()
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        
        # Build adjacency: for each I-vertex, which K-vertices it connects to
        # K-vertices: 0..k-1, I-vertices: k..n-1
        i_count = n - k
        # adj_i[i] = bitmask of K-neighbors of I-vertex (k+i)
        adj_i = [0] * i_count
        for u, v in cross_edges:
            i_idx = v - k
            adj_i[i_idx] |= (1 << u)
        
        # Track which K-K edges are "used" (covered by a clique containing an I-vertex)
        # K-K edges: (u, w) for 0 <= u < w < k
        # Represent as bitmask over k*(k-1)/2 edges, or just a set
        used_kk = set()
        
        cliques = []
        
        # Process I-vertices: for each, greedily extract largest compatible K-neighbor subset
        # "Compatible" means no K-K edge in the subset is already used
        # We want to maximize clique size to minimize number of cliques
        
        # Sort I-vertices by number of K-neighbors (descending) for better packing
        i_order = sorted(range(i_count), key=lambda i: bin(adj_i[i]).count('1'), reverse=True)
        
        for i in i_order:
            v = k + i
            mask = adj_i[i]
            if mask == 0:
                continue
            
            # Greedily build a clique S from neighbors of v
            # S must be a clique in the "available" K-graph (no used K-K edges within S)
            # Since K is a clique, all K-K edges exist; we just need none to be "used"
            
            # Try to build S by adding vertices one at a time
            # Heuristic: try vertices in order of how many "free" K-K edges they have with current S
            S = []
            remaining = mask
            
            while remaining:
                # Find best candidate: vertex u in remaining such that
                # (u, w) not in used_kk for all w in S
                best_u = None
                best_score = -1
                rem_bits = remaining
                while rem_bits:
                    # Extract lowest set bit
                    low = rem_bits & (-rem_bits)
                    u = low.bit_length() - 1
                    rem_bits ^= low
                    
                    # Check compatibility with all in S
                    compatible = True
                    for w in S:
                        edge = (min(u, w), max(u, w))
                        if edge in used_kk:
                            compatible = False
                            break
                    if compatible:
                        # Score: prefer vertices that free up more future options?
                        # Or just take first compatible (deterministic)
                        # Better: prefer vertex with most "free" connections to remaining
                        score = 0
                        for w in S:
                            edge = (min(u, w), max(u, w))
                            if edge not in used_kk:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_u = u
                
                if best_u is None:
                    # No compatible vertex found; break
                    break
                
                S.append(best_u)
                remaining &= ~(1 << best_u)
                
                # Check deadline
                if time.time() > _deadline - 3:
                    break
            
            if S:
                clique = [v] + S
                cliques.append(clique)
                # Mark all K-K edges within S as used
                for a, b in combinations(S, 2):
                    used_kk.add((min(a, b), max(a, b)))
            
            # If we broke early due to deadline, handle remaining edges as pairs
            if time.time() > _deadline - 3:
                # Fallback: cover remaining cross edges as pairs
                rem_mask = remaining
                while rem_mask:
                    low = rem_mask & (-rem_mask)
                    u = low.bit_length() - 1
                    rem_mask ^= low
                    cliques.append([v, u])
                break
        
        # Now cover remaining K-K edges
        # Build list of unused K-K edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_kk:
                    rem_edges.append((u, w))
        
        if rem_edges:
            # Greedy: repeatedly find largest clique in the remaining edge graph
            # Build adjacency for remaining edges
            adj_rem = [set() for _ in range(k)]
            for u, w in rem_edges:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
            
            remaining_edge_set = set(rem_edges)
            
            while remaining_edge_set and time.time() < _deadline - 2:
                # Find vertices involved in remaining edges
                verts = set()
                for u, w in remaining_edge_set:
                    verts.add(u)
                    verts.add(w)
                
                if not verts:
                    break
                
                # Greedy max clique in the induced subgraph on verts
                # Use a simple heuristic: repeatedly pick vertex with max degree in remaining graph
                C = set()
                candidates = set(verts)
                
                while candidates:
                    # Pick vertex with max degree in remaining graph restricted to candidates
                    best_v = None
                    max_deg = -1
                    for v in candidates:
                        deg = len(adj_rem[v] & candidates)
                        if deg > max_deg:
                            max_deg = deg
                            best_v = v
                    
                    if best_v is None or max_deg < 0:
                        break
                    
                    # Check if best_v is connected to all in C
                    can_add = True
                    for u in C:
                        if u not in adj_rem[best_v]:
                            can_add = False
                            break
                    
                    if can_add:
                        C.add(best_v)
                        candidates -= C
                        # Restrict candidates to neighbors of best_v
                        candidates &= adj_rem[best_v]
                    else:
                        candidates.discard(best_v)
                
                if not C:
                    # Should not happen; fallback to single edge
                    u, w = next(iter(remaining_edge_set))
                    C = {u, w}
                
                # Convert to clique
                clique_list = sorted(C)
                if len(clique_list) >= 2:
                    cliques.append(clique_list)
                
                # Remove all edges within C from remaining
                for u, w in combinations(sorted(C), 2):
                    edge = (min(u, w), max(u, w))
                    if edge in remaining_edge_set:
                        remaining_edge_set.discard(edge)
                
                # Update adj_rem
                for u in C:
                    for w in C:
                        if u != w:
                            if w in adj_rem[u]:
                                adj_rem[u].discard(w)
                                adj_rem[w].discard(u)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > _deadline - 5:
            break
        results.append(solve_one(*spec))
    
    return results
# EVOLVE-BLOCK-END
