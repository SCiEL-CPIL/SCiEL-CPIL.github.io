"""Clique partition of dense random graphs - structural rewrite."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_LIMIT = 130.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _clique_size(clique, adj, n):
    return len(clique)

def _is_clique(vertices, adj):
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            if not adj[vertices[i], vertices[j]]:
                return False
    return True

def _find_maximal_cliques_bit(adj, n, max_cliques=5000):
    """Find maximal cliques using bit-based Bron-Kerbosch with pivot."""
    # adj as numpy bool matrix; convert to list of bitmasks for speed
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    cliques = []
    
    def bronk(R, P, X):
        if not _deadline_ok():
            return
        if len(cliques) >= max_cliques:
            return
        if P == 0 and X == 0:
            if R > 0:
                cliques.append(R)
            return
        
        # Pick pivot u from P | X with max degree
        PX = P | X
        if PX == 0:
            return
        
        # Find pivot
        best_u = -1
        best_deg = -1
        temp = PX
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            deg = bin(adj_mask[u] & P).count('1')
            if deg > best_deg:
                best_deg = deg
                best_u = u
            temp &= temp - 1
        
        # Candidates: P \ N(u)
        candidates = P & ~(adj_mask[best_u] | (1 << best_u))
        
        while candidates:
            if not _deadline_ok() or len(cliques) >= max_cliques:
                return
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates &= candidates - 1
            
            bronk(R | (1 << v), P & adj_mask[v] & ~(1 << v), X & adj_mask[v] & ~(1 << v))
            P &= ~(1 << v)
            X |= (1 << v)
    
    bronk(0, (1 << n) - 1, 0)
    return cliques

def _mask_to_list(mask):
    result = []
    while mask:
        lsb = mask & (-mask)
        result.append(lsb.bit_length() - 1)
        mask &= mask - 1
    return result

def _edge_key(u, v):
    return (u, v) if u < v else (v, u)

def _clique_edges(clique):
    edges = set()
    for i in range(len(clique)):
        for j in range(i+1, len(clique)):
            edges.add(_edge_key(clique[i], clique[j]))
    return edges

def _greedy_clique_expansion(start_edge, adj, remaining_edges, n):
    """Given a starting edge, greedily expand to a maximal clique in the remaining-edge subgraph."""
    u, v = start_edge
    clique = set([u, v])
    
    # Candidates: vertices w such that (u,w) and (v,w) are in remaining_edges
    candidates = []
    for w in range(n):
        if w == u or w == v:
            continue
        e_uw = _edge_key(u, w)
        e_vw = _edge_key(v, w)
        if e_uw in remaining_edges and e_vw in remaining_edges:
            candidates.append(w)
    
    # Greedy: add vertex with max degree in candidate subgraph (within remaining_edges)
    while candidates:
        if not _deadline_ok():
            break
        best_w = -1
        best_deg = -1
        for w in candidates:
            deg = 0
            for x in candidates:
                if w == x:
                    continue
                e_wx = _edge_key(w, x)
                if e_wx in remaining_edges:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_w = w
        
        if best_w == -1:
            break
        
        clique.add(best_w)
        candidates.remove(best_w)
        
        new_candidates = []
        for w in candidates:
            e_wbw = _edge_key(w, best_w)
            if e_wbw in remaining_edges:
                new_candidates.append(w)
        candidates = new_candidates
    
    return list(clique)

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    adj = _build_adj(n, edges)
    edge_set = set(edges)
    total_edges = len(edge_set)
    
    # Find maximal cliques
    max_clique_masks = _find_maximal_cliques_bit(adj, n, max_cliques=3000)
    
    # Convert to sorted vertex lists
    max_cliques = []
    for mask in max_clique_masks:
        verts = _mask_to_list(mask)
        if len(verts) >= 2:
            max_cliques.append(sorted(verts))
    
    # Sort cliques by size descending (larger first)
    max_cliques.sort(key=lambda c: -len(c))
    
    # Greedy set cover: pick maximal cliques that cover the most uncovered edges
    # Use a scoring function: uncovered edges / (size * log(size))
    best_partition = None
    best_count = float('inf')
    
    rng = random.Random(seed + 7)
    
    num_restarts = 30
    for restart in range(num_restarts):
        if not _deadline_ok():
            break
        
        remaining = set(edge_set)
        partition = []
        
        # Shuffle maximal cliques with bias toward larger ones
        weighted_cliques = []
        for mc in max_cliques:
            # Weight by size
            w = len(mc) ** 2
            for _ in range(w):
                weighted_cliques.append(mc)
        
        if not weighted_cliques:
            weighted_cliques = max_cliques[:]
        
        # Also add individual edges as fallback cliques
        for e in edge_set:
            weighted_cliques.append(sorted(e))
        
        rng.shuffle(weighted_cliques)
        
        # Greedy: repeatedly pick the clique covering most uncovered edges
        while remaining:
            if not _deadline_ok():
                break
            
            best_mc = None
            best_covered = 0
            best_score = -1
            
            for mc in weighted_cliques:
                if len(mc) < 2:
                    continue
                
                # Check if mc is a clique in original graph
                if not _is_clique(mc, adj):
                    continue
                
                # Count uncovered edges in mc
                covered = 0
                for u, v in combinations(mc, 2):
                    e = _edge_key(u, v)
                    if e in remaining:
                        covered += 1
                
                if covered == 0:
                    continue
                
                # Score: covered edges / cost (cost = 1 for this clique)
                # Prefer larger coverage
                score = covered / (len(mc) ** 0.5)  # slight preference for larger cliques
                
                if score > best_score:
                    best_score = score
                    best_covered = covered
                    best_mc = mc
            
            if best_mc is None:
                # No good clique found; add a single edge
                e = next(iter(remaining))
                partition.append(sorted(e))
                remaining.discard(e)
                continue
            
            partition.append(best_mc)
            
            # Remove covered edges
            for u, v in combinations(best_mc, 2):
                e = _edge_key(u, v)
                if e in remaining:
                    remaining.discard(e)
        
        # Handle any remaining edges
        if remaining:
            for e in remaining:
                partition.append(sorted(e))
        
        count = len(partition)
        if count < best_count:
            best_count = count
            best_partition = [list(c) for c in partition]
        
        if not _deadline_ok():
            break
    
    # Local search: try to merge cliques if their union is a clique and edges are fully covered
    if best_partition is not None and _deadline_ok():
        current = [list(c) for c in best_partition]
        
        # Build edge-to-clique map
        def build_edge_map(part):
            emap = {}
            for i, c in enumerate(part):
                for u, v in combinations(c, 2):
                    e = _edge_key(u, v)
                    emap[e] = i
            return emap
        
        emap = build_edge_map(current)
        
        improved = True
        while improved and _deadline_ok():
            improved = False
            new_current = []
            used = [False] * len(current)
            
            for i in range(len(current)):
                if used[i]:
                    continue
                if not _deadline_ok():
                    break
                
                C_i = set(current[i])
                merged = list(C_i)
                
                # Try to merge with other cliques
                for j in range(i+1, len(current)):
                    if used[j]:
                        continue
                    if not _deadline_ok():
                        break
                    
                    C_j = set(current[j])
                    union = C_i | C_j
                    
                    # Check if union is a clique
                    if not _is_clique(list(union), adj):
                        continue
                    
                    # Check if all edges of union are covered by cliques i and j
                    union_edges = set()
                    for u, v in combinations(union, 2):
                        e = _edge_key(u, v)
                        if e in edge_set:  # must be an edge in G
                            union_edges.add(e)
                    
                    # Check coverage
                    covered_by_ij = set()
                    for u, v in combinations(merged, 2):
                        e = _edge_key(u, v)
                        if e in edge_set:
                            covered_by_ij.add(e)
                    for u, v in combinations(C_j, 2):
                        e = _edge_key(u, v)
                        if e in edge_set:
                            covered_by_ij.add(e)
                    
                    # Also need to check that no OTHER clique covers any edge in union_edges
                    # If union_edges == covered_by_ij, then we can merge
                    if union_edges == covered_by_ij:
                        # Merge: replace i and j with union
                        merged = list(union)
                        used[j] = True
                        improved = True
                        C_i = set(merged)
                        continue
                
                new_current.append(merged)
                used[i] = True
        
        # Add any remaining unused cliques
        for i in range(len(current)):
            if not used[i]:
                new_current.append(current[i])
        
        if len(new_current) < len(current):
            current = new_current
    
    if best_partition is None:
        return [[u, v] for u, v in edges]
    
    return current if len(current) < len(best_partition) else best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
