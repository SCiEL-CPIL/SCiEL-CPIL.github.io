"""Structural edge-cover minimization via local search on triangle packing."""
import time
import random
import heapq

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def trivial_solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        return [list(e) for e in edges]

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        m = len(edges)
        edge_set = set(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}

        if time.time() > DEADLINE - 5.0:
            return trivial_solve(n, k, p, seed)

        # Build triangle list: for each triangle (3 vertices), the 3 edge indices
        # Triangle: 3 vertices a<b<c where all 3 edges exist
        triangles = []  # list of (edge_i, edge_j, edge_k, vertices)
        
        # For efficiency, iterate over vertex triples
        # Type 1: all 3 in clique (all < k)
        for a in range(k):
            for b in range(a+1, k):
                for c in range(b+1, k):
                    e1 = (a, b)
                    e2 = (a, c)
                    e3 = (b, c)
                    if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                        triangles.append((edge_idx[e1], edge_idx[e2], edge_idx[e3], (a, b, c)))
        
        # Type 2: 2 in clique, 1 in independent set
        for a in range(k):
            for b in range(a+1, k):
                e_ab = (a, b)
                if e_ab not in edge_set:
                    continue
                for c in range(k, n):
                    e_ac = (a, c)
                    e_bc = (b, c)
                    if e_ac in edge_set and e_bc in edge_set:
                        triangles.append((edge_idx[e_ab], edge_idx[e_ac], edge_idx[e_bc], (a, b, c)))
        
        # For each edge, list of triangle indices that use it
        edge_to_tris = [[] for _ in range(m)]
        for ti, (e1, e2, e3, verts) in enumerate(triangles):
            edge_to_tris[e1].append(ti)
            edge_to_tris[e2].append(ti)
            edge_to_tris[e3].append(ti)
        
        # Greedy initial solution: pack triangles greedily
        # Use priority: prefer triangles that use edges with fewer triangle options
        tri_used = [False] * len(triangles)
        edge_used = [False] * m
        
        # Sort triangles by minimum edge degree (most constrained first)
        tri_order = list(range(len(triangles)))
        for ti in tri_order:
            e1, e2, e3, _ = triangles[ti]
            tri_order[ti] = (min(len(edge_to_tris[e1]), len(edge_to_tris[e2]), len(edge_to_tris[e3])), ti)
        tri_order.sort()
        
        for _, ti in tri_order:
            if tri_used[ti]:
                continue
            e1, e2, e3, _ = triangles[ti]
            if not edge_used[e1] and not edge_used[e2] and not edge_used[e3]:
                tri_used[ti] = True
                edge_used[e1] = True
                edge_used[e2] = True
                edge_used[e3] = True
        
        # Local search: try to improve by swapping
        # Move: remove a triangle, add another that frees up edges
        # Better move: find a triangle T1 that if removed allows adding T2, T3 (net gain)
        
        best_tri_used = tri_used[:]
        best_edge_used = edge_used[:]
        best_score = sum(tri_used)
        
        rng = random.Random(seed)
        
        # Simulated annealing / hill climbing
        current_tri_used = tri_used[:]
        current_edge_used = edge_used[:]
        current_score = sum(current_tri_used)
        
        max_iter = 50000
        iter_count = 0
        
        while iter_count < max_iter:
            if time.time() > DEADLINE - 3.0:
                break
            iter_count += 1
            
            # Try random moves:
            # Move type 1: add an unused triangle (if all 3 edges free)
            # Move type 2: remove a used triangle and add another
            # Move type 3: remove 2 used triangles and add 3 new ones (net +1)
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a triangle
                # Find triangles where all 3 edges are free
                candidates = []
                for ti in range(len(triangles)):
                    if current_tri_used[ti]:
                        continue
                    e1, e2, e3, _ = triangles[ti]
                    if not current_edge_used[e1] and not current_edge_used[e2] and not current_edge_used[e3]:
                        candidates.append(ti)
                
                if not candidates:
                    continue
                
                ti = rng.choice(candidates)
                e1, e2, e3, _ = triangles[ti]
                current_tri_used[ti] = True
                current_edge_used[e1] = True
                current_edge_used[e2] = True
                current_edge_used[e3] = True
                current_score += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_tri_used = current_tri_used[:]
                    best_edge_used = current_edge_used[:]
                    
            elif move_type < 0.7:
                # Swap: remove one used triangle, add one unused
                used_tris = [ti for ti in range(len(triangles)) if current_tri_used[ti]]
                if not used_tris:
                    continue
                
                ti_remove = rng.choice(used_tris)
                e1r, e2r, e3r, _ = triangles[ti_remove]
                
                # Temporarily remove
                current_tri_used[ti_remove] = False
                current_edge_used[e1r] = False
                current_edge_used[e2r] = False
                current_edge_used[e3r] = False
                current_score -= 1
                
                # Find triangles that can now be added
                candidates = []
                for ti in range(len(triangles)):
                    if current_tri_used[ti]:
                        continue
                    e1, e2, e3, _ = triangles[ti]
                    if not current_edge_used[e1] and not current_edge_used[e2] and not current_edge_used[e3]:
                        candidates.append(ti)
                
                if not candidates:
                    # Revert
                    current_tri_used[ti_remove] = True
                    current_edge_used[e1r] = True
                    current_edge_used[e2r] = True
                    current_edge_used[e3r] = True
                    current_score += 1
                    continue
                
                ti_add = rng.choice(candidates)
                e1a, e2a, e3a, _ = triangles[ti_add]
                current_tri_used[ti_add] = True
                current_edge_used[e1a] = True
                current_edge_used[e2a] = True
                current_edge_used[e3a] = True
                current_score += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_tri_used = current_tri_used[:]
                    best_edge_used = current_edge_used[:]
                    
            else:
                # Remove 2, add 3 (net +1)
                used_tris = [ti for ti in range(len(triangles)) if current_tri_used[ti]]
                if len(used_tris) < 2:
                    continue
                
                ti1, ti2 = rng.sample(used_tris, 2)
                e1a, e2a, e3a, _ = triangles[ti1]
                e1b, e2b, e3b, _ = triangles[ti2]
                
                # Temporarily remove both
                current_tri_used[ti1] = False
                current_tri_used[ti2] = False
                for e in (e1a, e2a, e3a, e1b, e2b, e3b):
                    current_edge_used[e] = False
                current_score -= 2
                
                # Find triangles that can be added
                candidates = []
                for ti in range(len(triangles)):
                    if current_tri_used[ti]:
                        continue
                    e1, e2, e3, _ = triangles[ti]
                    if not current_edge_used[e1] and not current_edge_used[e2] and not current_edge_used[e3]:
                        candidates.append(ti)
                
                if len(candidates) < 3:
                    # Revert
                    current_tri_used[ti1] = True
                    current_tri_used[ti2] = True
                    for e in (e1a, e2a, e3a, e1b, e2b, e3b):
                        current_edge_used[e] = True
                    current_score += 2
                    continue
                
                # Pick 3 candidates
                chosen = rng.sample(candidates, 3)
                for ti in chosen:
                    e1, e2, e3, _ = triangles[ti]
                    current_tri_used[ti] = True
                    current_edge_used[e1] = True
                    current_edge_used[e2] = True
                    current_edge_used[e3] = True
                current_score += 3
                
                if current_score > best_score:
                    best_score = current_score
                    best_tri_used = current_tri_used[:]
                    best_edge_used = current_edge_used[:]
        
        # Build final solution from best_tri_used
        cliques = []
        for ti in range(len(triangles)):
            if best_tri_used[ti]:
                cliques.append(list(triangles[ti][3]))
        
        for idx in range(m):
            if not best_edge_used[idx]:
                cliques.append(list(edges[idx]))
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        results.append(solve(*spec))

    out = []
    for lst in results:
        cl = []
        for c in lst:
            cl.append([int(x) for x in c])
        out.append(cl)
    return out

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END
