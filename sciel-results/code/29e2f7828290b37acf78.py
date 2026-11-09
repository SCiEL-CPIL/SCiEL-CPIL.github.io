"""Find dense triangle-free non-3-colorable graph on 16 vertices via local search."""
import time
import random
from itertools import combinations

def _is_triangle_free(adj_sets):
    """Check if graph with adjacency sets is triangle-free."""
    n = len(adj_sets)
    for u in range(n):
        for v in adj_sets[u]:
            if v > u:
                for w in adj_sets[v]:
                    if w > v and w in adj_sets[u]:
                        return False
    return True

def _is_3_colorable(adj_sets):
    """Check if graph is 3-colorable via backtracking with bitmask optimization."""
    n = len(adj_sets)
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda x: -len(adj_sets[x]))
    pos = {v: i for i, v in enumerate(order)}
    
    # Precompute neighbor color masks for each vertex
    # For each vertex in order, which earlier neighbors constrain it
    neighbor_masks = []  # neighbor_masks[i] = bitmask of colors used by already-colored neighbors
    # Actually, we need dynamic checking during backtracking
    
    color = [-1] * n
    
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        # Determine available colors via bitmask
        used_mask = 0
        for nb in adj_sets[v]:
            c = color[nb]
            if c != -1:
                used_mask |= (1 << c)
        
        # Try colors 0, 1, 2
        for c in range(3):
            if not (used_mask & (1 << c)):
                # Check consistency with already-colored neighbors
                ok = True
                for nb in adj_sets[v]:
                    if pos[nb] < i and color[nb] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack(i + 1):
                        return True
                    color[v] = -1
        return False
    
    return backtrack(0)

def _is_3_colorable_fast(adj_masks):
    """Fast 3-colorability check using bitmask adjacency. adj_masks[i] is int bitmask of neighbors of i."""
    n = len(adj_masks)
    order = sorted(range(n), key=lambda x: -bin(adj_masks[x]).count('1'))
    pos = {v: i for i, v in enumerate(order)}
    
    color = [-1] * n
    
    def backtrack(i):
        if i == n:
            return True
        v = order[i]
        # Get colors of already-colored neighbors
        used_mask = 0
        mask = adj_masks[v]
        for nb in range(n):
            if (mask >> nb) & 1:
                c = color[nb]
                if c != -1:
                    used_mask |= (1 << c)
        
        for c in range(3):
            if not (used_mask & (1 << c)):
                # Verify no conflict with already-colored neighbors
                ok = True
                mask = adj_masks[v]
                for nb in range(n):
                    if (mask >> nb) & 1 and pos[nb] < i and color[nb] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if backtrack(i + 1):
                        return True
                    color[v] = -1
        return False
    
    return backtrack(0)

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return u * 16 + v

def _key_to_uv(key):
    u = key // 16
    v = key % 16
    return u, v

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Start with Grötzsch graph (Mycielski of C5) on vertices 0-10, plus isolated 11-15
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
    ]
    c5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, w in c5:
        groetzsch_edges.append([u, 5 + w])
        groetzsch_edges.append([w, 5 + u])
    for u in range(5):
        groetzsch_edges.append([5 + u, 10])
    
    best_edges = set(tuple(sorted(e)) for e in groetzsch_edges)
    best_count = len(best_edges)
    
    # Build adjacency sets and masks for best
    best_adj = [set() for _ in range(16)]
    best_masks = [0] * 16
    for u, v in best_edges:
        best_adj[u].add(v)
        best_adj[v].add(u)
        best_masks[u] |= (1 << v)
        best_masks[v] |= (1 << u)
    
    # All possible edges
    all_possible = [(u, v) for u in range(16) for v in range(u + 1, 16)]
    
    # Precompute: for each edge (u,v), the set of vertices that are common neighbors
    # (adding edge would create triangle if any common neighbor exists)
    edge_common_neighbors = {}
    for (u, v) in all_possible:
        common = best_adj[u] & best_adj[v]
        edge_common_neighbors[(u, v)] = common
    
    random.seed(42)
    
    def try_add(adj_sets, edge_set, masks, edge):
        """Try to add edge. Returns (success, new_adj, new_edge_set, new_masks)."""
        u, v = edge
        if v in adj_sets[u]:
            return False, adj_sets, edge_set, masks
        # Check triangle-free: no common neighbor
        common = adj_sets[u] & adj_sets[v]
        if common:
            return False, adj_sets, edge_set, masks
        # Add edge and check 3-colorability
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        new_masks = masks[:]
        new_masks[u] |= (1 << v)
        new_masks[v] |= (1 << u)
        if _is_3_colorable_fast(new_masks):
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
            return False, adj_sets, edge_set, masks
        else:
            edge_set.add(edge)
            return True, adj_sets, edge_set, new_masks
    
    def try_remove(adj_sets, edge_set, masks, edge):
        """Remove edge. Returns (new_adj, new_edge_set, new_masks)."""
        u, v = edge
        if v not in adj_sets[u]:
            return adj_sets, edge_set, masks
        adj_sets[u].remove(v)
        adj_sets[v].remove(u)
        edge_set.discard(edge)
        new_masks = masks[:]
        new_masks[u] &= ~(1 << v)
        new_masks[v] &= ~(1 << u)
        return adj_sets, edge_set, new_masks
    
    # Local search
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try to add edges systematically
        # Sort candidate edges by how "easy" they are (fewer conflicts)
        candidates = [(e, len(best_adj[e[0]] & best_adj[e[1]])) for e in all_possible if e not in best_edges]
        candidates.sort(key=lambda x: x[1])
        
        for edge, conflict_count in candidates:
            if time.time() > deadline:
                break
            if conflict_count > 0:
                continue  # Would create triangle
            adj_copy = [s.copy() for s in best_adj]
            set_copy = set(best_edges)
            masks_copy = best_masks[:]
            success, new_adj, new_set, new_masks = try_add(adj_copy, set_copy, masks_copy, edge)
            if success:
                best_edges = new_set
                best_adj = new_adj
                best_masks = new_masks
                best_count += 1
                improved = True
                break
        
        if not improved:
            # Phase 2: Try swap: remove one edge, add one or two
            swap_improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Pick an edge to remove
                remove_edge = random.choice(list(best_edges))
                adj_copy = [s.copy() for s in best_adj]
                set_copy = set(best_edges)
                masks_copy = best_masks[:]
                adj_copy, set_copy, masks_copy = try_remove(adj_copy, set_copy, masks_copy, remove_edge)
                
                # Try to add edges
                added = 0
                candidates = [e for e in all_possible if e not in set_copy and e != remove_edge]
                random.shuffle(candidates)
                for cand in candidates:
                    if added >= 1:
                        break
                    s2, ns2, ns3, ns4 = try_add(adj_copy, set_copy, masks_copy, cand)
                    if s2:
                        set_copy = ns3
                        masks_copy = ns4
                        adj_copy = ns2
                        added += 1
                        # Try to add one more
                        candidates2 = [e for e in all_possible if e not in set_copy]
                        random.shuffle(candidates2)
                        for cand2 in candidates2:
                            s3, ns2b, ns3b, ns4b = try_add(adj_copy, set_copy, masks_copy, cand2)
                            if s3:
                                best_edges = ns3b
                                best_adj = ns2b
                                best_masks = ns4b
                                best_count += 1
                                improved = True
                                swap_improved = True
                                break
                        break
                if swap_improved:
                    break
        
        if not improved:
            # Phase 3: Try random restarts with small perturbations
            perturbed = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                # Remove 2-3 random edges
                adj_copy = [s.copy() for s in best_adj]
                set_copy = set(best_edges)
                masks_copy = best_masks[:]
                for _ in range(random.randint(2, 3)):
                    if set_copy:
                        rem = random.choice(list(set_copy))
                        adj_copy, set_copy, masks_copy = try_remove(adj_copy, set_copy, masks_copy, rem)
                
                # Try to add edges back
                candidates = [(e, len(adj_copy[e[0]] & adj_copy[e[1]])) for e in all_possible if e not in set_copy]
                candidates.sort(key=lambda x: x[1])
                for edge, conflict_count in candidates:
                    if conflict_count > 0:
                        continue
                    s2, ns2, ns3, ns4 = try_add(adj_copy, set_copy, masks_copy, edge)
                    if s2:
                        set_copy = ns3
                        masks_copy = ns4
                        adj_copy = ns2
                
                if len(set_copy) > best_count:
                    best_edges = set_copy
                    best_adj = adj_copy
                    best_masks = masks_copy
                    best_count = len(set_copy)
                    improved = True
                    perturbed = True
                    break
        
        if not improved and iteration > 50:
            # Try a different random seed for variety
            random.seed(iteration)
    
    # Final: try systematic addition of all possible edges one more time
    for edge in all_possible:
        if time.time() > deadline:
            break
        if edge in best_edges:
            continue
        adj_copy = [s.copy() for s in best_adj]
        set_copy = set(best_edges)
        masks_copy = best_masks[:]
        success, new_adj, new_set, new_masks = try_add(adj_copy, set_copy, masks_copy, edge)
        if success:
            best_edges = new_set
            best_adj = new_adj
            best_masks = new_masks
            best_count += 1
    
    return [list(e) for e in best_edges]
