"""K3,3-free graph construction on 20 vertices using projective plane polarity + local search."""
import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    n = 20
    
    # Precompute all triples of vertices for K3,3 checking
    all_triples = list(combinations(range(n), 3))
    
    def count_k33_violations(adj_bits):
        """Count the number of triples with >= 3 common neighbors (K3,3 subgraphs)."""
        violations = 0
        for (i, j, k) in all_triples:
            common = (adj_bits[i] & adj_bits[j] & adj_bits[k]).bit_count()
            if common >= 3:
                violations += 1
        return violations
    
    def can_add_edge_fast(adj_bits, u, v):
        """
        Fast check: can we add edge (u,v) without creating a new K3,3?
        After adding edge (u,v), the only new K3,3 violations can involve triples 
        where adding (u,v) creates a new common neighbor relationship.
        
        Actually, adding edge (u,v) changes N(u) and N(v). A triple (a,b,c) becomes 
        a violation if |N(a)&N(b)&N(c)| >= 3 after the addition.
        
        The triples that could newly become violations are those where:
        - One of a,b,c is u or v, and the new edge creates a common neighbor
        - Or the new edge makes a third common neighbor for some triple
        
        For simplicity and correctness, let's just check: after adding (u,v),
        which triples might have their common-neighbor count increase?
        
        Adding edge (u,v): u gains neighbor v, v gains neighbor u.
        For a triple (a,b,c), the common neighbors are N(a)&N(b)&N(c).
        The only way this changes is if one of a,b,c is u or v.
        
        Case 1: triple contains u (and not v, since v is the other endpoint)
          Actually both u and v could be in the triple. Let's handle all cases.
          
        Triples containing u: (u, x, y) for x < y, x,y != u
          Common neighbors before: N(u) & N(x) & N(y)
          Common neighbors after: (N(u) | {v}) & N(x) & N(y) = old + ({v} & N(x) & N(y))
          So new common neighbors = old common + (1 if v in N(x) and v in N(y) else 0)
          A triple (u,x,y) becomes a violation if old_common + [v in N(x) and v in N(y)] >= 3
          
        Similarly for triples containing v.
        Triples containing both u and v: (u, v, x)
          Before: N(u) & N(v) & N(x)
          After: (N(u)|{v}) & (N(v)|{u}) & N(x)
          = (N(u)&N(v)&N(x)) | ({v}&N(v)&N(x)) | (N(u)&{u}&N(x)) | ({v}&{u}&N(x))
          Note: v not in N(v) (no self-loops), u not in N(u).
          {v} & N(v) = empty (v not adjacent to itself)
          {u} & N(u) = empty
          {v} & {u} & N(x) = empty (u != v)
          So actually the common neighbors don't change for triples containing both u and v!
          Wait, that's not right. Let me reconsider.
          
          N(u) after = N(u) before ∪ {v}
          N(v) after = N(v) before ∪ {u}
          
          N(u)_after & N(v)_after & N(x) = (N(u) ∪ {v}) & (N(v) ∪ {u}) & N(x)
          = [(N(u) & N(v)) | (N(u) & {u}) | ({v} & N(v)) | ({v} & {u})] & N(x)
          = [N(u) & N(v)] & N(x)  (since u not in N(u), v not in N(v), u≠v)
          
          So indeed no change for triples containing both u and v.
          
        Triples containing u but not v: (u, x, y) where x,y ≠ v
          New common = old common + 1 if (v in N(x) and v in N(y)) else 0
          
        Triples containing v but not u: (v, x, y) where x,y ≠ u
          New common = old common + 1 if (u in N(x) and u in N(y)) else 0
          
        So we only need to check:
        1. For each pair (x,y) with x,y not in {u,v} and x≠y:
           - If v is adjacent to both x and y, check if triple (u,x,y) now has >= 3 common
           - If u is adjacent to both x and y, check if triple (v,x,y) now has >= 3 common
        """
        # Check triples containing u (but not v)
        for x in range(n):
            if x == u or x == v:
                continue
            # Check if v is adjacent to x
            if not (adj_bits[x] >> v) & 1:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                if not (adj_bits[y] >> v) & 1:
                    continue
                # Triple (u, x, y): count common neighbors after adding (u,v)
                # Common = (N(u) ∪ {v}) & N(x) & N(y)
                # = (N(u) & N(x) & N(y)) | ({v} & N(x) & N(y))
                # Since v in N(x) and v in N(y), {v} & N(x) & N(y) = {v}
                # So common = (N(u) & N(x) & N(y)) | {v}
                old_common = (adj_bits[u] & adj_bits[x] & adj_bits[y]).bit_count()
                # v might already be in old_common if v was already a common neighbor
                # But v is not in N(u) before (we're adding it), so v is not in old_common
                new_common = old_common + 1  # v becomes a common neighbor
                if new_common >= 3:
                    return False
        # Check triples containing v (but not u)
        for x in range(n):
            if x == u or x == v:
                continue
            if not (adj_bits[x] >> u) & 1:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                if not (adj_bits[y] >> u) & 1:
                    continue
                old_common = (adj_bits[v] & adj_bits[x] & adj_bits[y]).bit_count()
                new_common = old_common + 1
                if new_common >= 3:
                    return False
        return True
    
    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
    
    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2
    
    def build_polarity_graph_pg24():
        """
        Build the polarity graph of PG(2,4) on 21 points, then remove one point to get 20 vertices.
        
        PG(2,4) has q^2+q+1 = 16+4+1 = 21 points.
        A polarity maps points to lines such that the absolute points (fixed by polarity)
        form a set. The polarity graph connects two points if one lies on the polar of the other.
        
        For a standard polarity on PG(2,q), the number of edges depends on q.
        For q=4, each point is adjacent to (q+1) + (number of absolute points it's connected to) other points.
        
        Actually, let's use a simpler approach: the graph where vertices are the 21 points of PG(2,4),
        and two vertices are adjacent if they are incident under a symplectic polarity.
        
        A symplectic polarity on PG(2,4) has q+1 = 5 absolute points.
        Each point is adjacent to q = 4 non-absolute points in its polar line, plus...
        
        Let me just construct it explicitly using GF(4).
        
        GF(4) = {0, 1, α, α^2} where α^2 = α + 1 (using primitive polynomial x^2+x+1).
        
        Points of PG(2,4) can be represented as 1-dimensional subspaces of GF(4)^3.
        There are (4^3 - 1)/(4-1) = 63/3 = 21 points.
        
        A standard representation: list all non-zero vectors in GF(4)^3, and identify vectors
        that are scalar multiples.
        
        For a polarity, we use a symmetric bilinear form. A standard symplectic form on GF(4)^3...
        Actually in odd characteristic, symplectic forms have even dimension. PG(2,q) for q even
        can have a symplectic polarity.
        
        Let me use a quadratic form polarity. For q=4 (even), consider the quadratic form
        Q(x,y,z) = x^2 + y^2 + z^2 + x*y + y*z + z*x (or similar).
        
        Actually, this is getting complex. Let me use a known construction:
        
        The Erdős-Rényi graph ER(2,4) or the polarity graph of PG(2,4) has 21 vertices.
        Each vertex has degree (4+1) = 5? No, that's for the absolute points.
        
        For a non-degenerate polarity on PG(2,q) with q even:
        - Number of absolute points: q+1
        - Each absolute point is adjacent to q-1 non-absolute points + 0 absolute points (or some subset)
        - Each non-absolute point is adjacent to q+1 points (including 1 absolute point?)
        
        Total edges: Let's just say we'll construct it and count.
        
        Alternative simpler approach: Use the known fact that the polarity graph of PG(2,4) 
        is K3,3-free (or K5,5-free etc.) and has a specific number of edges.
        
        For simplicity, let me use a different well-known K3,3-free construction:
        The complete bipartite graph K_{2,9} is K3,3-free but has only 18 edges.
        The Clebsch graph (16 vertices, 5-regular) is K3,3-free but we need 20 vertices.
        
        Let me just use a greedy approach with multiple random restarts and the fast 
        can_add_edge check. The key insight is that can_add_edge is much faster than 
        checking all triples.
        """
        # I'll skip the explicit PG(2,4) construction and use multiple random greedy starts
        # with the fast incremental check. This should be sufficient.
        return None
    
    def greedy_build(seed=None):
        """Build a K3,3-free graph using greedy edge addition with fast check."""
        if seed is not None:
            random.seed(seed)
        
        adj_bits = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            add_edge(adj_bits, u, v)
            if can_add_edge_fast(adj_bits, u, v):
                pass  # Keep edge
            else:
                remove_edge(adj_bits, u, v)
        
        return adj_bits
    
    def local_search(adj_bits, max_time=10.0):
        """Local search: try to add missing edges, and do swaps to escape local optima."""
        search_deadline = time.time() + max_time
        current_adj = adj_bits[:]
        current_edges = count_edges(current_adj)
        
        def get_missing_edges(adj):
            return [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] >> j) & 1]
        
        def get_existing_edges(adj):
            return [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
        
        no_progress_rounds = 0
        
        while time.time() < search_deadline:
            # Phase 1: Try adding missing edges
            missing = get_missing_edges(current_adj)
            random.shuffle(missing)
            added_any = False
            
            for u, v in missing:
                if time.time() > search_deadline:
                    break
                # Check if we can add this edge
                # First, temporarily add it
                add_edge(current_adj, u, v)
                if can_add_edge_fast(current_adj, u, v):
                    current_edges += 1
                    added_any = True
                else:
                    remove_edge(current_adj, u, v)
            
            if added_any:
                no_progress_rounds = 0
                continue
            
            no_progress_rounds += 1
            
            # Phase 2: Try swaps
            existing = get_existing_edges(current_adj)
            missing = get_missing_edges(current_adj)
            
            if not existing or not missing:
                break
            
            swapped_any = False
            # Try random swaps
            max_swaps_to_try = 50
            for _ in range(max_swaps_to_try):
                if time.time() > search_deadline:
                    break
                e1 = random.choice(existing)
                e2 = random.choice(missing)
                
                # Temporarily remove e1 and add e2
                u1, v1 = e1
                u2, v2 = e2
                remove_edge(current_adj, u1, v1)
                add_edge(current_adj, u2, v2)
                
                # Check if the graph is still K3,3-free
                # After removing e1 and adding e2, we need to check if any new K3,3 is created
                # The only new K3,3 could be created by adding e2 (removing e1 can't create new violations)
                # But wait, removing e1 changes N(u1) and N(v1), which might affect triples
                # Actually, removing an edge can only REDUCE common neighbors, so it can't create new violations.
                # Adding e2 can create new violations.
                # So we just need to check if adding e2 (in the context of the graph without e1) creates a violation.
                
                if can_add_edge_fast(current_adj, u2, v2):
                    # Valid swap
                    current_edges = current_edges - 1 + 1  # Same count
                    swapped_any = True
                    # Update lists
                    existing.remove(e1)
                    existing.append(e2)
                    missing.remove(e2)
                    missing.append(e1)
                else:
                    # Revert
                    remove_edge(current_adj, u2, v2)
                    add_edge(current_adj, u1, v1)
                
                if swapped_any:
                    break
            
            if swapped_any:
                no_progress_rounds = 0
                continue
            
            # Phase 3: Perturbation - remove a few random edges to escape local minimum
            if no_progress_rounds >= 3 and existing:
                # Remove 2-3 random edges
                num_remove = min(3, len(existing))
                for _ in range(num_remove):
                    if not existing or time.time() > search_deadline:
                        break
                    e = random.choice(existing)
                    remove_edge(current_adj, e[0], e[1])
                    current_edges -= 1
                    existing.remove(e)
                    missing.append(e)
                no_progress_rounds = 0
                continue
            
            if no_progress_rounds > 10:
                break
        
        # Final greedy add
        missing = get_missing_edges(current_adj)
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > search_deadline:
                break
            add_edge(current_adj, u, v)
            if can_add_edge_fast(current_adj, u, v):
                current_edges += 1
            else:
                remove_edge(current_adj, u, v)
        
        return current_adj, current_edges
    
    # Try multiple diverse starting points
    best_adj = None
    best_edges = 0
    
    strategies = [
        lambda: greedy_build(seed=42),
        lambda: greedy_build(seed=123),
        lambda: greedy_build(seed=777),
        lambda: greedy_build(seed=2024),
    ]
    
    for i, strategy in enumerate(strategies):
        if time.time() > deadline - 30:
            break
        
        adj = strategy()
        if adj is None:
            continue
        
        edges = count_edges(adj)
        
        # Do local search on this starting point
        adj_opt, edges_opt = local_search(adj, max_time=min(20.0, max(5.0, deadline - time.time() - 15 - i*5)))
        
        if edges_opt > best_edges:
            best_edges = edges_opt
            best_adj = adj_opt
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 20:
        adj = greedy_build(seed=random.randint(0, 10**9))
        edges = count_edges(adj)
        
        remaining = deadline - time.time() - 10
        if remaining < 3:
            break
        
        adj_opt, edges_opt = local_search(adj, max_time=min(remaining, 15.0))
        
        if edges_opt > best_edges:
            best_edges = edges_opt
            best_adj = adj_opt
    
    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
