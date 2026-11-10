"""K3,3-free graph on 20 vertices maximizing edges."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _count_common_neighbors(adj, u, v):
    """Count common neighbors of u and v using bitsets."""
    return bin(adj[u] & adj[v]).count('1')

def _has_k33(adj, n):
    """Check if graph has a K3,3 subgraph.
    A K3,3 exists iff there are 3 vertices with >=3 common neighbors,
    OR more generally: there exist sets A={a1,a2,a3} and B={b1,b2,b3}
    such that all 9 edges exist.
    
    Equivalent: there exist 3 vertices a1,a2,a3 such that |N(a1)∩N(a2)∩N(a3)| >= 3
    where we also need the edges within A and within B... 
    
    Actually, K3,3 is bipartite: we need disjoint sets A,B of size 3 with all cross-edges.
    This means: for some 3-set A, the common neighbors of all three vertices in A
    (i.e., N(a1)∩N(a2)∩N(a3)) has size >= 3, AND those 3 common neighbors must be
    pairwise non-adjacent to each other (to form an independent set in B)... 
    
    Wait, no. K3,3 just requires the 9 cross edges. The vertices in B don't need to be
    independent. So: K3,3 exists iff there exist disjoint 3-sets A and B such that
    every vertex in A is adjacent to every vertex in B.
    
    Equivalently: there exist 3 vertices a1,a2,a3 and 3 vertices b1,b2,b3 (all distinct)
    such that each ai is adjacent to each bj.
    
    This is equivalent to: there exist 3 vertices whose common neighborhood has size >= 3.
    Because if N(a1)∩N(a2)∩N(a3) contains b1,b2,b3, then all 9 cross edges exist.
    We just need them to be distinct from a1,a2,a3 (which they are since the graph is simple
    and has no self-loops, so ai is not in N(ai), hence not in the intersection).
    
    Wait, actually ai could be in N(aj) for i≠j. But ai is NOT in N(ai). So ai is not in
    N(a1)∩N(a2)∩N(a3) because it's not in N(ai). So the common neighbors are automatically
    distinct from {a1,a2,a3}.
    
    So: K3,3 exists iff there exist 3 vertices with >= 3 common neighbors.
    """
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                common = adj[a] & adj[b] & adj[c]
                if bin(common).count('1') >= 3:
                    return True
    return False

def _is_k33_free(adj, n):
    """Check if the graph represented by adj (list of ints) is K3,3-free."""
    return not _has_k33(adj, n)

def _edge_count(adj, n):
    """Count edges in the graph."""
    total = 0
    for i in range(n):
        total += bin(adj[i]).count('1')
    return total // 2

def _greedy_construction(n=20, seed=42):
    """Greedy: add edges in order of maximizing density while maintaining K3,3-free."""
    rng = random.Random(seed)
    adj = [0] * n
    edges = []
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        # Check if adding edge (u,v) would create K3,3
        # New common neighbors: for any pair (x,y) where x=u or y=u or x=v or y=v...
        # Actually, adding edge (u,v) creates new K3,3 only if there exist a,b,c such that
        # after adding, |N(a)∩N(b)∩N(c)| >= 3 and the edge (u,v) is critical.
        # 
        # Simpler: just tentatively add and check.
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if _has_k33(adj, n):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            edges.append([u, v])
    
    return edges

def _local_search(edges, n=20, time_limit=100, seed=42):
    """Local search: try to add more edges or swap edges to increase count."""
    rng = random.Random(seed)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    best_edges = list(edges)
    best_count = len(best_edges)
    
    deadline = time.time() + time_limit
    
    # Try adding edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    existing = set((min(u,v), max(u,v)) for u, v in edges)
    missing = [(u, v) for u, v in all_edges if (u, v) not in existing]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _has_k33(adj, n):
                best_edges.append([u, v])
                best_count += 1
                missing.remove((u, v))
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    return best_edges

def _simulated_annealing(n=20, time_limit=110, seed=42):
    """Simulated annealing on edge sets to find dense K3,3-free graphs."""
    rng = random.Random(seed)
    
    # Start with greedy
    adj = [0] * n
    edges = []
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _has_k33(adj, n):
            edges.append((u, v))
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    current_edges = set(edges)
    current_count = len(current_edges)
    
    best_edges = set(current_edges)
    best_count = current_count
    
    deadline = time.time() + time_limit
    temperature = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        if temperature < 0.01:
            temperature = 5.0
            # Reset to best
            current_edges = set(best_edges)
            current_count = best_count
            adj = [0] * n
            for u, v in current_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            continue
        
        # Pick a random move: add, remove, or swap an edge
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random missing edge
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if (u,v) not in current_edges]
            if not missing:
                temperature *= cooling
                continue
            u, v = rng.choice(missing)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if _has_k33(adj, n):
                # Invalid - revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                # Reject (or accept with very low prob)
                if rng.random() < 0.01:
                    current_edges.add((u, v))
                    current_count += 1
                else:
                    pass
            else:
                current_edges.add((u, v))
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = set(current_edges)
        
        elif move_type < 0.6:
            # Remove a random edge
            if not current_edges:
                temperature *= cooling
                continue
            u, v = rng.choice(list(current_edges))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.discard((u, v))
            current_count -= 1
        else:
            # Swap: remove one edge, add another
            if not current_edges:
                temperature *= cooling
                continue
            u1, v1 = rng.choice(list(current_edges))
            missing = [(u, v) for u in range(n) for v in range(u+1, n) 
                      if (u,v) not in current_edges and (u,v) != (u1,v1) and (v,u) != (u1,v1)]
            if not missing:
                temperature *= cooling
                continue
            u2, v2 = rng.choice(missing)
            
            # Tentative
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            if _has_k33(adj, n):
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
            else:
                current_edges.discard((u1, v1))
                current_edges.add((u2, v2))
                # count stays same
        
        temperature *= cooling
    
    return list(best_edges)

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_edges = None
    best_count = 0
    
    # Phase 1: Multiple greedy constructions with different seeds
    for seed in range(20):
        if time.time() > deadline - 80:
            break
        edges = _greedy_construction(n, seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Phase 2: Local search from best
    if time.time() < deadline - 60 and best_edges:
        edges = _local_search(best_edges, n, time_limit=deadline - time.time() - 50, seed=42)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        edges = _simulated_annealing(n, time_limit=deadline - time.time() - 5, seed=42)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    if best_edges is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return best_edges

# EVOLVE-BLOCK-END
