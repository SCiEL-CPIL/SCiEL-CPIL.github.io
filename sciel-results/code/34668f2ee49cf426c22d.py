import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # Safety margin

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _is_clique(vertices, adj):
    """Check if vertices form a clique in the graph defined by adj."""
    v_list = list(vertices)
    for i in range(len(v_list)):
        for j in range(i + 1, len(v_list)):
            if v_list[j] not in adj[v_list[i]]:
                return False
    return True

def _edges_of_clique(clique, adj):
    """Get the set of edges present in the graph among the clique vertices."""
    edges = set()
    for u, v in combinations(clique, 2):
        if v in adj[u]:
            edges.add(_edge_key(u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    all_edges_set = set(edges)
    
    # 1. Initial Greedy Partition
    # We want to pack edges into cliques.
    # Strategy: Pick an edge, find the maximal clique containing it in the remaining graph.
    remaining_edges = set(all_edges_set)
    initial_cliques = []
    
    # Sort edges by degree of endpoints to pick "hard" edges first?
    # Or just random.
    edge_list = list(remaining_edges)
    
    while remaining_edges:
        if not _deadline_ok():
            break
            
        # Pick a random edge from remaining
        e = random.choice(list(remaining_edges))
        u, v = e
        
        # Find maximal clique containing u and v in the subgraph of remaining edges
        # Candidates: vertices w such that (u,w) and (v,w) are in remaining_edges
        candidates = []
        for w in range(n):
            if w == u or w == v:
                continue
            if _edge_key(u, w) in remaining_edges and _edge_key(v, w) in remaining_edges:
                candidates.append(w)
                
        clique = [u, v]
        
        # Greedily add candidates
        # To make it a clique, all new vertices must be connected to all existing ones
        # and edges must be in remaining_edges
        changed = True
        while changed:
            changed = False
            # Filter candidates to those connected to all in clique
            new_candidates = []
            for w in candidates:
                ok = True
                for x in clique:
                    if w == x:
                        continue
                    if _edge_key(w, x) not in remaining_edges:
                        ok = False
                        break
                if ok:
                    new_candidates.append(w)
            
            if not new_candidates:
                break
                
            # Pick one, maybe the one with max degree in candidates?
            # For now, just pick the first one to keep it simple, or random
            w = new_candidates[0]
            clique.append(w)
            candidates.remove(w)
            changed = True
            
        # Remove edges of this clique from remaining
        clique_edges = _edges_of_clique(clique, adj)
        for e_c in clique_edges:
            if e_c in remaining_edges:
                remaining_edges.remove(e_c)
                
        initial_cliques.append(clique)
        
    # If we broke out of deadline, we must ensure we return a valid partition
    # If remaining_edges is not empty, we need to cover them.
    # For simplicity, if we are close to deadline, we just add remaining edges as K2.
    if remaining_edges:
        for e in remaining_edges:
            initial_cliques.append([e[0], e[1]])
        remaining_edges = set()

    best_cliques = [list(c) for c in initial_cliques]
    best_cost = len(best_cliques)
    
    # 2. Simulated Annealing / Local Search
    # State: list of cliques (vertex lists)
    # Cost: len(cliques)
    
    # Helper to get edge set of a partition
    def _partition_edge_set(cliques):
        edge_set = set()
        for c in cliques:
            for e in _edges_of_clique(c, adj):
                edge_set.add(e)
        return edge_set

    # Helper to check if a move is valid
    # Move: Merge C_i, C_j
    def _try_merge(cliques, i, j):
        c1 = cliques[i]
        c2 = cliques[j]
        # Check if union is a clique
        union = list(set(c1) | set(c2))
        if not _is_clique(union, adj):
            return None
        
        # Check edge conflicts
        # Edges in union must not be covered by other cliques
        # We need to know which edges are covered by which clique.
        # This is expensive to maintain. 
        # Alternative: 
        # The union clique covers all edges in _edges_of_clique(union, adj).
        # For this to be a valid merge, no edge in _edges_of_clique(union, adj) 
        # should be covered by any clique other than c1 and c2.
        
        # Let's check:
        union_edges = _edges_of_clique(union, adj)
        
        # Edges covered by c1 and c2
        c1_edges = _edges_of_clique(c1, adj)
        c2_edges = _edges_of_clique(c2, adj)
        
        # If an edge in union_edges is in c1 or c2, it's fine.
        # If it's in another clique, it's a conflict.
        
        # We need to check all other cliques.
        for k in range(len(cliques)):
            if k == i or k == j:
                continue
            ck = cliques[k]
            ck_edges = _edges_of_clique(ck, adj)
            # Intersection of union_edges and ck_edges
            if union_edges & ck_edges:
                return None
                
        # If no conflict, merge is valid
        new_cliques = list(cliques)
        new_cliques[i] = union
        new_cliques.pop(j)
        return new_cliques

    # Move: Split a clique
    # Pick a clique, split it into two.
    # How? Remove one edge?
    # If we remove edge (u,v) from C, the vertices might still form a clique if we keep u and v?
    # No, if we remove the edge, the set of edges is no longer a complete graph on the vertex set.
    # So we must also remove a vertex?
    # If we remove vertex w from C, the new clique is C \ {w}.
    # The edges (w, x) for x in C\{w} are now uncovered.
    # We can create a new clique with those edges?
    # This is complex.
    
    # Let's stick to Merge and a simple "Re-pack" of a single clique.
    
    # Re-pack:
    # Take one clique C.
    # Remove its edges from the global coverage.
    # Try to re-insert its edges into other cliques or form new ones.
    
    # This is essentially a 1-opt move.
    
    # Given time constraints, I will implement a simple Hill Climbing with Merge and a "Shake" move.
    # Shake: Randomly split a clique into two random valid sub-cliques.
    
    def _random_split(clique, adj):
        """Split a clique into two valid cliques."""
        if len(clique) < 4:
            return None # Need at least 4 vertices to split into two cliques of size >= 2
            
        # Try to find a partition of vertices into two sets V1, V2 such that both are cliques
        # and the cross edges are not covered by others?
        # Actually, we just need V1 and V2 to be cliques in G.
        # The edges in V1 and V2 will be covered by the new cliques.
        # The cross edges between V1 and V2 must be covered by OTHER cliques.
        
        # This is hard to check without knowing the full partition state.
        # So we will do the split in the context of the full partition.
        
        return None # Not implemented for now

    # Let's just do Merge-based Hill Climbing with Random Restarts.
    
    max_restarts = 50
    for r in range(max_restarts):
        if not _deadline_ok():
            break
            
        current_cliques = [list(c) for c in initial_cliques]
        
        # Randomize the initial order or split?
        # For now, just try to merge.
        
        improved = True
        while improved:
            if not _deadline_ok():
                break
            improved = False
            for i in range(len(current_cliques)):
                for j in range(i + 1, len(current_cliques)):
                    new_cliques = _try_merge(current_cliques, i, j)
                    if new_cliques is not None:
                        current_cliques = new_cliques
                        improved = True
                        break
                if improved:
                    break
                    
        if len(current_cliques) < best_cost:
            best_cost = len(current_cliques)
            best_cliques = [list(c) for c in current_cliques]
            
    return best_cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
