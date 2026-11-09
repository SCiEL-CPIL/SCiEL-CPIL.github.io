"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
import itertools

# Set recursion limit for Bron-Kerbosch
sys.setrecursionlimit(100000)

# Deadline guard: 150s wall-clock limit, use 140s as safety margin
DEADLINE = 140.0
_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _random_graph_edges(n, p, seed):
    """Generate edges of a random graph using the same RNG as the verifier."""
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    """Build adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    Returns list of frozensets (each a maximal clique).
    """
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P ∪ X that maximizes |N(u) ∩ P|
        union_px = P | X
        if union_px:
            # Pick pivot with max overlap with P
            best_u = max(union_px, key=lambda u: len(adj[u] & P))
            candidates = P - adj[best_u]
        else:
            candidates = P
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            bronk(new_R, new_P, new_X)
            P.remove(v)
            X.add(v)
    
    bronk(frozenset(), frozenset(range(n)), frozenset())
    return cliques

def _solve_hitting_set(n, adj, cliques, deadline_check):
    """
    Find minimum hitting set (transversal) for the given maximal cliques.
    Uses branch and bound with good heuristics.
    
    A hitting set S must intersect every clique in `cliques`.
    We want to minimize |S|.
    """
    # Filter cliques: only those with size >= 2 (already done in bronk)
    # Remove dominated cliques: if clique A is subset of clique B, then hitting A implies hitting B? No, that's wrong.
    # Actually, if A ⊆ B, then any set that hits A might not hit B. But any set that hits B might not hit A.
    # However, if A ⊆ B, then hitting B does NOT imply hitting A. But hitting A does NOT imply hitting B either.
    # Wait: if S hits A (S ∩ A ≠ ∅) and A ⊆ B, then S ∩ B ⊇ S ∩ A ≠ ∅, so S hits B too.
    # So if A ⊆ B, hitting A is sufficient to hit B. Thus we can remove B (the superset) from consideration.
    # We only need to hit the minimal cliques (those not containing any other clique as a proper subset).
    
    # Remove cliques that contain another clique as a proper subset
    cliques_list = list(cliques)
    # Sort by size ascending to make subset checking easier
    cliques_list.sort(key=len)
    minimal_cliques = []
    for c in cliques_list:
        is_superset = False
        for mc in minimal_cliques:
            if mc < c:  # proper subset
                is_superset = True
                break
        if not is_superset:
            minimal_cliques.append(c)
    
    if not minimal_cliques:
        return set()
    
    # Get all vertices that appear in any clique
    all_vertices = set()
    for c in minimal_cliques:
        all_vertices |= c
    
    # Convert to list for indexing
    vertices = sorted(all_vertices)
    vert_index = {v: i for i, v in enumerate(vertices)}
    num_verts = len(vertices)
    
    # For each vertex, which cliques does it hit?
    vert_clique_masks = []
    for v in vertices:
        mask = 0
        for i, c in enumerate(minimal_cliques):
            if v in c:
                mask |= (1 << i)
        vert_clique_masks.append(mask)
    
    num_cliques = len(minimal_cliques)
    full_mask = (1 << num_cliques) - 1
    
    # Lower bound: greedy coloring of the primal or use simple bound
    # Simple lower bound: max clique size / max vertices per clique... not great.
    # Better: use LP relaxation or greedy set cover lower bound.
    
    # Branch and bound
    best_solution = [None]
    best_size = [num_verts + 1]
    
    # Greedy initial solution
    def greedy_solution():
        remaining = full_mask
        chosen = set()
        while remaining:
            # Pick vertex that covers most remaining cliques
            best_v = -1
            best_count = 0
            for i, v in enumerate(vertices):
                count = bin(vert_clique_masks[i] & remaining).count('1')
                if count > best_count:
                    best_count = count
                    best_v = i
            if best_v == -1:
                break
            chosen.add(best_v)
            remaining &= ~vert_clique_masks[best_v]
        return chosen
    
    init = greedy_solution()
    if init:
        best_size[0] = len(init)
        best_solution[0] = init
    
    # Sort vertices by frequency (more frequent first) for better branching
    vert_order = sorted(range(num_verts), key=lambda i: bin(vert_clique_masks[i]).count('1'), reverse=True)
    
    # Precompute for each vertex position in the order
    # Use iterative deepening or simple BnB
    
    # Simple BnB: decide for each vertex in order whether to include it
    # But that's 2^n which is too much. Instead, use the fact that we need to cover all cliques.
    
    # Better approach: recursive search where at each step, if not all cliques are covered,
    # we must pick at least one vertex from some uncovered clique.
    
    # State: (covered_mask, current_set, depth)
    # Branch on an uncovered clique: try each vertex in it.
    
    # Use a priority queue or simple DFS with pruning.
    
    def lower_bound(remaining_mask, current_size):
        """
        Lower bound on additional vertices needed.
        Simple: max over cliques of 1 if not covered, but that's just 1.
        Better: greedy set cover lower bound.
        """
        if remaining_mask == 0:
            return 0
        
        # Simple bound: for each uncovered clique, we need at least 1 more vertex.
        # But one vertex can cover multiple cliques.
        # Use: max number of pairwise disjoint uncovered cliques (packing bound)
        # Or simpler: ceil(number of uncovered cliques / max coverage per vertex)
        
        # Let's use a simple greedy lower bound
        rem = remaining_mask
        count = 0
        while rem:
            # Find vertex that covers most of rem
            best_count = 0
            for i in range(num_verts):
                c = bin(vert_clique_masks[i] & rem).count('1')
                if c > best_count:
                    best_count = c
            if best_count == 0:
                return current_size + 1  # shouldn't happen
            count += 1
            # Remove the best-covered clique (approximation)
            # Actually, let's just do: count += 1 and clear one clique
            # This is an admissible bound if we clear at least one clique per step
            # Find which clique is covered by the best vertex
            # Simplification: just clear the lowest set bit
            lsb = rem & (-rem)
            rem ^= lsb
        return current_size + count
    
    # Actually, the above lower bound is not admissible in general. Let me use a simpler one.
    # Admissible: if remaining_mask != 0, we need at least 1 more vertex.
    # That's weak but admissible.
    
    # Let me use a better approach: the maximum number of pairwise disjoint cliques in remaining.
    # But that's expensive to compute.
    
    # For now, use: lb = current_size + (1 if remaining_mask else 0)
    # And rely on the greedy initial solution to prune.
    
    def bnb(covered_mask, chosen_set, depth):
        if _time_left() < 5:
            return
        
        if covered_mask == full_mask:
            if len(chosen_set) < best_size[0]:
                best_size[0] = len(chosen_set)
                best_solution[0] = set(chosen_set)
            return
        
        # Prune: if current size + 1 >= best_size, we can't improve (need at least 1 more)
        if len(chosen_set) + 1 >= best_size[0]:
            return
        
        # Find an uncovered clique
        uncovered = full_mask ^ covered_mask
        # Pick the first uncovered clique (lowest index)
        cliq_idx = (uncovered & (-uncovered)).bit_length() - 1
        cliq = minimal_cliques[cliq_idx]
        
        # Try each vertex in this clique
        # Order by how many other uncovered cliques it covers
        verts_in_cliq = [vert_index[v] for v in cliq if v in vert_index]
        verts_in_cliq.sort(key=lambda i: bin(vert_clique_masks[i] & uncovered).count('1'), reverse=True)
        
        for vi in verts_in_cliq:
            if vi in chosen_set:
                continue
            new_chosen = chosen_set | {vi}
            new_covered = covered_mask | vert_clique_masks[vi]
            bnb(new_covered, new_chosen, depth + 1)
    
    # Start BnB
    bnb(0, set(), 0)
    
    if best_solution[0] is not None:
        return best_solution[0]
    else:
        # Fallback: return all vertices
        return set(vertices)

def construct():
    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if _time_left() < 10:
            # Not enough time, return greedy for remaining
            edges = _random_graph_edges(n, p, seed)
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            results.append(sorted(touched))
            continue
        
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        if not cliques:
            results.append([])
            continue
        
        solution = _solve_hitting_set(n, adj, cliques, _time_left)
        results.append(sorted(solution))
    
    return results

# EVOLVE-BLOCK-END
