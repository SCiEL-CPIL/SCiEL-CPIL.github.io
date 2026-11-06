"""Clique partition via Simulated Annealing on clique partitions."""
import time
import random
import heapq
import sys

_DEADLINE = 140.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _random_graph(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _validate(n, edges, cliques):
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set


def _is_clique(verts, adj):
    """Check if a set of vertices (list or set) forms a clique."""
    if len(verts) < 2:
        return False
    # Convert to set for fast lookup if needed, but adj is bitmask
    # Simple pair check
    for i in range(len(verts)):
        u = verts[i]
        for j in range(i + 1, len(verts)):
            v = verts[j]
            if not (adj[u] & (1 << v)):
                return False
    return True


def _greedy_initial_partition(n, edges, adj):
    """Generate an initial partition using greedy maximal clique cover."""
    # This is a simplified version of the seed's logic, 
    # but we just need a valid starting point for SA.
    # We will use the exact logic from the seed to get a good start.
    # However, to avoid code duplication and potential bugs, 
    # we can just start with all edges as 2-cliques and let SA improve it?
    # No, all edges is too far. 
    # Let's reuse the seed's _solve_instance logic but return the intermediate state?
    # Or just implement a fast greedy:
    
    uncovered = set(edges)
    cliques = []
    
    # Find all maximal cliques? That's expensive.
    # Instead, let's do a simple greedy:
    # While uncovered edges exist:
    #   Pick an edge (u,v) from uncovered.
    #   Find the largest clique containing u and v.
    #   Add it to cliques.
    #   Remove its edges from uncovered.
    
    # To find largest clique containing u,v:
    # Start with {u,v}. 
    # Candidates: vertices w such that w connected to u and v.
    # Greedily add w that maximizes connections to current clique.
    
    # This is O(E * N^2) which is fine for N=64.
    
    # Precompute common neighbors for pairs? 
    # adj[u] & adj[v] gives common neighbors.
    
    while uncovered:
        # Pick an arbitrary edge
        u, v = next(iter(uncovered))
        
        # Build a clique containing u and v
        clique = {u, v}
        candidates = set()
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # Iterate bits
        temp = common
        while temp:
            lb = temp & (-temp)
            w = lb.bit_length() - 1
            temp ^= lb
            candidates.add(w)
        
        # Greedily expand clique
        # We want to add vertices that are connected to all in clique.
        # Since we start with u,v, any w in candidates is connected to u,v.
        # When we add w, new candidates must be connected to w as well.
        
        current_clique = set(clique)
        while True:
            # Find candidates connected to all in current_clique
            valid_candidates = set()
            for w in candidates:
                is_connected = True
                for x in current_clique:
                    if not (adj[w] & (1 << x)):
                        is_connected = False
                        break
                if is_connected:
                    valid_candidates.add(w)
            
            if not valid_candidates:
                break
            
            # Pick one that maximizes something? 
            # Just pick one that has most connections to other valid candidates?
            # Or just pick the first one to save time.
            # Let's pick the one with max degree in the subgraph of valid_candidates?
            # For simplicity and speed, pick random or first.
            # Let's pick the one with max |N(w) ∩ valid_candidates|
            best_w = None
            max_conn = -1
            for w in valid_candidates:
                conn = 0
                for x in valid_candidates:
                    if x != w and (adj[w] & (1 << x)):
                        conn += 1
                if conn > max_conn:
                    max_conn = conn
                    best_w = w
            
            if best_w is not None:
                current_clique.add(best_w)
                # Remove best_w from candidates
                candidates.remove(best_w)
                # Update candidates: only those connected to best_w remain valid
                # Actually, the loop above re-evaluates all candidates against current_clique.
                # This is O(|candidates| * |clique|).
                # Since n is small, this is fine.
            else:
                break
        
        cliques.append(current_clique)
        
        # Remove edges covered by this clique
        k = len(current_clique)
        verts = list(current_clique)
        for i in range(k):
            a = verts[i]
            for j in range(i + 1, k):
                b = verts[j]
                e = (a, b) if a < b else (b, a)
                if e in uncovered:
                    uncovered.remove(e)
                    
    return [list(c) for c in cliques]


def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    edge_set = set(edges)
    if not edges:
        return []

    # Adjacency bitmasks
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # 1. Initial Partition
    cliques = _greedy_initial_partition(n, edges, adj)
    # Convert to list of frozensets for easier manipulation
    cliques = [frozenset(c) for c in cliques]
    
    # 2. Local Search (Simulated Annealing)
    # Cost: number of cliques
    current_cost = len(cliques)
    best_cliques = list(cliques)
    best_cost = current_cost
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    rng = random.Random(seed + 100)
    
    while T > min_T and _elapsed() < _DEADLINE - 5:
        # Generate a move
        move_type = rng.random()
        new_cliques = None
        delta_cost = 0
        
        if move_type < 0.5:
            # Merge: Pick two random cliques
            if len(cliques) < 2:
                continue
            i, j = rng.sample(range(len(cliques)), 2)
            c1 = cliques[i]
            c2 = cliques[j]
            union = c1 | c2
            if _is_clique(list(union), adj):
                # Merge
                new_cliques = list(cliques)
                # Remove i and j, add union
                # To keep list stable, just create new list
                new_cliques = [c for k, c in enumerate(cliques) if k != i and k != j]
                new_cliques.append(union)
                delta_cost = -1
        elif move_type < 0.8:
            # Shift: Move vertex from C1 to C2
            if len(cliques) < 2:
                continue
            # Pick a clique with size >= 3
            candidates = [i for i, c in enumerate(cliques) if len(c) >= 3]
            if not candidates:
                continue
            i = rng.choice(candidates)
            c1 = cliques[i]
            v = rng.choice(list(c1))
            
            # Pick another clique
            j = rng.choice([k for k in range(len(cliques)) if k != i])
            c2 = cliques[j]
            
            # Check if v can move to c2
            # 1. v connected to all in c2
            can_move = True
            for u in c2:
                if not (adj[v] & (1 << u)):
                    can_move = False
                    break
            
            if can_move:
                # 2. c1 \ {v} is a clique
                c1_rem = c1 - {v}
                if len(c1_rem) >= 2 and _is_clique(list(c1_rem), adj):
                    # Valid shift
                    new_cliques = list(cliques)
                    new_cliques[i] = c1_rem
                    new_cliques[j] = c2 | {v}
                    delta_cost = 0
        else:
            # Split: Split a clique into two
            candidates = [i for i, c in enumerate(cliques) if len(c) >= 4] # Min size 4 to split into 2+2
            if not candidates:
                continue
            i = rng.choice(candidates)
            c = cliques[i]
            verts = list(c)
            # Pick random split
            # Random subset size
            min_size = 2
            max_size = len(c) - 2
            if min_size > max_size:
                continue
            split_size = rng.randint(min_size, max_size)
            subset = set(rng.sample(verts, split_size))
            complement = c - subset
            
            if _is_clique(list(subset), adj) and _is_clique(list(complement), adj):
                new_cliques = list(cliques)
                new_cliques[i] = subset
                new_cliques.append(complement)
                delta_cost = 1

        if new_cliques is None:
            continue
            
        new_cost = current_cost + delta_cost
        
        # Acceptance criterion
        if new_cost < current_cost:
            # Accept
            cliques = new_cliques
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_cliques = list(cliques)
        else:
            # SA acceptance
            if rng.random() < pow(2.71828, -delta_cost / T):
                cliques = new_cliques
                current_cost = new_cost
                
        T *= cooling
        
    # Final validation and fallback
    result = [list(c) for c in best_cliques]
    if not _validate(n, edges, result):
        # Fallback to greedy result if SA broke it (shouldn't happen)
        result = _greedy_initial_partition(n, edges, adj)
        
    return result


def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
