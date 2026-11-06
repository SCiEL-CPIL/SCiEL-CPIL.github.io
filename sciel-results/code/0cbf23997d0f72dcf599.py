"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import defaultdict

N = 40

def _build_seed():
    """Build a valid seed: fan graph + path, cycles as consecutive vertex lists."""
    edges = set()
    # Fan: 0 connected to 1..39
    for i in range(1, N):
        edges.add((0, i))
    # Path: 1-2-3-...-39
    for i in range(1, N - 1):
        edges.add((i, i + 1))
    
    # Cycles: for each length k (3..40), use vertices 0..k-1 in order
    # Cycle 0-1-2-...-(k-1)-0
    cycles = []
    for k in range(3, N + 1):
        cyc = list(range(k))
        cycles.append(cyc)
    
    return edges, cycles

def _edges_from_cycles(cycles):
    """Extract the set of edges required by the given cycles."""
    edges = set()
    for cyc in cycles:
        k = len(cyc)
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return edges

def _validate(edges, cycles):
    """Check that all cycle edges exist in the graph and vertices are distinct."""
    edge_set = set()
    for e in edges:
        a, b = e
        if a > b:
            a, b = b, a
        edge_set.add((a, b))
    
    for cyc in cycles:
        # Check distinct vertices
        if len(set(cyc)) != len(cyc):
            return False
        # Check each edge exists
        k = len(cyc)
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                return False
    return True

def _try_remove_edge(edges, cycles, edge):
    """Try removing an edge; return new edges if still valid, else None."""
    new_edges = edges - {edge}
    if _validate(new_edges, cycles):
        return new_edges
    return None

def _try_add_edge(edges, cycles, edge):
    """Add an edge to the set."""
    return edges | {edge}

def _find_unused_edges(edges, cycles):
    """Find edges in the graph that are not used by any cycle."""
    used = _edges_from_cycles(cycles)
    return edges - used

def _random_cycle(k, rng):
    """Generate a random cycle of length k on vertices 0..39."""
    verts = rng.sample(range(N), k)
    # Random permutation
    rng.shuffle(verts)
    return verts

def _try_repair(edges, cycles, rng, max_tries=200):
    """Try to add random edges to make cycles valid."""
    current_edges = set(edges)
    for _ in range(max_tries):
        if _validate(current_edges, cycles):
            return current_edges
        # Find which cycles are broken
        edge_set = set()
        for e in current_edges:
            a, b = e
            if a > b:
                a, b = b, a
            edge_set.add((a, b))
        
        missing = False
        for cyc in cycles:
            k = len(cyc)
            for i in range(k):
                a = cyc[i]
                b = cyc[(i + 1) % k]
                if a > b:
                    a, b = b, a
                if (a, b) not in edge_set:
                    # Add this missing edge
                    current_edges.add((a, b))
                    edge_set.add((a, b))
                    missing = True
                    break
            if missing:
                break
        if not missing:
            break
    if _validate(current_edges, cycles):
        return current_edges
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    edges, cycles = _build_seed()
    best_edges = set(edges)
    best_score = len(best_edges) - 40
    
    rng = random.Random(42)
    
    # Phase 1: Remove unused edges greedily
    while time.time() < deadline:
        unused = _find_unused_edges(edges, cycles)
        if not unused:
            break
        removed_any = False
        for edge in list(unused):
            if time.time() >= deadline:
                break
            new_edges = _try_remove_edge(edges, cycles, edge)
            if new_edges is not None:
                edges = new_edges
                removed_any = True
        if not removed_any:
            break
    
    if len(edges) - 40 < best_score:
        best_score = len(edges) - 40
        best_edges = set(edges)
    
    # Phase 2: Simulated annealing - try to remove edges, occasionally perturb
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        if len(edges) - 40 < best_score:
            best_score = len(edges) - 40
            best_edges = set(edges)
        
        # Try to remove a random edge
        if edges:
            edge = rng.choice(list(edges))
            new_edges = _try_remove_edge(edges, cycles, edge)
            if new_edges is not None:
                # Accept (removal always reduces edge count)
                edges = new_edges
            else:
                # Try a perturbation: swap an edge
                if len(edges) > 1:
                    # Find a random edge to remove and try to add another
                    rem_edge = rng.choice(list(edges))
                    # Try adding a random edge
                    a = rng.randint(0, N - 1)
                    b = rng.randint(0, N - 1)
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    add_edge = (a, b)
                    
                    # Check if this swap helps
                    candidate = (edges - {rem_edge}) | {add_edge}
                    if _validate(candidate, cycles):
                        # Accept if it doesn't increase edge count (same count)
                        # or with probability based on T
                        edges = candidate
        T *= cooling
    
    if len(edges) - 40 < best_score:
        best_score = len(edges) - 40
        best_edges = set(edges)
    
    # Final cleanup: remove any remaining unused edges
    for _ in range(100):
        if time.time() >= deadline:
            break
        unused = _find_unused_edges(best_edges, cycles)
        if not unused:
            break
        removed = False
        for edge in list(unused):
            new_edges = _try_remove_edge(best_edges, cycles, edge)
            if new_edges is not None:
                best_edges = new_edges
                removed = True
        if not removed:
            break
    
    # Convert to list of tuples
    edge_list = sorted([(min(a,b), max(a,b)) for a,b in best_edges])
    
    return {'edges': edge_list, 'cycles': cycles}

# EVOLVE-BLOCK-END
