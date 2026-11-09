import time
import itertools
import sys

def _is_2_colorable(n, edges):
    """
    Exact 2-colourability check via backtracking with unit propagation.
    Returns True if 2-colourable, False otherwise.
    """
    if not edges:
        return True

    # Deduplicate and sort edges
    unique_edges = list(set(tuple(sorted(e)) for e in edges))
    if not unique_edges:
        return True

    # Precompute for each vertex, the list of edge indices containing it
    # And for each edge, the set of vertices
    edge_vertices = [set(e) for e in unique_edges]
    
    # Degree ordering
    deg = [0] * n
    for e in unique_edges:
        for v in e:
            deg[v] += 1
    
    order = sorted(range(n), key=lambda v: -deg[v])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # Remap edges
    remapped = [tuple(pos[v] for v in e) for e in unique_edges]
    
    # For unit propagation: for each edge, we need to know if 4 vertices are same color
    # We'll use a recursive approach with propagation
    
    colors = [-1] * n
    
    def propagate(k):
        changed = True
        while changed:
            changed = False
            for e in remapped:
                c0 = c1 = 0
                unassigned = []
                for v in e:
                    if v < k:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                    else:
                        unassigned.append(v)
                
                if not unassigned:
                    if c0 == 5 or c1 == 5:
                        return False
                else:
                    if c0 == 4 and len(unassigned) == 1:
                        colors[unassigned[0]] = 1
                        changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        colors[unassigned[0]] = 0
                        changed = True
        return True

    def backtrack(k):
        if k == n:
            for e in remapped:
                c0 = sum(1 for v in e if colors[v] == 0)
                if c0 == 5:
                    return False
                c1 = sum(1 for v in e if colors[v] == 1)
                if c1 == 5:
                    return False
            return True
        
        # Try both colors
        for c in [0, 1]:
            colors[k] = c
            if propagate(k + 1):
                if backtrack(k + 1):
                    return True
        colors[k] = -1
        return False
    
    return backtrack(0)


def _greedy_set_cover(n, time_limit=130):
    """
    Greedy set-cover heuristic to find a minimal non-2-colorable 5-uniform hypergraph.
    
    Key insight: A hypergraph is NOT 2-colorable iff for EVERY 2-coloring of vertices,
    at least one edge is monochromatic.
    
    So we need to "cover" all 2^n colorings. Each edge e covers the colorings where
    e is monochromatic (i.e., all 5 vertices in e have the same color). There are 2 such
    colorings per edge (all 0 or all 1 on those 5 vertices, arbitrary elsewhere).
    
    We want the minimum number of edges such that every coloring has at least one
    monochromatic edge.
    """
    start_time = time.time()
    
    # All possible 5-element subsets
    all_edges = list(itertools.combinations(range(n), 5))
    
    # For each coloring (represented as an integer 0..2^n-1), determine which edges are monochromatic
    # A coloring is a bitstring of length n. Edge (v1..v5) is monochromatic if all 5 bits are same.
    
    # For efficiency, represent colorings as integers where bit i = color of vertex i
    # Edge (v1,v2,v3,v4,v5) is monochromatic if all bits at positions v1..v5 are 0 OR all are 1.
    
    num_colorings = 1 << n
    
    # For each edge, compute the bitmask of colorings it covers (makes monochromatic)
    # Coloring c covers edge e if for all v in e, (c >> v) & 1 is the same value.
    # This means: either all bits at positions in e are 0, or all are 1.
    # The set of colorings where all bits in e are 0: fix those 5 bits to 0, others free.
    # The set where all bits in e are 1: fix those 5 bits to 1, others free.
    
    # For edge e = (v1,...,v5), the mask of free bits is ~((1<<v1)|...|(1<<v5)) & ((1<<n)-1)
    # Colorings with all 0 on e: bits in e are 0, free bits arbitrary. That's 2^(n-5) colorings.
    # Colorings with all 1 on e: bits in e are 1, free bits arbitrary. That's 2^(n-5) colorings.
    
    # We need to track which colorings are "covered" (have at least one monochromatic edge).
    # Initially no coloring is covered. We add edges greedily, each time picking the edge
    # that covers the most uncovered colorings.
    
    # Represent covered colorings as a set of integers (or a bitset for large n)
    # For n=11, 2^11 = 2048, manageable as a Python set or boolean list.
    # For n=12, 2^12 = 4096, still fine.
    
    uncovered = set(range(num_colorings))
    
    selected_edges = []
    
    # Precompute for each edge the set of colorings it covers
    edge_coverage = {}
    for e in all_edges:
        # Colorings where all 5 bits are 0
        mask_0 = 0
        for v in e:
            mask_0 |= (1 << v)
        
        # Free bits mask
        free_mask = ((1 << n) - 1) & ~mask_0
        
        # All colorings with these 5 bits = 0
        cover_0 = set()
        for free_val in range(1 << (n - 5)):
            # Distribute free bits into the free positions
            c = 0
            bit_idx = 0
            for v in range(n):
                if v not in e:
                    c |= ((free_val >> bit_idx) & 1) << v
                    bit_idx += 1
            cover_0.add(c)
        
        # All colorings with these 5 bits = 1
        cover_1 = set()
        base = mask_0  # all 5 bits set
        for free_val in range(1 << (n - 5)):
            c = base
            bit_idx = 0
            for v in range(n):
                if v not in e:
                    c |= ((free_val >> bit_idx) & 1) << v
                    bit_idx += 1
            cover_1.add(c)
        
        edge_coverage[e] = cover_0 | cover_1
    
    # Greedy set cover
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        best_edge = None
        best_cover_count = -1
        
        for e in all_edges:
            cov = edge_coverage[e]
            count = len(cov & uncovered)
            if count > best_cover_count:
                best_cover_count = count
                best_edge = e
        
        if best_edge is None or best_cover_count == 0:
            break
        
        selected_edges.append(best_edge)
        uncovered -= edge_coverage[best_edge]
        
        # Remove best_edge from consideration? No, we can reuse but it won't help.
        # Actually once selected, it's already covered. But the edge itself is still valid.
        # We just need to not re-select it.
        all_edges.remove(best_edge)
        
        if not uncovered:
            break
    
    if uncovered:
        return None  # Couldn't cover all colorings in time
    
    return [list(e) for e in selected_edges]


def _try_remove_edges(n, edges, time_limit=5):
    """
    Try to remove edges while maintaining non-2-colorability.
    """
    start_time = time.time()
    current = list(edges)
    
    # Try removing each edge in random order
    import random
    indices = list(range(len(current)))
    
    for _ in range(50):  # Multiple passes
        if time.time() - start_time > time_limit:
            break
        random.shuffle(indices)
        progress = False
        for i in indices:
            if i >= len(current):
                continue
            if time.time() - start_time > time_limit:
                break
            trial = current[:i] + current[i+1:]
            if not _is_2_colorable(n, trial):
                current = trial
                progress = True
                break
        if not progress:
            break
    
    return current


def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s margin before 150s kill
    
    # Strategy: Use greedy set cover on small n values
    # n=11: 2^11=2048 colorings, C(11,5)=462 edges
    # n=10: 2^10=1024 colorings, C(10,5)=252 edges
    # n=12: 2^12=4096 colorings, C(12,5)=792 edges
    
    best_result = None
    best_edges_count = float('inf')
    
    for n in [11, 10, 12, 13]:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            break
        
        try:
            edges = _greedy_set_cover(n, time_limit=min(remaining - 5, 60))
            if edges is not None:
                # Try to reduce
                edges = _try_remove_edges(n, edges, time_limit=min(5, remaining - 10))
                
                if len(edges) < best_edges_count:
                    best_edges_count = len(edges)
                    best_result = {'n': n, 'edges': edges}
                    
                    # If we found something good, stop
                    if len(edges) <= 55:
                        break
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}


if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, edges={len(res['edges'])}")
