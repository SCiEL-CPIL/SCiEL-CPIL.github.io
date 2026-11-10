import time
import random
from collections import deque

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    n = 16
    FULL = (1 << n) - 1

    def is_3_colorable(adj):
        """Exact 3-colorability via bitset backtracking with MRV + forward checking."""
        colors = [-1] * n
        # order: high degree first
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])

        def backtrack(pos):
            if pos == n:
                return True
            v = order[pos]
            # colors used by colored neighbors
            used = 0
            nb = adj[v]
            while nb:
                lsb = nb & -nb
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    used |= (1 << colors[u])
                nb ^= lsb
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(pos + 1):
                        return True
                    colors[v] = -1
            return False

        return backtrack(0)

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    # common neighbors of i and j must be empty
                    if adj[i] & adj[j]:
                        return False
        return True

    def can_add_edge(adj, u, v):
        # check no common neighbor
        return (adj[u] & adj[v]) == 0

    def greedy_maximize(adj):
        # repeatedly try to add an edge that keeps triangle-free and check non-3-colorability
        # We'll do a simple greedy: try all non-edges, add if triangle-free preserved.
        # To maintain non-3-colorability, we check after each addition; if it becomes 3-colorable,
        # we skip (or we can be more aggressive: add edge, then if 3-colorable, try to fix).
        # Simpler: only add edges that preserve triangle-freeness. The 4-chromatic property
        # may be lost, so we do a final check and back off if needed.
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1 and can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
        return adj

    def local_search(adj, budget):
        """
        Local search: remove edges to break 3-colorability, then greedily re-add.
        Cycle: if current graph is 3-colorable, remove random edges until not 3-colorable,
        then greedily add edges back.
        """
        best = adj[:]
        best_ec = edge_count(best)
        t0 = time.time()
        
        while time.time() - t0 < budget:
            # If best is already not 3-colorable, try to improve it
            if not is_3_colorable(best):
                # Try to add more edges
                improved = True
                while improved and time.time() - t0 < budget:
                    improved = False
                    for u in range(n):
                        for v in range(u + 1, n):
                            if not (best[u] >> v) & 1 and can_add_edge(best, u, v):
                                best[u] |= (1 << v)
                                best[v] |= (1 << u)
                                # Check if still not 3-colorable
                                if not is_3_colorable(best):
                                    ec = edge_count(best)
                                    if ec > best_ec:
                                        best_ec = ec
                                        improved = True
                                else:
                                    # Revert
                                    best[u] &= ~(1 << v)
                                    best[v] &= ~(1 << u)
            else:
                # best is 3-colorable, need to remove edges to make it 4-chromatic
                # Remove random edges until not 3-colorable
                trial = best[:]
                ec = edge_count(trial)
                edges = get_edges(trial)
                random.shuffle(edges)
                for u, v in edges:
                    if time.time() - t0 > budget:
                        break
                    if is_3_colorable(trial):
                        trial[u] &= ~(1 << v)
                        trial[v] &= ~(1 << u)
                        ec -= 1
                    else:
                        break
                if not is_3_colorable(trial):
                    # Now greedily add edges back
                    trial = greedy_maximize(trial)
                    ec = edge_count(trial)
                    if ec > best_ec and not is_3_colorable(trial):
                        best = trial
                        best_ec = ec
                    elif ec > best_ec:
                        # Got more edges but it's 3-colorable, skip
                        pass
                    else:
                        # Try to add edges to trial even if it's 3-colorable now
                        # (greedy_maximize may have made it 3-colorable again)
                        pass
                else:
                    # Couldn't make it 4-chromatic by removing edges? That's unusual.
                    # The Grötzsch-based seed should be 4-chromatic.
                    pass
        
        return best, best_ec

    # Build initial seed: blow-up of Grötzsch graph
    # Grötzsch: 11 vertices. Blow up to 16 by distributing vertices among the 11 parts.
    # Grötzsch graph structure:
    # C5: vertices 0,1,2,3,4 (cycle)
    # For each i in C5, a vertex 5+i connected to i-1 and i+1 in C5
    # Vertex 10 connected to all of 5,6,7,8,9
    # Let's create a blow-up: assign 16 vertices to the 11 Grötzsch vertices
    # Distribution: 16 = 11 + 5, so 5 vertices get an extra copy
    # Or more evenly: some parts get 2, some get 1
    # 16 / 11 ≈ 1.45, so 5 parts get 2, 6 parts get 1
    dist = [2]*5 + [1]*6  # 5 parts of size 2, 6 parts of size 1, total = 10+6=16

    # Map: part_id -> list of actual vertex indices
    part_to_verts = {}
    vid = 0
    for p in range(11):
        part_to_verts[p] = []
        for _ in range(dist[p]):
            part_to_verts[p].append(vid)
            vid += 1

    # Build adjacency
    adj = [0] * n
    # Grötzsch edges (part-level):
    # C5: (0,1),(1,2),(2,3),(3,4),(4,0)
    # For each i in 0..4: (5+i, i-1 mod 5) and (5+i, i+1 mod 5)
    # (10, 5), (10,6), (10,7), (10,8), (10,9)
    groetzsch_edges = []
    for i in range(5):
        groetzsch_edges.append((i, (i+1) % 5))
    for i in range(5):
        groetzsch_edges.append((5+i, (i-1) % 5))
        groetzsch_edges.append((5+i, (i+1) % 5))
    for i in range(5):
        groetzsch_edges.append((10, 5+i))

    # Add all edges between parts that have an edge in Grötzsch
    for a, b in groetzsch_edges:
        for u in part_to_verts[a]:
            for v in part_to_verts[b]:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # This is a complete blow-up, which is triangle-free (since Grötzsch is triangle-free)
    # and not 3-colorable (since Grötzsch is not 3-colorable and blow-ups preserve chromatic number)
    
    # Now try to add more edges while maintaining triangle-freeness and non-3-colorability
    # First, greedily add all possible edges that maintain triangle-freeness
    # This may make it 3-colorable, so we need to be careful

    # Strategy: 
    # 1. Start with the blow-up
    # 2. Try to add edges that maintain triangle-freeness
    # 3. If the result is 3-colorable, we need to remove some edges
    # 4. Use local search to find a dense 4-chromatic triangle-free graph

    # Let's first try the pure greedy (may lose 4-chromaticity)
    trial = adj[:]
    trial = greedy_maximize(trial)
    trial_ec = edge_count(trial)
    trial_4chrom = not is_3_colorable(trial)

    best_adj = adj[:]
    best_ec = edge_count(adj)

    if trial_4chrom and trial_ec > best_ec:
        best_adj = trial
        best_ec = trial_ec

    # Now do local search to find even denser 4-chromatic triangle-free graphs
    remaining_time = DEADLINE - (time.time() - start)
    if remaining_time > 5:
        found, found_ec = local_search(best_adj, remaining_time - 2)
        if found_ec > best_ec:
            best_adj = found
            best_ec = found_ec

    # Final verification
    if not is_triangle_free(best_adj) or is_3_colorable(best_adj):
        # Fallback: return the original blow-up which is guaranteed valid
        best_adj = adj[:]
        best_ec = edge_count(adj)

    return get_edges(best_adj)

# EVOLVE-BLOCK-START

# (The construct function above is the entry point)

# EVOLVE-BLOCK-END
