"""Rewritten program for Q3-free graph on 16 vertices, maximizing edges."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    total_edges = 120

    # Q3 adjacency: 8 vertices, each with 3 neighbors
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    # Precompute Q3 edges as set of (i,j) with i<j
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if j in q3_adj[i]:
                q3_edges.add((i, j))

    # All 8-vertex subsets
    all_8sets = list(combinations(range(16), 8))

    def check_q3(mask):
        """Check if graph represented by mask contains Q3 as subgraph."""
        for vs in all_8sets:
            # Build adjacency within vs
            adj = {}
            for i in range(8):
                u = vs[i]
                nbrs = set()
                for j in range(8):
                    if i == j:
                        continue
                    v = vs[j]
                    e = (min(u, v), max(u, v))
                    if mask & (1 << (e[0] * 16 + e[1])):
                        nbrs.add(v)
                adj[u] = nbrs

            # All vertices must have degree >= 3 within vs
            valid = True
            for u in vs:
                if len(adj[u]) < 3:
                    valid = False
                    break
            if not valid:
                continue

            # Try all 8! = 40320 assignments, but prune heavily
            # Backtracking with pruning
            assignment = [-1] * 8
            used = [False] * 8

            def backtrack(qi):
                if qi == 8:
                    return True
                for vi in range(8):
                    if used[vi]:
                        continue
                    ok = True
                    for qj in range(qi):
                        if qj in q3_adj[qi]:
                            if vs[used_index(qj)] not in adj[vs[vi]]:
                                ok = False
                                break
                    # Also check forward neighbors that are already assigned
                    # Actually, let's do it differently: check all already-assigned Q3 vertices
                    for qj in range(qi):
                        if (qi, qj) in q3_edges or (qj, qi) in q3_edges:
                            if vs[used_index(qj)] not in adj[vs[vi]]:
                                ok = False
                                break
                    if ok:
                        assignment[qi] = vi
                        used[vi] = True
                        if backtrack(qi + 1):
                            return True
                        used[vi] = False
                        assignment[qi] = -1
                return False

            # This inner function is too slow. Let's use a different approach.
            # Instead, try all permutations of vs assigned to Q3 vertices 0..7
            # But 40320 * 120 edges check is ~4.8M ops per 8-set, too slow.

            # Better: use bitmasks for the subgraph induced on vs
            # Map vs[0..7] to 0..7
            # Build 8x8 adjacency matrix as bitmask
            sub_adj = [0] * 8
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = vs[i], vs[j]
                    e = (min(u, v), max(u, v))
                    if mask & (1 << (e[0] * 16 + e[1])):
                        sub_adj[i] |= (1 << j)
                        sub_adj[j] |= (1 << i)

            # Now check if this 8-vertex graph contains Q3 as a subgraph
            # Q3 has 12 edges. We need to find a permutation p of 0..7 such that
            # for every (a,b) in q3_edges, sub_adj[p[a]] has bit p[b] set.

            # Use backtracking with bitmasks
            perm = [-1] * 8
            used_bits = 0

            def bt(qi):
                if qi == 8:
                    return True
                # Candidate bits: vertices not yet used that satisfy constraints with already placed Q3 vertices
                # For qi, which Q3 vertices < qi are adjacent to qi?
                candidates = 0
                for k in range(8):
                    if used_bits & (1 << k):
                        continue
                    # Check if k can be placed at qi
                    ok = True
                    for qj in range(qi):
                        if (qi, qj) in q3_edges:
                            pj = perm[qj]
                            if not (sub_adj[k] & (1 << pj)):
                                ok = False
                                break
                    if ok:
                        candidates |= (1 << k)
                while candidates:
                    k = (candidates & -candidates).bit_length() - 1
                    candidates ^= (1 << k)
                    perm[qi] = k
                    if bt(qi + 1):
                        return True
                    perm[qi] = -1
                return False

            # Wait, the above still uses perm[qj] which might be -1. Let me fix.
            # Actually the issue is that I need to check adjacency in the subgraph.
            # Let me redo this more carefully.

            # For each assignment of Q3 vertex qi to subgraph vertex k:
            # For every qj < qi that is adjacent to qi in Q3, we need k adjacent to perm[qj] in subgraph.

            # This is correct. The bt function above should work. Let's test logic:
            # sub_adj[k] & (1 << perm[qj]) checks if k is adjacent to perm[qj] in subgraph.
            # Yes, that's correct.

            if bt(0):
                return True

        return False

    # The above check_q3 is likely too slow. Let's use a faster incremental approach.
    # For now, let's use a simpler but still effective approach.

    def is_q3_free_mask(mask):
        """Faster Q3-free check using precomputed 8-sets."""
        for vs in all_8sets:
            # Quick degree check
            degs = [0] * 8
            for i in range(8):
                u = vs[i]
                for j in range(8):
                    if i == j:
                        continue
                    v = vs[j]
                    e = (min(u, v), max(u, v))
                    if mask & (1 << (e[0] * 16 + e[1])):
                        degs[i] += 1
            if any(d < 3 for d in degs):
                continue

            # Build sub_adj as bitmask
            sub_adj = [0] * 8
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = vs[i], vs[j]
                    e = (min(u, v), max(u, v))
                    if mask & (1 << (e[0] * 16 + e[1])):
                        sub_adj[i] |= (1 << j)
                        sub_adj[j] |= (1 << i)

            # Backtracking with pruning
            perm = [-1] * 8
            used = 0

            def bt(qi):
                if qi == 8:
                    return True
                # Determine candidates
                for k in range(8):
                    if used & (1 << k):
                        continue
                    ok = True
                    for qj in range(qi):
                        if (qi, qj) in q3_edges or (qj, qi) in q3_edges:
                            if not (sub_adj[k] & (1 << perm[qj])):
                                ok = False
                                break
                    if ok:
                        perm[qi] = k
                        if bt(qi + 1):
                            return True
                        perm[qi] = -1
                return False

            if bt(0):
                return False

        return True

    # Start with a good initial graph
    # K_{3,13} plus some extra edges
    def initial_graph():
        mask = 0
        # Add edges between {0,1,2} and {3,4,...,15}
        for a in range(3):
            for b in range(3, 16):
                mask |= (1 << (a * 16 + b))
        # Try to add some edges within {3..15}
        # Use greedy
        for i in range(3, 16):
            for j in range(i+1, 16):
                new_mask = mask | (1 << (i * 16 + j))
                if is_q3_free_mask(new_mask):
                    mask = new_mask
        return mask

    best_mask = initial_graph()
    best_count = bin(best_mask).count('1')

    # Simulated annealing
    random.seed(42)
    T = 1.0
    cool_rate = 0.9995
    min_T = 0.01

    current_mask = best_mask
    current_count = best_count

    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    edge_bit = {(i, j): (1 << (i * 16 + j)) for i, j in all_edges}

    while time.time() < deadline - 10:
        if T < min_T:
            # Restart with higher temperature if stuck
            T = 0.5
            # Keep current best

        # Random move: flip one edge
        e = random.choice(all_edges)
        bit = edge_bit[e]
        new_mask = current_mask ^ bit
        new_count = current_count + (1 if (current_mask & bit) == 0 else -1)

        # Check if new graph is Q3-free
        if is_q3_free_mask(new_mask):
            if new_count > current_count:
                # Accept
                current_mask = new_mask
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_mask = new_mask
            elif random.random() < min(1, (new_count - current_count) / T):
                # Accept with probability
                current_mask = new_mask
                current_count = new_count
        # T cooling
        T *= cool_rate

        # Occasional restart
        if random.random() < 0.01 and T > 0.5:
            current_mask = best_mask
            current_count = best_count
            T = 1.0

    # Final local search from best
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for e in all_edges:
            bit = edge_bit[e]
            if best_mask & bit:
                # Try removing
                new_mask = best_mask ^ bit
                new_count = best_count - 1
                if is_q3_free_mask(new_mask) and new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
                    improved = True
            else:
                # Try adding
                new_mask = best_mask | bit
                new_count = best_count + 1
                if is_q3_free_mask(new_mask):
                    best_mask = new_mask
                    best_count = new_count
                    improved = True

    # Convert mask to list of edges
    result = []
    for i in range(16):
        for j in range(i+1, 16):
            if best_mask & (1 << (i * 16 + j)):
                result.append([i, j])

    return result
# EVOLVE-BLOCK-END
