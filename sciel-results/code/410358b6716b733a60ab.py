"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    TOTAL_DEADLINE = t0 + 140.0  # safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = gen_edges(n, p, seed)
        if not edges:
            return []

        # Build adjacency as bitmasks for fast intersection
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Remaining edges tracked via bitmask per vertex
        rem = [0] * n
        for u, v in edges:
            rem[u] |= (1 << v)
            rem[v] |= (1 << u)

        remaining_count = len(edges)
        cliques = []

        def mask_to_list(mask):
            res = []
            while mask:
                lsb = mask & (-mask)
                res.append(lsb.bit_length() - 1)
                mask ^= lsb
            return res

        def find_max_clique_bk():
            """Bron-Kerbosch with pivot selection to find maximum clique in remaining subgraph."""
            if not any(rem):
                return []

            best_clique = []
            best_size = 0

            # Get candidate vertices that have at least one remaining edge
            P_mask = 0
            for v in range(n):
                if rem[v]:
                    P_mask |= (1 << v)

            def bron_kerbosch(P, X, clique):
                nonlocal best_clique, best_size

                if not P and not X:
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = clique[:]
                    return

                # Pivot selection: choose u from P ∪ X that maximizes |N(u) ∩ P|
                PX = P | X
                if not PX:
                    return

                # Find pivot
                best_pivot = None
                best_pivot_count = -1
                temp = PX
                while temp:
                    lsb = temp & (-temp)
                    u = lsb.bit_length() - 1
                    count = bin(P & rem[u]).count('1')
                    if count > best_pivot_count:
                        best_pivot_count = count
                        best_pivot = u
                    temp ^= lsb

                # Iterate over P \ N(pivot)
                candidates = P & ~rem[best_pivot] if best_pivot is not None else P
                while candidates:
                    lsb = candidates & (-candidates)
                    v = lsb.bit_length() - 1
                    candidates ^= lsb

                    new_P = P & rem[v]
                    new_X = X & rem[v]
                    bron_kerbosch(new_P, new_X, clique + [v])

                    P &= ~(1 << v)
                    X |= (1 << v)

                    # Time check
                    if time.time() > t0 + time_budget:
                        return

            bron_kerbosch(P_mask, 0, [])
            return best_clique

        while remaining_count > 0 and time.time() < t0 + time_budget:
            clique = find_max_clique_bk()

            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem[u] &= ~(1 << v)
                            rem[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue

            # Remove all edges within clique from remaining
            removed = 0
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    if (rem[u] >> v) & 1:
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                for u in range(n):
                    mask = rem[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem[u] &= ~(1 << v)
                            rem[v] &= ~(1 << u)
                            remaining_count -= 1
                            break
                        mask ^= lsb
                    if remaining_count < len(edges):
                        break
                continue

            cliques.append(clique)
            remaining_count -= removed

        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = rem[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem[u] &= ~(1 << v)
                    rem[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    # Allocate time budget: more time for the larger instance
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        elapsed = time.time() - t0
        remaining_time = TOTAL_DEADLINE - elapsed
        if remaining_time < 5:
            # Emergency fallback
            edges = gen_edges(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            # Allocate proportional time based on problem size
            budget = min(remaining_time - 5, 70 if n == 32 else 65)
            result.append(solve_instance(n, p, seed, budget))

    return result
# EVOLVE-BLOCK-END
