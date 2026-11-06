"""Strong edge colouring via clique-cover heuristic on the strong conflict graph."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmask(edges, adj):
        """Build conflict graph as bitmasks."""
        m = len(edges)
        edge_idx = {e: i for i, e in enumerate(edges)}
        conflict = [0] * m

        for i, (u, v) in enumerate(edges):
            for w in adj.get(u, ()):
                e2 = (min(u, w), max(u, w))
                if e2 in edge_idx and e2 != (u, v):
                    conflict[i] |= (1 << edge_idx[e2])
            for w in adj.get(v, ()):
                e2 = (min(v, w), max(v, w))
                if e2 in edge_idx and e2 != (u, v):
                    conflict[i] |= (1 << edge_idx[e2])
            for x in adj.get(u, ()):
                for y in adj.get(x, ()):
                    if y == u:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        conflict[i] |= (1 << edge_idx[e2])
            for x in adj.get(v, ()):
                for y in adj.get(x, ()):
                    if y == v:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_idx and e2 != (u, v):
                        conflict[i] |= (1 << edge_idx[e2])

        return conflict

    def greedy_induced_matching(conflict, m, order):
        """Greedy: build colour classes as induced matchings."""
        colour = [-1] * m
        colour_masks = []  # list of bitmasks, one per colour class

        for i in order:
            placed = False
            for c, mask in enumerate(colour_masks):
                if (conflict[i] & mask) == 0:
                    colour[i] = c
                    colour_masks[c] |= (1 << i)
                    placed = True
                    break
            if not placed:
                c = len(colour_masks)
                colour[i] = c
                colour_masks.append(1 << i)

        return colour, len(colour_masks), colour_masks

    def try_eliminate_colour(conflict, m, colour, colour_count, colour_masks, max_c):
        """Try to eliminate colour max_c by moving its edges to other classes."""
        if max_c >= colour_count:
            return False

        mask = colour_masks[max_c]
        to_move = [i for i in range(m) if (mask >> i) & 1]
        if not to_move:
            return False

        new_colour = list(colour)
        new_masks = list(colour_masks)

        for i in to_move:
            new_masks[max_c] &= ~(1 << i)
            found = False
            for c in range(max_c):
                if (conflict[i] & new_masks[c]) == 0:
                    new_masks[c] |= (1 << i)
                    new_colour[i] = c
                    found = True
                    break
            if not found:
                return False

        colour[:] = new_colour
        colour_masks[:] = new_masks
        return True

    def two_swap_improve(conflict, m, colour, colour_count, colour_masks, max_iters=100):
        """Try 2-edge swaps to reduce colour count."""
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for i in range(m):
                if time.time() > deadline:
                    return colour_count
                ci = colour[i]
                for j in range(i + 1, m):
                    cj = colour[j]
                    if ci == cj:
                        continue
                    # Try swapping: i->cj, j->ci
                    mask_ci = colour_masks[ci] & ~(1 << i)
                    mask_cj = colour_masks[cj] & ~(1 << j)
                    if (conflict[i] & mask_cj) == 0 and (conflict[j] & mask_ci) == 0:
                        # Perform swap
                        colour[i], colour[j] = cj, ci
                        colour_masks[ci] = mask_ci | (1 << j)
                        colour_masks[cj] = mask_cj | (1 << i)
                        improved = True
                        # Check if we can eliminate a colour
                        for c in range(colour_count - 1, -1, -1):
                            if colour_masks[c] == 0:
                                colour_count -= 1
                                del colour_masks[c]
                                for k in range(m):
                                    if colour[k] > c:
                                        colour[k] -= 1
                                break
                        break
                if improved:
                    break
        return colour_count

    def clique_cover_greedy(conflict, m, order):
        """
        Structural approach: treat each colour class as a clique in the
        non-conflict graph (i.e., an induced matching). Use greedy set-cover
        style: repeatedly pick the edge with highest 'pressure' and assign it
        to the best existing class or create a new one.
        """
        colour = [-1] * m
        colour_masks = []

        # Sort by conflict degree descending for better packing
        sorted_order = sorted(order, key=lambda i: -bin(conflict[i]).count('1'))

        for i in sorted_order:
            best_c = -1
            best_score = -1
            for c, mask in enumerate(colour_masks):
                if (conflict[i] & mask) == 0:
                    # Score: prefer classes with fewer edges (to balance)
                    score = -bin(mask).count('1')
                    if score > best_score:
                        best_score = score
                        best_c = c
            if best_c >= 0:
                colour[i] = best_c
                colour_masks[best_c] |= (1 << i)
            else:
                c = len(colour_masks)
                colour[i] = c
                colour_masks.append(1 << i)

        return colour, len(colour_masks), colour_masks

    def local_refinement(conflict, m, colour, colour_count, colour_masks):
        """Iterative refinement: eliminate colours, then 2-swaps."""
        # Repeatedly try to eliminate the highest colour
        changed = True
        while changed:
            changed = False
            for max_c in range(colour_count - 1, -1, -1):
                if try_eliminate_colour(conflict, m, colour, colour_count, colour_masks, max_c):
                    colour_count -= 1
                    del colour_masks[max_c]
                    for k in range(m):
                        if colour[k] > max_c:
                            colour[k] -= 1
                    changed = True
                    break

        # 2-swap improvements
        colour_count = two_swap_improve(conflict, m, colour, colour_count, colour_masks)

        # Another round of elimination
        changed = True
        while changed:
            changed = False
            for max_c in range(colour_count - 1, -1, -1):
                if try_eliminate_colour(conflict, m, colour, colour_count, colour_masks, max_c):
                    colour_count -= 1
                    del colour_masks[max_c]
                    for k in range(m):
                        if colour[k] > max_c:
                            colour[k] -= 1
                    changed = True
                    break

        return colour_count

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            out.append([])
            continue

        edges = instance(*spec)
        m = len(edges)
        if m == 0:
            out.append([])
            continue

        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        conflict = build_conflict_bitmask(edges, adj)

        rng = random.Random(42)
        best_colour = None
        best_cc = float('inf')

        # Strategy 1: DSATUR-like greedy
        order1 = list(range(m))
        order1.sort(key=lambda i: -bin(conflict[i]).count('1'))
        colour, cc, masks = greedy_induced_matching(conflict, m, order1)
        cc = local_refinement(conflict, m, colour, cc, masks)
        if cc < best_cc:
            best_cc = cc
            best_colour = list(colour)

        # Strategy 2: Clique-cover greedy with balancing
        order2 = list(range(m))
        colour, cc, masks = clique_cover_greedy(conflict, m, order2)
        cc = local_refinement(conflict, m, colour, cc, masks)
        if cc < best_cc:
            best_cc = cc
            best_colour = list(colour)

        # Strategy 3-12: Randomized restarts
        for trial in range(12):
            if time.time() > deadline:
                break

            order = list(range(m))
            rng.shuffle(order)
            # Mix of degree-biased and random
            if trial % 3 == 0:
                order.sort(key=lambda i: -bin(conflict[i]).count('1'))
            elif trial % 3 == 1:
                order.sort(key=lambda i: (-bin(conflict[i]).count('1'), rng.random()))
            # else: keep random

            colour, cc, masks = greedy_induced_matching(conflict, m, order)
            cc = local_refinement(conflict, m, colour, cc, masks)
            if cc < best_cc:
                best_cc = cc
                best_colour = list(colour)

        out.append([best_colour[i] for i in range(m)])

    return out
# EVOLVE-BLOCK-END
