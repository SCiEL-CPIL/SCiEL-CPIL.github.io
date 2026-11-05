import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))

    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 448

    # Precompute for each edge, the 6 triples of edges that with it form a C4
    # An edge (u,v) with diff bit i; for each other bit j, the C4 is u-v-v'-u'
    # The other 3 edges are: (u,u'), (v,v'), (u',v')
    edge_cycles = [[] for _ in range(num_edges)]
    for (u, v) in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        ei = edge_index[(u, v)]
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            e_opp = (min(u_prime, v_prime), max(u_prime, v_prime))
            e_side1 = (min(u, u_prime), max(u, u_prime))
            e_side2 = (min(v, v_prime), max(v, v_prime))
            # For this edge to complete a C4, all three of these must be present
            # We store the indices of these 3 edges
            edge_cycles[ei].append((edge_index[e_opp], edge_index[e_side1], edge_index[e_side2]))

    # For fast checking, we maintain a set of chosen edge indices
    # When we try to add edge e, we check if any of its C4-triples are fully satisfied

    best_count = 0
    best_edges_set = set()

    # Use a smarter heuristic: instead of pure random shuffle, use a weighted greedy
    # where we prioritize edges that are "harder" to place (more C4 constraints already tight)
    # But first, let's try many random trials with a twist: reverse the order sometimes,
    # and use a slight local improvement pass.

    num_trials = 5000

    for trial in range(num_trials):
        # Random shuffle with some structure variation
        order = list(range(num_edges))
        random.shuffle(order)

        # Occasionally use a bit-reversed or sorted order as starting point
        if trial % 100 == 0:
            order = sorted(range(num_edges), key=lambda i: all_edges[i][0])
        elif trial % 100 == 50:
            order = sorted(range(num_edges), key=lambda i: all_edges[i][1])

        chosen = set()
        chosen_list = []

        for ei in order:
            ok = True
            for (a, b, c) in edge_cycles[ei]:
                if a in chosen and b in chosen and c in chosen:
                    ok = False
                    break
            if ok:
                chosen.add(ei)
                chosen_list.append(ei)

        # Local improvement: try to swap edges to gain more
        # Simple local search: try adding non-chosen edges that can fit, removing conflicts
        # Actually, let's do a simpler improvement: try to find edges not chosen that can be added
        # if we remove one conflicting edge. This is complex, so skip for now.

        if len(chosen_list) > best_count:
            best_count = len(chosen_list)
            best_edges_set = set(chosen_list)

            if best_count >= 308:
                break

    # Convert back to edges
    return [[all_edges[i][0], all_edges[i][1]] for i in best_edges_set]
