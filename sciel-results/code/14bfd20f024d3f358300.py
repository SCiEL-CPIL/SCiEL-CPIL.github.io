<<<<<<< SEARCH
import time

# EVOLVE-BLOCK-START
=======
import time
import itertools
import random

# EVOLVE-BLOCK-START
>>>>>>> REPLACE
<<<<<<< SEARCH
    def find_biclique(remaining, adj, n):
        best = None
        best_size = 0
        # 1) edge-seed expansion
        for (u, v) in remaining:
            L, R = expand({u}, {v}, remaining)
            if not valid_biclique(L, R, remaining):
                continue
            s = len(L) * len(R)
            if s > best_size:
                best_size = s
                best = (L, R)
        # 2) common-neighbor seeding
        for u in range(n):
            nbrs = set()
            for v in adj[u]:
                if mk(u, v) in remaining:
                    nbrs.add(v)
            if not nbrs:
                continue
            L, R = expand({u}, nbrs, remaining)
            if not valid_biclique(L, R, remaining):
                continue
            s = len(L) * len(R)
            if s > best_size:
                best_size = s
                best = (L, R)
        return best
=======
    def find_biclique(remaining, adj, n):
        best = None
        best_size = 0
        # 1) edge-seed expansion
        for (u, v) in remaining:
            L, R = expand({u}, {v}, remaining)
            if not valid_biclique(L, R, remaining):
                continue
            s = len(L) * len(R)
            if s > best_size:
                best_size = s
                best = (L, R)
        # 2) common-neighbor seeding
        for u in range(n):
            nbrs = set()
            for v in adj[u]:
                if mk(u, v) in remaining:
                    nbrs.add(v)
            if not nbrs:
                continue
            L, R = expand({u}, nbrs, remaining)
            if not valid_biclique(L, R, remaining):
                continue
            s = len(L) * len(R)
            if s > best_size:
                best_size = s
                best = (L, R)
        # 3) try flipping sides of best found so far
        if best is not None:
            L, R = best
            # Try the reverse orientation
            L2, R2 = set(R), set(L)
            if valid_biclique(L2, R2, remaining):
                s = len(L2) * len(R2)
                if s > best_size:
                    best_size = s
                    best = (L2, R2)
        # 4) try seed with 2-vertex sets on one side
        for u in range(n):
            nbrs = []
            for v in adj[u]:
                if mk(u, v) in remaining:
                    nbrs.append(v)
            if len(nbrs) < 2:
                continue
            # Pick up to 3 pairs of neighbors as seed for right side
            sample_pairs = list(itertools.combinations(nbrs[:6], 2))[:5]
            for pair in sample_pairs:
                L, R = expand({u}, set(pair), remaining)
                if not valid_biclique(L, R, remaining):
                    continue
                s = len(L) * len(R)
                if s > best_size:
                    best_size = s
                    best = (L, R)
        return best
>>>>>>> REPLACE
<<<<<<< SEARCH
    def try_move(bicliques, edge_set, n, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return
                L, R = bicliques[i]
                for side in (0, 1):
                    if side == 0:
                        source, target = L, R
                    else:
                        source, target = R, L
                    for v in source:
                        # candidate new host side = target ∪ {v}
                        host = list(target) + [v]
                        # the other side becomes source without v
                        other = [x for x in source if x != v]
                        # new biclique = [other, host]
                        newL = other
                        newR = host
                        if not newL or not newR:
                            continue
                        # every cross pair must be a host edge
                        ok = True
                        for a in newL:
                            for b in newR:
                                if a == b or mk(a, b) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # the move must not change the covered edge set
                        old_edges = set()
                        for a in L:
                            for b in R:
                                old_edges.add(mk(a, b))
                        new_edges = set()
                        for a in newL:
                            for b in newR:
                                new_edges.add(mk(a, b))
                        if new_edges != old_edges:
                            continue
                        # apply
                        if side == 0:
                            bicliques[i] = [newL, newR]
                        else:
                            bicliques[i] = [newR, newL]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
=======
    def try_move(bicliques, edge_set, n, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return
                L, R = bicliques[i]
                for side in (0, 1):
                    if side == 0:
                        source, target = L, R
                    else:
                        source, target = R, L
                    for v in source:
                        # candidate new host side = target ∪ {v}
                        host = list(target) + [v]
                        # the other side becomes source without v
                        other = [x for x in source if x != v]
                        # new biclique = [other, host]
                        newL = other
                        newR = host
                        if not newL or not newR:
                            continue
                        # every cross pair must be a host edge
                        ok = True
                        for a in newL:
                            for b in newR:
                                if a == b or mk(a, b) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # the move must not change the covered edge set
                        old_edges = set()
                        for a in L:
                            for b in R:
                                old_edges.add(mk(a, b))
                        new_edges = set()
                        for a in newL:
                            for b in newR:
                                new_edges.add(mk(a, b))
                        if new_edges != old_edges:
                            continue
                        # apply
                        if side == 0:
                            bicliques[i] = [newL, newR]
                        else:
                            bicliques[i] = [newR, newL]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            # Also try moving vertices between different bicliques
            if time.time() > deadline:
                return
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return
                for j in range(len(bicliques)):
                    if i == j or time.time() > deadline:
                        continue
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]
                    # Try moving a vertex from Li to Lj (if it connects to all of Rj)
                    for v in Li:
                        if v in Lj or v in Rj:
                            continue
                        ok = all(mk(v, r) in edge_set for r in Rj)
                        if not ok:
                            continue
                        # Check that removing v from Li and adding to Lj keeps both valid
                        Li_new = [x for x in Li if x != v]
                        Lj_new = Lj + [v]
                        if not Li_new or not Lj_new:
                            continue
                        # Li_new must still be a valid biclique with Ri
                        ok_i = all(mk(a, b) in edge_set and a != b for a in Li_new for b in Ri)
                        if not ok_i:
                            continue
                        # Lj_new must still be a valid biclique with Rj
                        ok_j = all(mk(a, b) in edge_set and a != b for a in Lj_new for b in Rj)
                        if not ok_j:
                            continue
                        # Apply
                        bicliques[i] = [Li_new, Ri]
                        bicliques[j] = [Lj_new, Rj]
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
>>>>>>> REPLACE
<<<<<<< SEARCH
    # local refinement: move then merge, a few passes
    for _ in range(3):
        if time.time() > deadline - 4:
            break
        try_move(bicliques, edge_set, n, deadline - 4)
        try_merge(bicliques, edge_set, n, deadline - 4)

    # occasional reseed to escape local minima (only if clearly beneficial)
    if time.time() < deadline - 6 and len(bicliques) > 3:
        for _ in range(2):
            if time.time() > deadline - 6:
                break
            alt = reseed(set(edge_set), adj, n, deadline - 6)
            if len(alt) < len(bicliques):
                bicliques = alt
                try_move(bicliques, edge_set, n, deadline - 4)
                try_merge(bicliques, edge_set, n, deadline - 4)
=======
    # local refinement: move then merge, more passes
    for _ in range(5):
        if time.time() > deadline - 4:
            break
        try_move(bicliques, edge_set, n, deadline - 4)
        try_merge(bicliques, edge_set, n, deadline - 4)

    # occasional reseed to escape local minima (only if clearly beneficial)
    if time.time() < deadline - 6 and len(bicliques) > 3:
        for _ in range(3):
            if time.time() > deadline - 6:
                break
            alt = reseed(set(edge_set), adj, n, deadline - 6)
            if len(alt) < len(bicliques):
                bicliques = alt
                for _ in range(3):
                    if time.time() > deadline - 4:
                        break
                    try_move(bicliques, edge_set, n, deadline - 4)
                    try_merge(bicliques, edge_set, n, deadline - 4)
>>>>>>> REPLACE
