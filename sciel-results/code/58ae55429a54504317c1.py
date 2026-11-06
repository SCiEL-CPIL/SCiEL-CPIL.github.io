import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2  # 448

    # 1. Build edges
    edges = []
    edge_to_idx = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx

    # 2. Build squares (faces)
    squares = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(nverts):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ mask
                    e1 = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    e2 = edge_to_idx[(min(v2, v4), max(v2, v4))]
                    e3 = edge_to_idx[(min(v4, v3), max(v4, v3))]
                    e4 = edge_to_idx[(min(v3, v1), max(v3, v1))]
                    squares.append([e1, e2, e3, e4])

    num_squares = len(squares)  # 672

    # 3. Precompute incidence + per-square "other 3 edges" bitmask
    edge_to_squares = [[] for _ in range(total_edges)]
    square_edges = [set(sq) for sq in squares]
    other_mask = [0] * num_squares  # for square s, bitmask of the 3 other edges (per position)
    other_masks_by_edge = [[] for _ in range(total_edges)]  # for edge e, list of masks of squares containing e (each mask = the other 3)

    for s_idx, sq in enumerate(squares):
        for k, e_idx in enumerate(sq):
            edge_to_squares[e_idx].append(s_idx)
            om = 0
            for m in range(4):
                if m != k:
                    om |= (1 << sq[m])
            other_masks_by_edge[e_idx].append(om)

    # 4. Hitting-set (transversal) formulation:
    #    R = set of edges hitting every face.  S = E \ R is C4-free, |S| = 448 - |R|.
    #    Maximize |S|  <=>  minimize |R|.

    def face_unhit(R, s_idx):
        """True if face s_idx has NO edge of R (i.e. all 4 edges in S) -> invalid."""
        return (R & other_mask[s_idx]) == 0

    def is_valid(R):
        for s_idx in range(num_squares):
            if face_unhit(R, s_idx):
                return False
        return True

    def greedy_hitting_set(rng):
        """Greedy set cover to build a hitting set R."""
        uncovered = set(range(num_squares))
        # For each edge, how many currently-uncovered faces it hits
        uncov = [0] * total_edges
        for e in range(total_edges):
            uncov[e] = len(edge_to_squares[e])

        R = 0
        R_set = set()
        while uncovered:
            maxc = -1
            best = []
            for e in range(total_edges):
                if e in R_set:
                    continue
                if uncov[e] > maxc:
                    maxc = uncov[e]
                    best = [e]
                elif uncov[e] == maxc:
                    best.append(e)
            if not best:
                break
            e = rng.choice(best)
            R_set.add(e)
            R |= (1 << e)
            for f in edge_to_squares[e]:
                if f in uncovered:
                    uncovered.discard(f)
                    for e2 in square_edges[f]:
                        if e2 != e:
                            uncov[e2] -= 1
        return R_set

    def remove_redundant(R_set, time_budget=None):
        """Drop edges of R that are not the unique hit for any face."""
        changed = True
        while changed:
            changed = False
            for e in list(R_set):
                redundant = True
                for f in edge_to_squares[e]:
                    c = 0
                    for e2 in square_edges[f]:
                        if e2 in R_set:
                            c += 1
                            if c >= 2:
                                break
                    if c <= 1:
                        redundant = False
                        break
                if redundant:
                    R_set.discard(e)
                    changed = True
        return R_set

    def local_search(R_set, rng, time_budget=5.0):
        """Local search to shrink |R| while keeping every face hit.
        Moves: (a) remove redundant edge, (b) swap e_out for e_in that covers
        all faces e_out uniquely hits, then prune redundancies."""
        start = time.time()
        improved = True
        while improved and (time.time() - start) < time_budget:
            improved = False
            R_list = list(R_set)
            rng.shuffle(R_list)
            for e_out in R_list:
                if time.time() > deadline - 5:
                    return R_set
                # faces that e_out uniquely hits
                unique = []
                for f in edge_to_squares[e_out]:
                    c = 0
                    for e2 in square_edges[f]:
                        if e2 in R_set:
                            c += 1
                    if c == 1:
                        unique.append(f)
                if not unique:
                    R_set.discard(e_out)
                    improved = True
                    break
                # candidate e_in must hit every face in `unique`
                cands = set()
                for f in unique:
                    cands.update(square_edges[f])
                cands.discard(e_out)
                found = False
                cands = list(cands)
                rng.shuffle(cands)
                for e_in in cands:
                    if e_in in R_set:
                        continue
                    ok = True
                    for f in unique:
                        if f not in edge_to_squares[e_in]:
                            ok = False
                            break
                    if ok:
                        R_set.discard(e_out)
                        R_set.add(e_in)
                        R_set = remove_redundant(R_set)
                        improved = True
                        found = True
                        break
                if found:
                    break
        return R_set

    best_size = 0
    best_R = set()

    rng = random.Random(12345)
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
        R_set = greedy_hitting_set(rng)
        R_set = remove_redundant(R_set)
        R_set = local_search(R_set, rng, time_budget=2.0)
        sz = total_edges - len(R_set)
        if sz > best_size:
            best_size = sz
            best_R = set(R_set)

    # Final high-precision polish on the best solution (longer budget, more swaps)
    if time.time() < deadline - 3:
        best_R = local_search(best_R, rng, time_budget=15.0)
        best_R = remove_redundant(best_R)
        best_size = total_edges - len(best_R)

    # Build S = E \ R  ->  list of [u, v]
    result = []
    for e_idx in range(total_edges):
        if e_idx not in best_R:
            u, v = edges[e_idx]
            result.append([u, v])

    return result
