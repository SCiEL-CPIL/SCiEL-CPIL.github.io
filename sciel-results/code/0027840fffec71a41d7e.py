import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 112.0  # Safety margin before 120s kill

    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))

    # 1. Generate all edges and map them to indices
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = idx
                edges.append((u, v))
                idx += 1

    # 2. Generate all squares (2D faces)
    square_set = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)

                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, u), max(v2, u))

                idxs = []
                for e in (e1, e2, e3, e4):
                    if e not in edge_index:
                        continue
                    idxs.append(edge_index[e])

                if len(idxs) == 4:
                    square_set.add(frozenset(idxs))

    squares = [list(s) for s in square_set]
    n_squares = len(squares)

    # 3. Build incidence lists
    # edge_squares[i] = list of square indices containing edge i
    edge_squares = [[] for _ in range(n_edges)]
    for sq_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_squares[e_idx].append(sq_idx)

    # For each square, precompute the other 3 edges for each edge in it
    # square_other_edges[sq_idx][e_idx] = (e1, e2, e3) where {e_idx, e1, e2, e3} = sq
    square_other_edges = [{} for _ in range(n_squares)]
    for sq_idx, sq in enumerate(squares):
        for e_idx in sq:
            others = [e for e in sq if e != e_idx]
            square_other_edges[sq_idx][e_idx] = tuple(others)

    def time_exceeded():
        return (time.time() - start_time) > time_limit

    def run_local_search(initial_set):
        in_set = [False] * n_edges
        for e in initial_set:
            in_set[e] = True

        # blockers[i] = number of squares Q containing i such that other 3 edges of Q are in in_set
        blockers = [0] * n_edges

        for sq_idx, sq in enumerate(squares):
            count_in = 0
            for e in sq:
                if in_set[e]:
                    count_in += 1

            if count_in == 3:
                for e in sq:
                    if not in_set[e]:
                        blockers[e] += 1

        current_size = len(initial_set)

        def add_all_possible():
            nonlocal in_set, blockers, current_size
            changed = True
            while changed:
                changed = False
                for i in range(n_edges):
                    if not in_set[i] and blockers[i] == 0:
                        in_set[i] = True
                        current_size += 1
                        changed = True
                        for sq_idx in edge_squares[i]:
                            sq = squares[sq_idx]
                            for j in sq:
                                if j == i:
                                    continue
                                if not in_set[j]:
                                    others = square_other_edges[sq_idx][j]
                                    if all(in_set[k] for k in others):
                                        blockers[j] += 1

        add_all_possible()

        best_size = current_size
        best_state = in_set[:]

        # Hill Climbing: remove one edge, try to add back
        edges_in_set = [e for e in range(n_edges) if in_set[e]]

        while not time_exceeded():
            random.shuffle(edges_in_set)
            improvement_found = False

            for e in edges_in_set:
                if time_exceeded():
                    break
                if not in_set[e]:
                    continue

                saved_in_set = in_set[:]
                saved_blockers = blockers[:]
                saved_size = current_size

                # Remove e
                in_set[e] = False
                current_size -= 1

                # Update blockers
                for sq_idx in edge_squares[e]:
                    sq = squares[sq_idx]
                    for j in sq:
                        if j == e:
                            continue
                        if not in_set[j]:
                            others = square_other_edges[sq_idx][j]
                            if all(in_set[k] for k in others):
                                blockers[j] -= 1

                # Try to add back
                add_all_possible()

                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    improvement_found = True
                    edges_in_set = [x for x in range(n_edges) if in_set[x]]
                else:
                    in_set = saved_in_set
                    blockers = saved_blockers
                    current_size = saved_size

            if not improvement_found:
                break

        best_edges = []
        for i in range(n_edges):
            if best_state[i]:
                best_edges.append(i)
        return best_edges

    best_edges_global = []
    best_size_global = 0

    # Multiple restarts with randomized greedy initialization
    while not time_exceeded():
        random_edges = list(range(n_edges))
        random.shuffle(random_edges)

        init_set = []
        in_set = [False] * n_edges
        for e in random_edges:
            ok = True
            for sq_idx in edge_squares[e]:
                sq = squares[sq_idx]
                count_in = 0
                for k in sq:
                    if k == e:
                        continue
                    if in_set[k]:
                        count_in += 1
                if count_in == 3:
                    ok = False
                    break
            if ok:
                in_set[e] = True
                init_set.append(e)

        result_edges = run_local_search(init_set)

        if len(result_edges) > best_size_global:
            best_size_global = len(result_edges)
            best_edges_global = result_edges

        if best_size_global >= 308:
            break

    final_edges = []
    for e_idx in best_edges_global:
        u, v = edges[e_idx]
        final_edges.append([u, v])

    return final_edges
