"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    # Generate hexagonal spiral points
    # Layer 0: (0,0)
    # Layer k has 6k points.

    points = [(0,0)]

    layers = [
        [], # k=0
        [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)],
        [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)],
        [(3,0), (2,1), (1,2), (0,3), (-1,3), (-2,3), (-3,3), (-3,2), (-3,1), (-3,0), (-2,-1), (-1,-2), (0,-3), (1,-3), (2,-3), (3,-3), (3,-2), (3,-1)],
        [(4,0), (3,1), (2,2), (1,3), (0,4), (-1,4), (-2,4), (-3,4), (-4,4), (-4,3), (-4,2), (-4,1), (-4,0), (-3,-1), (-2,-2), (-1,-3), (0,-4), (1,-4), (2,-4), (3,-4), (4,-4), (4,-3), (4,-2), (4,-1)],
        [(5,0), (4,1), (3,2), (2,3), (1,4), (0,5), (-1,5), (-2,5), (-3,5), (-4,5), (-5,5), (-5,4), (-5,3), (-5,2), (-5,1), (-5,0), (-4,-1), (-3,-2), (-2,-3), (-1,-4), (0,-5), (1,-5), (2,-5), (3,-5), (4,-5), (5,-5), (5,-4), (5,-3), (5,-2), (5,-1)]
    ]

    for k in range(1, 6):
        points.extend(layers[k])

    # 1 + 6 + 12 + 18 + 24 + 30 = 91 points
    # Optimal 32 points are within H_4 (first 37 points), but let's use a larger pool.

    # Neighbor offsets for unit distance on triangular lattice
    NEIGHBORS = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]

    def count_edges(pts_list):
        pts_set = set(pts_list)
        edges = 0
        for p in pts_list:
            a, b = p
            for da, db in NEIGHBORS:
                if (a+da, b+db) in pts_set:
                    edges += 1
        return edges // 2

    # Initial solution: first 32 points
    initial_solution = points[:32]
    best_solution = set(initial_solution)
    best_edges = count_edges(list(best_solution))

    # Candidate pool: first 50 points
    candidate_pool = set(points[:50])

    # Incremental edge counting: for a set S, edges(S) = sum over p in S of (number of neighbors of p in S) / 2
    # When we remove p_in and add p_out:
    #   delta = edges involving p_out in (S - {p_in} + {p_out}) - edges involving p_in in S
    #   = (neighbors of p_out in S - {p_in} + {p_out}) - (neighbors of p_in in S)
    #   Since p_out is not in S, neighbors of p_out in S - {p_in} = neighbors of p_out in S minus 1 if p_in is neighbor of p_out
    #   So delta = (deg_S(p_out) - [p_in in N(p_out)]) - deg_S(p_in)
    #   where deg_S(p) = number of neighbors of p that are in S

    # Precompute adjacency within pool for speed
    pool_list = list(candidate_pool)
    pool_set = set(pool_list)

    # Build adjacency: for each point in pool, which other pool points are neighbors
    adj = {}
    for p in pool_list:
        a, b = p
        nbrs = []
        for da, db in NEIGHBORS:
            n = (a+da, b+db)
            if n in pool_set:
                nbrs.append(n)
        adj[p] = nbrs

    # Degree of a point p in a set S: count neighbors of p that are in S
    def deg_in_set(p, S):
        return sum(1 for n in adj[p] if n in S)

    # 1-swap local search with incremental counting
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order for diversification
        p_in_list = list(best_solution)
        random.shuffle(p_in_list)
        p_out_list = list(candidate_pool - best_solution)
        random.shuffle(p_out_list)

        for p_in in p_in_list:
            if time.time() > deadline:
                break
            deg_in = deg_in_set(p_in, best_solution)
            for p_out in p_out_list:
                # Compute delta
                # neighbors of p_out in best_solution
                deg_out_in_set = deg_in_set(p_out, best_solution)
                # Check if p_in is a neighbor of p_out
                is_neighbor = 0
                for n in adj[p_out]:
                    if n == p_in:
                        is_neighbor = 1
                        break
                # delta = (deg_out_in_set - is_neighbor) - deg_in
                delta = (deg_out_in_set - is_neighbor) - deg_in
                if delta > 0:
                    best_edges += delta
                    best_solution.discard(p_in)
                    best_solution.add(p_out)
                    improved = True
                    break  # Restart inner loop
            if improved:
                break  # Restart outer loop

    # 2-swap local search: remove two points, add two points
    if time.time() < deadline:
        improved2 = True
        while improved2 and time.time() < deadline:
            improved2 = False
            p_in_list = list(best_solution)
            random.shuffle(p_in_list)
            p_out_list = list(candidate_pool - best_solution)
            random.shuffle(p_out_list)

            for i in range(len(p_in_list)):
                if time.time() > deadline:
                    break
                p_in1 = p_in_list[i]
                deg_in1 = deg_in_set(p_in1, best_solution)
                for j in range(i+1, len(p_in_list)):
                    p_in2 = p_in_list[j]
                    deg_in2 = deg_in_set(p_in2, best_solution)
                    # Check if p_in1 and p_in2 are neighbors
                    in1_in2_nbr = 0
                    for n in adj[p_in1]:
                        if n == p_in2:
                            in1_in2_nbr = 1
                            break

                    for k in range(len(p_out_list)):
                        if time.time() > deadline:
                            break
                        p_out1 = p_out_list[k]
                        for l in range(k+1, len(p_out_list)):
                            p_out2 = p_out_list[l]
                            # Compute edges gained by adding p_out1, p_out2 to (S - {p_in1, p_in2})
                            # and edges lost by removing p_in1, p_in2 from S
                            # Edges lost: deg_in1 + deg_in2 - in1_in2_nbr (the edge between them counted twice)
                            edges_lost = deg_in1 + deg_in2 - in1_in2_nbr

                            # Edges gained: neighbors of p_out1 in (S - {p_in1,p_in2} + {p_out1,p_out2})
                            # + neighbors of p_out2 in same, but don't double count edge between them
                            # deg of p_out1 in S minus those neighbors that are p_in1 or p_in2, plus 1 if p_out2 is neighbor
                            # deg of p_out2 in S minus those neighbors that are p_in1 or p_in2, plus 1 if p_out1 is neighbor
                            # But we need to be careful: the "in S" part should exclude p_in1, p_in2

                            # Let's compute more carefully:
                            # New set S' = S - {p_in1, p_in2} + {p_out1, p_out2}
                            # Edges in S' involving p_out1: count neighbors of p_out1 in S'
                            #   = (neighbors of p_out1 in S) - (neighbors of p_out1 that are p_in1 or p_in2) + (1 if p_out2 is neighbor of p_out1)
                            # Edges in S' involving p_out2: count neighbors of p_out2 in S'
                            #   = (neighbors of p_out2 in S) - (neighbors of p_out2 that are p_in1 or p_in2) + (1 if p_out1 is neighbor of p_out2)
                            # Total new edges involving p_out1 or p_out2 = above sum, but edge between p_out1 and p_out2 counted once
                            # Actually: edges_involving = deg_{S'}(p_out1) + deg_{S'}(p_out2) - [p_out1 ~ p_out2]
                            # where deg_{S'}(p) counts neighbors in S'

                            deg_out1_in_S = deg_in_set(p_out1, best_solution)
                            deg_out2_in_S = deg_in_set(p_out2, best_solution)

                            # Subtract neighbors that are p_in1 or p_in2
                            out1_adj_in1 = 0
                            out1_adj_in2 = 0
                            out2_adj_in1 = 0
                            out2_adj_in2 = 0
                            for n in adj[p_out1]:
                                if n == p_in1: out1_adj_in1 = 1
                                if n == p_in2: out1_adj_in2 = 1
                            for n in adj[p_out2]:
                                if n == p_in1: out2_adj_in1 = 1
                                if n == p_in2: out2_adj_in2 = 1

                            out1_out2_nbr = 0
                            for n in adj[p_out1]:
                                if n == p_out2:
                                    out1_out2_nbr = 1
                                    break

                            deg_out1_in_S_prime = deg_out1_in_S - out1_adj_in1 - out1_adj_in2 + out1_out2_nbr
                            deg_out2_in_S_prime = deg_out2_in_S - out2_adj_in1 - out2_adj_in2 + out1_out2_nbr

                            edges_gained = deg_out1_in_S_prime + deg_out2_in_S_prime - out1_out2_nbr

                            delta = edges_gained - edges_lost
                            if delta > 0:
                                best_edges += delta
                                best_solution.discard(p_in1)
                                best_solution.discard(p_in2)
                                best_solution.add(p_out1)
                                best_solution.add(p_out2)
                                improved2 = True
                                break
                        if improved2:
                            break
                    if improved2:
                        break
                if improved2:
                    break

    result = [[a, b] for a, b in best_solution]

    # Verify constraints
    if len(result) != 32:
        result = [[a, b] for a, b in initial_solution]

    return result
# EVOLVE-BLOCK-END
