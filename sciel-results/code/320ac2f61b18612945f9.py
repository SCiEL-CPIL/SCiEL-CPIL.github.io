"""GRASP construction + Simulated Annealing local search to minimize largest book."""
import time
import random
import math

def construct():
    n = 40
    m = 401
    deadline = time.time() + 135  # safety margin before 150s kill

    def all_edges():
        """Generate all possible edges as a list of (u,v) with u<v."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                edges.append((u, v))
        return edges

    def compute_books_from_adj(adj, edge_set):
        """Compute book sizes for all edges in edge_set given adjacency."""
        books = {}
        for u, v in edge_set:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def grasp_construct(randomness_param=0.3):
        """
        Greedy Randomized Adaptive Search Procedure:
        Build graph edge by edge, at each step choose from a candidate list
        of edges that would create the smallest book sizes, with randomization.
        """
        adj = [set() for _ in range(n)]
        edge_set = set()
        books = {}  # edge -> book size

        # Precompute all possible edges
        all_edges_list = all_edges()
        remaining = set(all_edges_list)

        for step in range(m):
            if time.time() > deadline:
                break

            # For each candidate edge, compute what its book size would be
            # if we add it (considering current adj)
            candidate_scores = []
            for (u, v) in remaining:
                # Book size of (u,v) = |N(u) ∩ N(v)|
                bk = len(adj[u] & adj[v])
                candidate_scores.append((bk, (u, v)))

            if not candidate_scores:
                break

            # Sort by book size
            candidate_scores.sort(key=lambda x: x[0])

            # Get the minimum book size
            min_bk = candidate_scores[0][0]

            # Build restricted candidate list: edges with book size <= min_bk + tolerance
            # Higher randomness_param means wider tolerance
            tolerance = int(randomness_param * 3)
            restricted = [(bk, e) for bk, e in candidate_scores if bk <= min_bk + tolerance]

            if not restricted:
                restricted = [candidate_scores[0]]

            # Randomly pick from restricted list
            chosen_bk, chosen_edge = random.choice(restricted)
            u, v = chosen_edge

            # Add edge
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            books[(u, v)] = chosen_bk
            remaining.discard((u, v))

            # Update book sizes of existing edges affected by adding (u,v)
            # An edge (a,b) gets its book size incremented if u or v is a common neighbor
            # Actually, adding (u,v) increases book size of edge (a,b) if:
            #   u is adjacent to both a and b, OR v is adjacent to both a and b
            # More precisely: book(a,b) = |N(a) ∩ N(b)|
            # Adding edge (u,v) means u gains neighbor v and v gains neighbor u.
            # For edge (a,b):
            #   If u is in N(a) and u is in N(b), then u was already counted.
            #   The new edge (u,v) means:
            #     - v becomes a neighbor of u, so for any edge (u, x), x's... 
            #   Actually let me think more carefully.
            #   book(a,b) = number of common neighbors of a and b.
            #   Adding (u,v): 
            #     - u now has v as neighbor, v now has u as neighbor.
            #     - For edge (a,b), a common neighbor c is in N(a) ∩ N(b).
            #     - The only new potential common neighbors are u and v.
            #     - u becomes a common neighbor of (a,b) if a~v and b~v (i.e., v is in N(a) ∩ N(b))
            #       Wait no. u is in N(a) ∩ N(b) iff a~u and b~u.
            #       Adding (u,v) doesn't change whether a~u or b~u.
            #     - v becomes a common neighbor of (a,b) if a~v and b~v.
            #       Adding (u,v) doesn't change whether a~v or b~v either.
            #   
            #   Hmm, actually adding edge (u,v) only changes the adjacency of u and v.
            #   For an existing edge (a,b) where a != u,v and b != u,v:
            #     N(a) and N(b) don't change, so book(a,b) doesn't change.
            #   For an edge (u, x) where x != v:
            #     N(u) gains v. So book(u,x) = |N(u) ∩ N(x)|.
            #     v is in N(u) now. Is v in N(x)? If x~v, then v is a common neighbor.
            #     So book(u,x) increases by 1 if x~v.
            #   Similarly for edge (v, x) where x != u:
            #     book(v,x) increases by 1 if x~u.
            #   For edge (u,v) itself: book(u,v) = |N(u) ∩ N(v)| as computed.
            #
            # So we need to update:
            #   - For each neighbor x of u (x != v): if x is also neighbor of v, book(u,x) += 1
            #   - For each neighbor x of v (x != u): if x is also neighbor of u, book(v,x) += 1

            # Update books for edges incident to u
            for x in list(adj[u]):
                if x == v:
                    continue
                if x in adj[v]:
                    # v is now a common neighbor of u and x
                    key = (min(u, x), max(u, x))
                    if key in books:
                        books[key] += 1

            # Update books for edges incident to v
            for x in list(adj[v]):
                if x == u:
                    continue
                if x in adj[u]:
                    # u is now a common neighbor of v and x
                    key = (min(v, x), max(v, x))
                    if key in books:
                        books[key] += 1

        return edge_set, adj, books

    def simulated_annealing(edge_set, adj, books, t_start=2.0, t_end=0.01, steps_per_temp=50, cooling=0.95):
        """
        Simulated annealing local search.
        Move: remove one edge, add another (keep m constant).
        Objective: minimize max book size, then minimize sum of book sizes.
        """
        current_max = max_book(books)
        current_sum = sum(books.values())
        best_max = current_max
        best_sum = current_sum
        best_edge_set = set(edge_set)
        best_books = dict(books)
        best_adj = [set(s) for s in adj]

        t = t_start
        total_steps = 0

        while t > t_end and time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                total_steps += 1

                # Pick a random edge to remove
                edge_list = list(edge_set)
                if not edge_list:
                    break
                u, v = random.choice(edge_list)

                # Try to find a good edge to add
                # Candidate: a non-edge that would have small book size
                # Sample some random non-edges
                found_swap = False
                for attempt in range(20):
                    c = random.randrange(n)
                    d = random.randrange(n)
                    if c == d:
                        continue
                    c, d = min(c, d), max(c, d)
                    if (c, d) in edge_set:
                        continue
                    if (c, d) == (u, v):
                        continue

                    # Compute book size of (c,d) after removing (u,v)
                    # After removing (u,v):
                    #   adj[u] loses v, adj[v] loses u
                    #   For (c,d): 
                    #     If c==u: N(c) = adj[u] - {v}
                    #     If c==v: N(c) = adj[v] - {u}
                    #     Else: N(c) = adj[c]
                    #   Similarly for d.

                    if c == u:
                        nc = adj[u] - {v}
                    elif c == v:
                        nc = adj[v] - {u}
                    else:
                        nc = adj[c]

                    if d == u:
                        nd = adj[u] - {v}
                    elif d == v:
                        nd = adj[v] - {u}
                    else:
                        nd = adj[d]

                    new_bk_cd = len(nc & nd)

                    # Estimate the change in max book and sum
                    # Removing (u,v) with book bk_uv:
                    bk_uv = books[(u, v)]

                    # Adding (c,d) with book new_bk_cd
                    # The max book changes if:
                    #   - bk_uv was the max and we remove it
                    #   - new_bk_cd is larger than current max

                    # Also, removing (u,v) and adding (c,d) changes books of other edges.
                    # This is complex, so let's do a full recomputation for acceptance.
                    # But that's expensive. Let's use a heuristic.

                    # Heuristic: accept if new_bk_cd < bk_uv (reduces a large book)
                    # or if new_bk_cd is small.
                    # For SA, we need a cost function. Let's use:
                    # cost = max_book + 0.1 * sum_books
                    # But computing exact cost requires full recompute.

                    # Let's just do full recompute for candidate swaps.
                    # Tentatively apply swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[c].add(d)
                    adj[d].add(c)

                    # Recompute all books (expensive but correct)
                    new_books = {}
                    for eu, ev in edge_set:
                        new_books[(eu, ev)] = len(adj[eu] & adj[ev])
                    new_max = max_book(new_books)
                    new_sum = sum(new_books.values())

                    new_cost = new_max + 0.1 * new_sum
                    old_cost = current_max + 0.1 * current_sum

                    # Simulated annealing acceptance
                    if new_cost < old_cost:
                        # Accept
                        current_max = new_max
                        current_sum = new_sum
                        books = new_books
                        if new_max < best_max or (new_max == best_max and new_sum < best_sum):
                            best_max = new_max
                            best_sum = new_sum
                            best_edge_set = set(edge_set)
                            best_books = dict(new_books)
                            best_adj = [set(s) for s in adj]
                        found_swap = True
                        break
                    else:
                        # Reject with probability exp(-delta/t)
                        delta = new_cost - old_cost
                        if random.random() < math.exp(-delta / max(t, 0.001)):
                            # Accept anyway
                            current_max = new_max
                            current_sum = new_sum
                            books = new_books
                            if new_max < best_max or (new_max == best_max and new_sum < best_sum):
                                best_max = new_max
                                best_sum = new_sum
                                best_edge_set = set(edge_set)
                                best_books = dict(new_books)
                                best_adj = [set(s) for s in adj]
                            found_swap = True
                            break
                        else:
                            # Revert
                            edge_set.discard((c, d))
                            edge_set.add((u, v))
                            adj[u].add(v)
                            adj[v].add(u)
                            adj[c].discard(d)
                            adj[d].discard(c)

                if not found_swap:
                    # No swap accepted, try a different random edge to remove
                    pass

            t *= cooling

        return best_edge_set, best_adj, best_books, best_max

    # Multi-start GRASP + SA
    best_overall_max = float('inf')
    best_overall_edge_set = None
    best_overall_adj = None
    best_overall_books = None

    num_starts = 30
    for start in range(num_starts):
        if time.time() > deadline:
            break

        # Vary randomness parameter
        rp = 0.1 + 0.8 * random.random()
        edge_set, adj, books = grasp_construct(rp)

        if len(edge_set) < m:
            # Fill remaining edges randomly
            all_e = all_edges()
            for e in all_e:
                if len(edge_set) >= m:
                    break
                if e not in edge_set:
                    edge_set.add(e)
                    u, v = e
                    adj[u].add(v)
                    adj[v].add(u)
            books = compute_books_from_adj(adj, edge_set)

        current_max = max_book(books)
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_edge_set = set(edge_set)
            best_overall_adj = [set(s) for s in adj]
            best_overall_books = dict(books)

        # Run SA on this starting point
        if time.time() > deadline - 30:
            # Not enough time for SA, skip
            continue

        sa_edge_set, sa_adj, sa_books, sa_max = simulated_annealing(
            edge_set, adj, books,
            t_start=1.5, t_end=0.05, steps_per_temp=30, cooling=0.93
        )

        if sa_max < best_overall_max:
            best_overall_max = sa_max
            best_overall_edge_set = sa_edge_set
            best_overall_adj = sa_adj
            best_overall_books = sa_books

    # Final local search refinement on best solution
    if best_overall_edge_set is not None and time.time() < deadline - 10:
        # Do a final hill-climbing pass
        edge_set = set(best_overall_edge_set)
        adj = [set(s) for s in best_overall_adj]
        books = dict(best_overall_books)
        current_max = max_book(books)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Sort edges by book size descending
            sorted_edges = sorted(books.items(), key=lambda x: -x[1])
            for (u, v), bk in sorted_edges:
                if time.time() > deadline - 5:
                    break
                if bk < current_max:
                    break
                # Try to replace this edge
                for attempt in range(50):
                    c = random.randrange(n)
                    d = random.randrange(n)
                    if c == d:
                        continue
                    c, d = min(c, d), max(c, d)
                    if (c, d) in edge_set or (c, d) == (u, v):
                        continue

                    # Tentative swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[c].add(d)
                    adj[d].add(c)

                    new_books = {}
                    for eu, ev in edge_set:
                        new_books[(eu, ev)] = len(adj[eu] & adj[ev])
                    new_max = max_book(new_books)

                    if new_max < current_max:
                        books = new_books
                        current_max = new_max
                        improved = True
                        if current_max < best_overall_max:
                            best_overall_max = current_max
                            best_overall_edge_set = set(edge_set)
                            best_overall_adj = [set(s) for s in adj]
                            best_overall_books = dict(books)
                        break
                    else:
                        # Revert
                        edge_set.discard((c, d))
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[c].discard(d)
                        adj[d].discard(c)

    # Return best solution
    if best_overall_edge_set is None:
        # Fallback: random graph
        edge_set = set()
        all_e = all_edges()
        random.shuffle(all_e)
        for e in all_e[:m]:
            edge_set.add(e)
        return [[u, v] for u, v in edge_set]

    return [[u, v] for u, v in best_overall_edge_set]
