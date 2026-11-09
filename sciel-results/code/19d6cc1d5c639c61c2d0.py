"""Rewritten program: structural change to a full-graph constraint-satisfaction model.

Instead of restricting to a bipartite subgraph, we treat the complete graph K_20 as the
domain.  Each edge (u,v) has a colour in 0..19 or is absent.  The constraint is that
every 4-cycle (a,b,c,d) with all four edges present must have four distinct colours.

We precompute all 4-cycles in K_20 (there are C(20,4)*3 = 4845*3 = 14535 of them, but
many share edges).  For each edge we store the list of cycle-indices it participates in.

The search is a greedy-addition heuristic with local repair:
  1. Start with an empty graph.
  2. In a random order of edges, try to add each edge with a colour that keeps all
     4-cycles rainbow.  We maintain a "conflict" counter per cycle: a cycle is "bad"
     if it has 4 present edges with a colour repetition.
  3. After the greedy pass, run a local-search phase that tries to (a) add missing
     edges and (b) recolour existing edges to free up capacity for more edges.

To make this fast we use numpy for vectorised cycle checks.  The key data structures:
  - M: a 20x20 int array, M[u,v] = colour or -1.  Only upper triangle used.
  - For each of the 14535 cycles, we store the 4 edge positions as indices into a
    flat 190-element array (the upper-triangle edges).

We precompute:
  - edge_index: dict from (u,v) to flat index 0..189
  - cycle_edges: array of shape (14535, 4) of flat edge indices

Then a vectorised check:
  - cols = M_flat[cycle_edges]  (shape 14535 x 4)
  - present = (cols != -1).all(axis=1)
  - distinct = all pairwise comparisons
  - bad = present & ~distinct

This is O(14535*4) per full check, which is fast in numpy.

For incremental updates when adding/removing one edge, we only need to check the
cycles containing that edge.  Each edge in K_20 is in C(18,2) = 153 four-cycles
(the other two vertices are chosen from the remaining 18).  So per-edge update is
O(153*4) which is very fast.

We'll use the incremental approach for speed.
"""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    n = 20
    num_colors = 20

    # Precompute all edges in upper triangle
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
            edge_index[(u, v)] = idx
            idx += 1
    total_edges = len(edges)  # 190

    # Precompute all 4-cycles
    # A 4-cycle is (a, b, c, d) with a < b, and we enumerate all 4 distinct vertices
    # For each set of 4 vertices {a,b,c,d} with a<b<c<d, there are 3 distinct 4-cycles:
    #   (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # But we need to be careful: a 4-cycle is defined by 4 edges.
    # The 3 cycles on vertices {a,b,c,d} are:
    #   a-b-c-d-a: edges (a,b),(b,c),(c,d),(d,a)
    #   a-b-d-c-a: edges (a,b),(b,d),(d,c),(c,a)
    #   a-c-b-d-a: edges (a,c),(c,b),(b,d),(d,a)

    cycles = []  # list of tuples of 4 edge indices
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index[(a, b)], edge_index[(b, c)], edge_index[(c, d)], edge_index[(d, a)])
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index[(a, b)], edge_index[(b, d)], edge_index[(d, c)], edge_index[(c, a)])
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index[(a, c)], edge_index[(c, b)], edge_index[(b, d)], edge_index[(d, a)])
                    cycles.append(e3)

    num_cycles = len(cycles)
    cycle_edges_arr = np.array(cycles, dtype=np.int32)  # shape (num_cycles, 4)

    # For each edge, which cycles contain it
    edge_to_cycles = defaultdict(list)
    for ci in range(num_cycles):
        for ei in cycle_edges_arr[ci]:
            edge_to_cycles[ei].append(ci)

    # Convert to numpy arrays
    edge_to_cycles_arr = {}
    for ei, cl in edge_to_cycles.items():
        edge_to_cycles_arr[ei] = np.array(cl, dtype=np.int32)

    # M: flat array of length total_edges, -1 means absent
    def count_bad(M):
        """Count bad cycles. M is flat array of length total_edges."""
        cols = M[cycle_edges_arr]  # shape (num_cycles, 4)
        present = (cols != -1).all(axis=1)
        c0, c1, c2, c3 = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        bad = present & ~distinct
        return np.sum(bad), bad

    def check_edge_bad(M, ei):
        """Check if adding edge ei (with M[ei] set) creates any bad cycle.
        Returns number of bad cycles involving edge ei."""
        if ei not in edge_to_cycles_arr:
            return 0
        cys = edge_to_cycles_arr[ei]
        cols = M[cycle_edges_arr[cys]]  # shape (len(cys), 4)
        present = (cols != -1).all(axis=1)
        c0, c1, c2, c3 = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        bad = present & ~distinct
        return np.sum(bad)

    def try_add_edge(M, ei, colors):
        """Try to add edge ei with one of the given colors. Returns True if successful."""
        old = M[ei]
        for c in colors:
            M[ei] = c
            if check_edge_bad(M, ei) == 0:
                return True
        M[ei] = old
        return False

    # Greedy construction
    best_M = None
    best_count = 0

    random.seed(42)

    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 10.0:
            break

        M = np.full(total_edges, -1, dtype=np.int32)

        # Random order of edges
        order = list(range(total_edges))
        random.shuffle(order)

        for ei in order:
            if time.time() > deadline - 10.0:
                break
            colors = list(range(num_colors))
            random.shuffle(colors)
            try_add_edge(M, ei, colors)

        edge_count = np.sum(M != -1)
        bad_count, _ = count_bad(M)

        if bad_count == 0 and edge_count > best_count:
            best_count = int(edge_count)
            best_M = M.copy()

    # Local search: try to add more edges to the best solution
    if best_M is not None:
        M = best_M.copy()
        current_count = int(np.sum(M != -1))

        # Get missing edges in random order
        missing = [i for i in range(total_edges) if M[i] == -1]
        random.shuffle(missing)

        for ei in missing:
            if time.time() > deadline - 5.0:
                break
            colors = list(range(num_colors))
            random.shuffle(colors)
            if try_add_edge(M, ei, colors):
                current_count += 1

        if current_count > best_count:
            best_count = current_count
            best_M = M.copy()

    # Second phase: try to improve by recolouring
    if best_M is not None:
        M = best_M.copy()
        current_count = int(np.sum(M != -1))
        missing = [i for i in range(total_edges) if M[i] == -1]
        random.shuffle(missing)

        # Try multiple times to add edges, possibly after random recolouring
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break

            # Try to add a random missing edge
            if missing:
                ei = random.choice(missing)
                colors = list(range(num_colors))
                random.shuffle(colors)
                if try_add_edge(M, ei, colors):
                    current_count += 1
                    missing.remove(ei)
                    if current_count > best_count:
                        best_count = current_count
                        best_M = M.copy()
                else:
                    # Try recolouring a random existing edge to free up a colour
                    existing = [i for i in range(total_edges) if M[i] != -1]
                    if existing:
                        ei2 = random.choice(existing)
                        old_c = M[ei2]
                        new_c = random.randint(0, num_colors - 1)
                        if new_c != old_c:
                            M[ei2] = new_c
                            bad = check_edge_bad(M, ei2)
                            if bad > 0:
                                M[ei2] = old_c  # revert
                            else:
                                # Try again to add the missing edge
                                if missing:
                                    ei3 = random.choice(missing)
                                    colors2 = list(range(num_colors))
                                    random.shuffle(colors2)
                                    if try_add_edge(M, ei3, colors2):
                                        current_count += 1
                                        missing.remove(ei3)
                                        if current_count > best_count:
                                            best_count = current_count
                                            best_M = M.copy()

    # Fallback: if we have no valid solution, return a star
    if best_M is None:
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result

    # Convert M to edge list
    result = []
    for ei in range(total_edges):
        if M[ei] != -1:
            u, v = edges[ei]
            c = int(M[ei])
            result.append([u, v, c])

    return result
