import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_triple_indices(n):
        return list(combinations(range(n), 3))

    def verify(n, colors):
        """colors: list of 0/1 in lexicographic triple order. Return True if valid."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False
        for quint in combinations(range(n), 5):
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    break
            else:
                return False
        return True

    def partition_search(n, time_limit, seed=0, init_mode=0):
        """
        Per-vertex partition search with fast bitset oracle.
        Returns (colors_list, is_valid).
        """
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # Precompute: for each vertex v, mask of other vertices u such that (min(v,u), max(v,u), w) ... 
        # Actually: red_mask[v] = bitmask of vertices u such that the "pair constraint" is violated.
        # Better: for red K4 detection, we need: for any 4-set {a,b,c,d}, not all 4 triples red.
        # Equivalent: for vertex v, define red_neighbors[v] = {u : triple (min(v,u), max(v,u), w) is red for all w between?}
        # Simpler: precompute for each pair (v,u) the set of w such that (v,u,w) is red.
        # Then a 4-set {a,b,c,d} is all-red iff for each vertex, the other 3 form red triples with it.

        # Most efficient: for each vertex v, build red_mask[v] as bitmask where bit u is set
        # iff triple (v, u, w) is red for ALL w in the other two... no.
        # Let's use: red_mask[v] = bitmask of u such that (min(v,u), max(v,u), x) is red 
        #   for x being each of the other vertices... this doesn't simplify.

        # Correct approach: For red K4 {a,b,c,d}, all 4 triples must be red.
        # For a fixed vertex v, define: red_partner_mask[v] = bitmask of u != v such that
        #   for EVERY w in the set, (v,u,w) is red... no, that's circular.

        # Let me just precompute: for each triple (a,b,c), store its color.
        # For red K4 check on set S={a,b,c,d}: check 4 specific triples.
        # For blue K5 check on set S of size 5: check 10 triples.

        # For the search, I'll maintain:
        # - red_vertices: set of vertices assigned red
        # - blue_vertices: set of vertices assigned blue
        # And use the following oracle:
        #   red_violation: exists 4-set in red_vertices with all 4 triples red
        #   blue_violation: exists 5-set in blue_vertices with all 10 triples blue

        # Fast oracle using bitsets:
        # For each vertex v, precompute red_mask[v] = bitmask of u such that 
        #   the triple (min(v,u), max(v,u), w) is red for all w... 
        # Actually the cleanest: for each vertex v, precompute which pairs (u,w) with u<w, u!=v, w!=v
        # have (min(v,u), max(v,u), min(v,w), max(v,w))... this is getting complicated.

        # Let me use a different fast oracle:
        # Precompute for each pair (i,j), i<j: red_count[i,j] = number of k such that triple (i,j,k) is red.
        # Then a 4-set {a,b,c,d} is all-red iff for each of the 4 pairs, the 3rd vertex makes it red.
        # This still requires checking 4 triples.

        # For practical speed, I'll precompute:
        # red_triple_mask[v] = for vertex v, a bitmask over other vertices where bit u is set
        #   if the triple involving v and u and any third vertex... 
        # 
        # OK, simplest fast approach: precompute for each vertex v, the set of 2-subsets {u,w} 
        # such that (v,u,w) is red. Store as a set of frozensets or a bitmask.
        # Then red K4 exists iff there's a 4-set {a,b,c,d} where for each vertex, the other 3 
        # form a red triple with it.

        # For the search, I'll use incremental tracking:
        # Maintain red_violation_count and blue_violation_count incrementally.

        # Given time constraints, let me use a simpler but still fast approach:
        # Precompute all K4s and K5s as before, but use the vertex-partition representation.

        # Build K4 and K5 structures
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Vertex-to-K4 and vertex-to-K5 mapping
        vertex_to_k4s = [[] for _ in range(n)]
        for i, idxs in enumerate(k4s):
            quad = list(combinations(range(n), 4))[i] if i < len(list(combinations(range(n), 4))) else None
            # This is inefficient; let me rebuild
        # Rebuild properly
        vertex_to_k4s = [[] for _ in range(n)]
        for i, quad in enumerate(combinations(range(n), 4)):
            for v in quad:
                vertex_to_k4s[v].append(i)

        vertex_to_k5s = [[] for _ in range(n)]
        for i, quint in enumerate(combinations(range(n), 5)):
            for v in quint:
                vertex_to_k5s[v].append(i)

        # Initialize vertex colors: 0=red, 1=blue
        np.random.seed(seed)
        if init_mode == 0:
            vertex_colors = np.random.randint(0, 2, n, dtype=np.int8)
        elif init_mode == 1:
            vertex_colors = np.array([v % 2 for v in range(n)], dtype=np.int8)
        elif init_mode == 2:
            vertex_colors = np.array([0 if (v * 7 + 3) % 3 == 0 else 1 for v in range(n)], dtype=np.int8)
        elif init_mode == 3:
            # Start with all red, then randomize
            vertex_colors = np.zeros(n, dtype=np.int8)
            for _ in range(n // 2):
                vertex_colors[np.random.randint(n)] = 1
        else:
            vertex_colors = np.random.randint(0, 2, n, dtype=np.int8)

        def compute_triple_colors(vc):
            """Compute triple colors from vertex colors: triple (a,b,c) is red (0) if 
            all three vertices are red, blue (1) otherwise? No, that's not right.
            
            Actually, in this formulation, we need a mapping from vertex assignments to triple colors.
            The standard approach: a triple is red if all three vertices are in the red set,
            blue otherwise. But that's too restrictive.
            
            Let me reconsider: the problem is about coloring triples directly. The vertex-partition
            approach only works if we define triple color as a function of vertex colors.
            
            A common construction: triple (a,b,c) is red iff all three are in the same "part",
            blue otherwise. With 2 parts, triple is red iff all 3 in part 0 or all 3 in part 1.
            But that gives many red triples and may not work.
            
            Alternative: use a 3-coloring of vertices, triple is red iff all same color.
            With 3 parts, this is the standard construction for R(3,3,3;2) type problems.
            
            For our problem, let me use: assign each vertex a value in {0,1,2}.
            Triple (a,b,c) is red iff a==b==c (all same), blue otherwise.
            But that makes very few red triples.
            
            Actually, the known construction for R(4,5;3) > 34 likely uses a more sophisticated
            algebraic construction. Since I don't know it, let me use the vertex-partition
            approach with a different triple-color rule:
            
            Triple (a,b,c) is red iff the three vertex values are all even, blue otherwise.
            Or: red iff (val[a] + val[b] + val[c]) % 2 == 0.
            
            Let me try: vertex values in {0,1}, triple red iff sum % 2 == 0 (even number of 1s... 
            no, sum%2==0 means 0 or 2 ones).
            
            Actually, for the search to be meaningful, I should just directly search over triple
            colorings but use a smarter local move: flip a "pattern" of triples related to one vertex.
            """
            pass

        # The vertex-partition approach is tricky because the triple colors aren't determined
        # by vertex colors in a simple way for this problem. Let me fall back to a hybrid:
        # Use the original triple-based SA but with much faster delta computation using
        # precomputed counts, and add a "cluster flip" move that flips all triples
        # containing a given vertex.

        # Precompute: for each triple t, list of K4s and K5s it belongs to (already have)
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        # Also: for each vertex v, list of triples containing v
        vertex_to_triples = [[] for _ in range(n)]
        for t_idx, (a, b, c) in enumerate(triples):
            vertex_to_triples[a].append(t_idx)
            vertex_to_triples[b].append(t_idx)
            vertex_to_triples[c].append(t_idx)

        # Initialize triple colors
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)

        def count_violations(col):
            red_count = 0
            for idxs in k4s:
                if all(col[i] == 0 for i in idxs):
                    red_count += 1
            blue_count = 0
            for idxs in k5s:
                if all(col[i] == 1 for i in idxs):
                    blue_count += 1
            return red_count + blue_count

        current_cost = count_violations(colors)
        best_cost = current_cost
        best_colors = colors.copy()

        if current_cost == 0:
            return colors.tolist(), True

        # Precompute for fast delta: for each K4, track how many red triples it has
        k4_red_counts = np.array([sum(1 for i in idxs if colors[i] == 0) for idxs in k4s], dtype=np.int16)
        # For each K5, track how many blue triples it has
        k5_blue_counts = np.array([sum(1 for i in idxs if colors[i] == 1) for idxs in k5s], dtype=np.int16)

        temp = 1.5
        cool_rate = 0.9997
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 3000

        while time.time() < time_limit:
            # Choose move type: 70% single flip, 30% cluster flip (all triples through one vertex)
            if np.random.rand() < 0.7:
                t = np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color
                # Compute delta
                dc = 0
                for k4_i in triple_to_k4s[t]:
                    rc = k4_red_counts[k4_i]
                    if old_color == 0:
                        if rc == 4:
                            dc -= 1
                        k4_red_counts[k4_i] = rc - 1
                    else:
                        if rc == 3:
                            dc += 1
                        k4_red_counts[k4_i] = rc + 1
                for k5_i in triple_to_k5s[t]:
                    bc = k5_blue_counts[k5_i]
                    if old_color == 1:
                        if bc == 10:
                            dc -= 1
                        k5_blue_counts[k5_i] = bc - 1
                    else:
                        if bc == 9:
                            dc += 1
                        k5_blue_counts[k5_i] = bc + 1
                # Tentatively apply
                colors[t] = new_color
                new_cost = current_cost + dc
                if dc > 0 and not (temp > min_temp and np.random.rand() < np.exp(-dc / temp)):
                    # Reject: revert
                    colors[t] = old_color
                    for k4_i in triple_to_k4s[t]:
                        k4_red_counts[k4_i] += (1 if old_color == 0 else -1)
                    for k5_i in triple_to_k5s[t]:
                        k5_blue_counts[k5_i] += (1 if old_color == 1 else -1)
                    no_improve += 1
                    temp *= cool_rate
                    continue
                current_cost = new_cost
            else:
                # Cluster flip: flip all triples through a random vertex
                v = np.random.randint(n)
                affected = vertex_to_triples[v]
                if not affected:
                    temp *= cool_rate
                    continue
                old_colors = colors[affected].copy()
                new_colors = 1 - old_colors
                # Compute total delta
                dc = 0
                for t in affected:
                    oc = old_colors[list(affected).index(t)]
                    nc = 1 - oc
                    for k4_i in triple_to_k4s[t]:
                        rc = k4_red_counts[k4_i]
                        if oc == 0:
                            if rc == 4:
                                dc -= 1
                            k4_red_counts[k4_i] = rc - 1
                        else:
                            if rc == 3:
                                dc += 1
                            k4_red_counts[k4_i] = rc + 1
                    for k5_i in triple_to_k5s[t]:
                        bc = k5_blue_counts[k5_i]
                        if oc == 1:
                            if bc == 10:
                                dc -= 1
                            k5_blue_counts[k5_i] = bc - 1
                        else:
                            if bc == 9:
                                dc += 1
                            k5_blue_counts[k5_i] = bc + 1
                colors[affected] = new_colors
                new_cost = current_cost + dc
                if dc > 0 and not (temp > min_temp and np.random.rand() < np.exp(-dc / temp)):
                    # Reject: revert
                    colors[affected] = old_colors
                    for idx, t in enumerate(affected):
                        oc = old_colors[idx]
                        nc = 1 - oc
                        for k4_i in triple_to_k4s[t]:
                            k4_red_counts[k4_i] += (1 if oc == 0 else -1)
                        for k5_i in triple_to_k5s[t]:
                            k5_blue_counts[k5_i] += (1 if oc == 1 else -1)
                    no_improve += 1
                    temp *= cool_rate
                    continue
                current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                no_improve = 0
                if best_cost == 0:
                    return best_colors.tolist(), True
            else:
                no_improve += 1

            temp *= cool_rate

            if no_improve > max_no_improve and temp > min_temp:
                temp = max(0.5, temp * 1.5)
                no_improve = 0
            if no_improve > max_no_improve * 4:
                colors = best_colors.copy()
                current_cost = best_cost
                # Recompute counts
                k4_red_counts = np.array([sum(1 for i in idxs if colors[i] == 0) for idxs in k4s], dtype=np.int16)
                k5_blue_counts = np.array([sum(1 for i in idxs if colors[i] == 1) for idxs in k5s], dtype=np.int16)
                temp = max(0.8, temp)
                no_improve = 0

        return best_colors.tolist(), (best_cost == 0)

    final_n = 0
    final_colors = []

    target_ns = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break
        if n <= final_n:
            continue
        remaining = deadline - time.time()
        if remaining < 3:
            break

        if n == 35:
            time_budget = min(50.0, remaining * 0.7)
        elif n == 34:
            time_budget = min(45.0, remaining * 0.65)
        elif n >= 30:
            time_budget = min(20.0, remaining * 0.5)
        elif n >= 25:
            time_budget = min(15.0, remaining * 0.4)
        elif n >= 20:
            time_budget = min(10.0, remaining * 0.3)
        else:
            time_budget = min(5.0, remaining * 0.2)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget
        found = False
        num_seeds = 30 if n >= 35 else 20

        for seed in range(num_seeds):
            if time.time() > time_limit:
                break
            init_mode = seed % 4
            try:
                colors, is_valid = partition_search(n, time_limit, seed=seed, init_mode=init_mode)
                if is_valid:
                    if verify(n, colors):
                        final_n = n
                        final_colors = colors
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    if final_n == 0 or not final_colors:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            np.random.seed(42)
            for _ in range(1000):
                colors = [np.random.randint(0, 2) for _ in range(len(triples))]
                if verify(n, colors):
                    final_n = n
                    final_colors = colors
                    break
            else:
                final_n = 1
                final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
