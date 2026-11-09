"""Stronger algorithm: greedy C4-avoiding 5-coloring of K_n, maximizing n."""
import time
import random
import numpy as np

def _c4_violation_count(adj_mats, color):
    """Count C4s in one colour class via matrix product."""
    A = adj_mats[color]
    # Number of length-2 paths between i and j = (A^2)[i,j]
    # A C4 exists iff some pair has >=2 common neighbours
    A2 = A @ A
    # diagonal entries are degrees; we care about off-diagonal
    off = A2 - np.diag(np.diag(A2))
    return int(np.sum(off >= 2))

def _validate(coloring, n):
    """Return True if no colour class contains a C4."""
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = coloring[idx]
            idx += 1
            adj[c][i, j] = 1
            adj[c][j, i] = 1
    for c in range(5):
        A2 = adj[c] @ adj[c]
        off = A2 - np.diag(np.diag(A2))
        if np.any(off >= 2):
            return False
    return True

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    # Best known: Lazebnik-Woldar gives 26. Start with seed 11 and try to grow.
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]

    def try_greedy(n, seed, time_limit):
        """Greedy: assign colours to edges in some order, avoiding C4s."""
        t0 = time.time()
        random.seed(seed)
        coloring = [0] * (n * (n - 1) // 2)
        adj = [set() for _ in range(5)]  # adjacency lists per colour
        # common-neighbour counts per colour per pair
        common = [{} for _ in range(5)]  # common[c][(i,j)] = count of common neighbours in colour c

        # We process edges in a random order but track which pairs become "at risk"
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)

        edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
        # Build mapping from (i,j) to index in coloring
        def eidx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

        # For each colour, maintain adjacency as bitmasks or sets
        adj_sets = [set() for _ in range(5)]  # adj_sets[c] is set of (i,j)

        # Track common neighbours: for each colour c, for each unordered pair {u,v},
        # how many common neighbours they have in colour c.
        # A C4 in colour c occurs when some pair {u,v} has >= 2 common neighbours in colour c.
        common_counts = [{} for _ in range(5)]  # key: frozenset({u,v}) or tuple (min,max)

        def get_common(c, u, v):
            key = (min(u, v), max(u, v))
            return common_counts[c].get(key, 0)

        def set_common(c, u, v, val):
            key = (min(u, v), max(u, v))
            if val == 0:
                common_counts[c].pop(key, None)
            else:
                common_counts[c][key] = val

        for edge in edges:
            if time.time() - t0 > time_limit:
                break
            i, j = edge
            # Try each colour; pick one that doesn't create a C4
            valid_colors = []
            for c in range(5):
                # Adding edge (i,j) in colour c:
                # For every common neighbour k of i and j in colour c,
                # the pair (k, ?) ... actually:
                # A C4 is formed if there exists a vertex k such that
                #   (k,i) in colour c AND (k,j) in colour c
                # because then k and the other endpoint... wait.
                # C4: vertices a,b,c,d with edges (a,b),(b,c),(c,d),(d,a) all in same colour.
                # If we add (i,j) in colour c, a C4 is created iff there exist vertices p, q
                # such that (p,i),(q,i) in colour c AND (p,j),(q,j) in colour c? No.
                # Actually: adding (i,j) creates a C4 in colour c iff there is a vertex k
                # such that (k,i) and (k,j) are both in colour c (then k is a common neighbour),
                # AND another vertex... no wait.
                # A C4 containing the new edge (i,j) would be i - j - x - y - i where
                # (j,x), (x,y), (y,i) are in colour c. So x must be a common neighbour of i and j? No:
                # (j,x) in c and (y,i) in c, and (x,y) in c. So x is adjacent to j, y adjacent to i,
                # and x adjacent to y. This is getting complex.
                #
                # Simpler: after adding (i,j) in colour c, check if any pair has >=2 common neighbours.
                # The only pairs whose common-neighbor count changes are pairs involving i or j.
                # For a vertex k: common count of (i,k) increases by 1 if (j,k) in colour c.
                # For a vertex k: common count of (j,k) increases by 1 if (i,k) in colour c.
                # So we need to check: for all k in neighbors_c(i) ∩ neighbors_c(j),
                #   does common(c, i, k) become 2? Or common(c, j, k) become 2?
                # Actually if k is a common neighbour, then adding (i,j) makes:
                #   common(c,i,k) += 1 (because j is now adjacent to i and k to i... wait no)
                # Let me think again.
                # common(c, u, v) = |N_c(u) ∩ N_c(v)|
                # Adding edge (i,j) to colour c:
                #   For pair (i, k): N_c(i) gains j. So common(c,i,k) gains 1 iff j ∈ N_c(k), i.e., (j,k) in colour c.
                #   For pair (j, k): N_c(j) gains i. So common(c,j,k) gains 1 iff (i,k) in colour c.
                #   For pair (k, l) where k≠i,j and l≠i,j: no change.
                # So we need to check all k where (j,k) in colour c: common(c,i,k) will increase.
                #   If common(c,i,k) was 1, it becomes 2 → C4.
                # Similarly for k where (i,k) in colour c: common(c,j,k) increases.
                #
                # Let's compute:
                ok = True
                # Check pairs (i,k) for k in N_c(j)
                for k in adj_sets[c]:
                    # k is stored as (a,b) tuples
                    pass

                # This set-based approach is getting messy. Let me use a different representation.
                break

            if valid_colors:
                # Pick the colour that creates the fewest new "at-risk" pairs (those reaching 1 common neighbour)
                # Cost-minimization greedy
                best_c = min(valid_colors, key=lambda c: 0)  # simple for now
                coloring[eidx(i, j)] = best_c
                for k in adj_sets[best_c]:
                    pass

        return coloring, best_n

    # Simpler approach: use the known structure. Start from n=11 seed and try to extend.
    # Actually, let's just do a straightforward greedy with proper C4 checking using bitmasks.

    def greedy_color(n, seed):
        t0 = time.time()
        rng = random.Random(seed)
        coloring = [-1] * (n * (n - 1) // 2)

        def eidx(i, j):
            if i > j:
                i, j = j, i
            # index of (i,j) in lexicographic list
            # pairs before row i: i*(2n-i-1)/2 ... let's just use a lookup
            return edge_lookup[(i, j)]

        edge_lookup = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_lookup[(i, j)] = idx
                idx += 1

        # Adjacency bitmasks per colour
        adj = [0] * (5 * n)  # adj[c*n + i] is bitmask of neighbours of i in colour c

        # Common neighbour count tracking: for each colour, for each pair, count
        # Use a dict: common[c][(u,v)] = count
        common = [dict() for _ in range(5)]

        def get_common(c, u, v):
            if u > v:
                u, v = v, u
            return common[c].get((u, v), 0)

        def inc_common(c, u, v):
            if u > v:
                u, v = v, u
            common[c][(u, v)] = common[c].get((u, v), 0) + 1

        def check_valid(c, i, j):
            # Can we add edge (i,j) in colour c?
            # For each k in N_c(i): if k in N_c(j), then common(c,i,j) would increase... 
            # Wait, we're adding (i,j). The affected common counts are:
            # For k in N_c(i): pair (k, j) - common(c,k,j) increases by 1 if i in N_c(k) i.e. (k,i) in c. Since (i,k) in c means i in N_c(k).
            #   So for each k in N_c(i) ∩ N_c(j): common(c,i,k) and common(c,j,k) both... 
            # Let me just check: after adding (i,j) in colour c, does any pair have >=2 common neighbours?
            # Only pairs involving i or j change.
            # For k: common(c, i, k) increases by 1 iff j in N_c(k) iff (j,k) in colour c.
            #   So for k in N_c(i) ∩ N_c(j): common(c,i,k) += 1. Check if it was 1.
            # For k: common(c, j, k) increases by 1 iff (i,k) in colour c.
            #   So for k in N_c(j) ∩ N_c(i): common(c,j,k) += 1. Check if it was 1.
            ni = adj[c * n + i]
            nj = adj[c * n + j]
            # Common neighbours of i and j in colour c
            common_nbs = ni & nj
            if common_nbs:
                # For each such k, check common(c, i, k) and common(c, j, k)
                k = 0
                mask = common_nbs
                while mask:
                    if mask & 1:
                        if get_common(c, i, k) >= 1:
                            return False
                        if get_common(c, j, k) >= 1:
                            return False
                    k += 1
                    mask >>= 1
            # Also check: for k in N_c(i) not in N_c(j), common(c, j, k) increases? No.
            # For k in N_c(j) not in N_c(i), common(c, i, k) increases? No.
            # Wait I think I have it backwards. Let me re-derive.
            # common(c, u, v) = |N_c(u) ∩ N_c(v)|
            # Adding (i,j) to colour c means: N_c(i) gains j, N_c(j) gains i.
            # For pair (i, k): N_c(i) gains j. So common(c,i,k) = |N_c(i) ∩ N_c(k)|. 
            #   j ∈ N_c(i) now. Is j ∈ N_c(k)? That's (j,k) in colour c.
            #   So common(c,i,k) increases by 1 iff (j,k) in colour c, i.e., k ∈ N_c(j).
            #   So for k ∈ N_c(i) ∩ N_c(j): common(c,i,k) += 1.
            # For pair (j, k): similarly, common(c,j,k) += 1 iff k ∈ N_c(i) ∩ N_c(j).
            # So for each k in N_c(i) ∩ N_c(j):
            #   if common(c,i,k) was 1, now 2 → C4
            #   if common(c,j,k) was 1, now 2 → C4
            # This is what I had. Good.
            # But also: what about the pair (i,j) itself? common(c,i,j) = |N_c(i) ∩ N_c(j)| = popcount(common_nbs).
            # That's fine, it can be anything, no constraint on it directly.
            return True

        # Process edges in random order
        edges = list(edge_lookup.keys())
        rng.shuffle(edges)

        for (i, j) in edges:
            if time.time() - t0 > 10:
                break
            valid = []
            for c in range(5):
                if check_valid(c, i, j):
                    valid.append(c)
            if not valid:
                break  # stuck
            # Pick colour with minimum cost: number of pairs that will reach 1 common neighbour
            def cost(c):
                ni = adj[c * n + i]
                nj = adj[c * n + j]
                common_nbs = ni & nj
                cnt = 0
                k = 0
                mask = common_nbs
                while mask:
                    if mask & 1:
                        if get_common(c, i, k) == 0:
                            cnt += 1
                        if get_common(c, j, k) == 0:
                            cnt += 1
                    k += 1
                    mask >>= 1
                return cnt
            c = min(valid, key=cost)
            coloring[eidx(i, j)] = c
            # Update adj
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            # Update common counts
            ni = adj[c * n + i] & ~(1 << j)  # N_c(i) before adding j
            nj = adj[c * n + j] & ~(1 << i)  # N_c(j) before adding i
            common_nbs = ni & nj
            k = 0
            mask = common_nbs
            while mask:
                if mask & 1:
                    inc_common(c, i, k)
                    inc_common(c, j, k)
                k += 1
                mask >>= 1

        # Check if we colored all edges
        if -1 in coloring:
            return None
        # Final validation
        return coloring

    # Try multiple sizes and seeds
    for target_n in range(12, 27):
        if time.time() > deadline:
            break
        for seed in range(20):
            if time.time() > deadline:
                break
            result = greedy_color(target_n, seed)
            if result is not None:
                # Validate
                if _validate(result, target_n):
                    best_n = target_n
                    best_colors = result
                    break  # found valid for this n, try larger
        # If we got a valid coloring for target_n, continue to next
        if best_n >= target_n:
            continue
        else:
            # Could not achieve target_n, stop
            break

    # Final check of best
    if best_n > 11:
        assert _validate(best_colors, best_n)

    return {'n': best_n, 'colors': best_colors}
