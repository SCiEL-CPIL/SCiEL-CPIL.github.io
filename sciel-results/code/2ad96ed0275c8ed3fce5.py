"""Dense odd-order graphs avoiding equal-degree path endpoints."""
import time
import random
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 82.0

    # Precompute all edges
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105

    def is_valid(adj, deg):
        """Check: for every edge (u,v), for s in N(u)\{v}, t in N(v)\{u}, s!=t, deg[s]!=deg[t]."""
        for (u, v) in edge_list:
            if v not in adj[u]:
                continue
            ns = adj[u] - {v}
            nt = adj[v] - {u}
            if not ns or not nt:
                continue
            # Check: no degree value appears in both ns and nt (with s!=t)
            # Union of ns and nt: each degree can appear at most once total
            seen_deg = set()
            for s in ns:
                d = deg[s]
                if d in seen_deg:
                    return False
                seen_deg.add(d)
            for t in nt:
                d = deg[t]
                if d in seen_deg:
                    return False
                seen_deg.add(d)
        return True

    def make_adj_deg(mask):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for i in range(E):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return adj, deg

    def valid_mask(mask):
        adj, deg = make_adj_deg(mask)
        return is_valid(adj, deg)

    # Strategy: Find densest graph with all-distinct degrees
    # Try all possible degree sequences (permutations of distinct values)
    # that are graphical, and find the densest valid one.

    # First, let's try a smart construction:
    # Assign degrees d_0 > d_1 > ... > d_14 (distinct, from 0 to 14)
    # Use Havel-Hakimi to check graphicality, then construct the graph.

    def havel_hakimi(seq):
        """Check if degree sequence is graphical. Returns (bool, edges) if graphical."""
        seq = sorted(seq, reverse=True)
        if seq[0] > len(seq) - 1:
            return False, []
        edges = []
        def hh(s):
            s = sorted(s, reverse=True)
            if not s or s[0] == 0:
                return []
            if s[0] > len(s) - 1:
                return None
            d = s[0]
            rest = s[1:]
            if d > len(rest):
                return None
            new_rest = []
            for i in range(d):
                if rest[i] <= 0:
                    return None
                new_rest.append(rest[i] - 1)
            for i in range(d, len(rest)):
                new_rest.append(rest[i])
            return new_rest
        # Actually, let's just check graphicality properly
        def is_graphical(s):
            s = sorted(s, reverse=True)
            if not s:
                return True
            if s[0] < 0 or s[0] > len(s) - 1:
                return False
            d = s[0]
            rest = s[1:]
            if d > len(rest):
                return False
            new_rest = [rest[i] - 1 for i in range(d)] + [rest[i] for i in range(d, len(rest))]
            if any(x < 0 for x in new_rest):
                return False
            return is_graphical(new_rest)
        return is_graphical(seq), []

    # For each permutation of a subset of {0,1,...,14} of size 15 (i.e., a permutation of 0..14),
    # check if graphical. But 15! is too large.
    # Instead, try: the densest graphical sequence with distinct degrees.
    # Known: the sequence (14,13,12,...,0) is not graphical.
    # Let's try (13,12,11,...,0): sum = 91. Is it graphical?
    # Actually, let's just try to construct a graph with distinct degrees directly.

    # Alternative: use a threshold graph or a specific construction.
    # Threshold graphs have distinct degrees if constructed properly.

    def threshold_graph_from_seq(seq):
        """seq[i] = 1 if vertex i is dominating, 0 if isolated."""
        adj = [set() for _ in range(n)]
        for i in range(n):
            if seq[i] == 1:
                for j in range(i):
                    adj[i].add(j)
                    adj[j].add(i)
        mask = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    mask |= (1 << edge_to_idx[(i, j)])
        return mask, adj

    # Try all threshold sequences and find the densest valid one
    best_mask = 0
    best_count = 0

    # Try threshold sequences with k zeros for k=1..6
    for k in range(1, 7):
        if time.time() > deadline - 15:
            break
        for zeros in combinations(range(n), k):
            seq = [1] * n
            for z in zeros:
                seq[z] = 0
            mask, adj = threshold_graph_from_seq(seq)
            cnt = bin(mask).count('1')
            if cnt <= best_count:
                continue
            deg = [len(adj[i]) for i in range(n)]
            if is_valid(adj, deg):
                best_mask = mask
                best_count = cnt

    # Now try to improve by adding edges
    def try_add_edges(mask, count):
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            changed = False
            for i in range(E):
                if mask & (1 << i):
                    continue
                new_mask = mask | (1 << i)
                if valid_mask(new_mask):
                    mask = new_mask
                    count += 1
                    changed = True
            if not changed:
                break
        return mask, count

    best_mask, best_count = try_add_edges(best_mask, best_count)

    # Local search with swaps
    rng = random.Random(42)
    for iteration in range(3000):
        if time.time() > deadline - 5:
            break
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]
        if not present or not absent:
            break
        improved = False
        for i in absent:
            new_mask = best_mask | (1 << i)
            if valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                improved = True
                break
        if improved:
            continue
        if len(absent) >= 2 and len(present) >= 1:
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                ri = rng.choice(present)
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                if a1 == a2:
                    continue
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                if valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break

    best_mask, best_count = try_add_edges(best_mask, best_count)

    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
