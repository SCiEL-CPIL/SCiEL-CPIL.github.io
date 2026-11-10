import random
import time

def construct():
    start = time.time()
    deadline = start + 105.0
    n = 18
    full_mask = (1 << n) - 1

    def is_k4_free(adj):
        for i in range(n):
            ni = adj[i]
            for j in range(i + 1, n):
                if not (ni >> j) & 1:
                    continue
                common = ni & adj[j]
                # need a vertex l > k with j < k < l
                for k in range(j + 1, n):
                    if not (common >> k) & 1:
                        continue
                    # candidate l > k common neighbor of i,j,k
                    cand = common & adj[k]
                    # restrict to l > k
                    cand &= full_mask ^ ((1 << (k + 1)) - 1)
                    if cand:
                        return False
        return True

    def tri_masks(adj):
        masks = []
        for i in range(n):
            ni = adj[i]
            for j in range(i + 1, n):
                if not (ni >> j) & 1:
                    continue
                common = ni & adj[j] & full_mask ^ ((1 << (j + 1)) - 1)
                m = common
                while m:
                    lsb = m & -m
                    k = lsb.bit_length() - 1
                    masks.append((1 << i) | (1 << j) | lsb)
                    m ^= lsb
        return masks

    def has_tf_superset(mask, T):
        # Does any triangle lie entirely outside mask? i.e., exists t with t & mask == 0
        for t in T:
            if t & mask == 0:
                return True
        return False

    def max_tf_size(adj, T):
        if not T:
            return n
        # quick upper bound: min vertex degree complement? use simple n
        # check sizes descending
        from itertools import combinations
        # Precompute for speed
        for s in range(n, 0, -1):
            # iterate all subsets of size s? too many for s large.
            # Instead use complement: check if there exists a set of size s with no triangle.
            # A set S of size s is triangle-free iff no triangle is a subset of S.
            # Equivalent: S contains at most 2 vertices from every triangle.
            # We want the max s. Use the following: S is tf iff S is independent in the 3-uniform
            # hypergraph of triangles. Use branch and bound over vertices.
            pass
        # Branch and bound
        best = [0]
        # order vertices by degree in triangle hypergraph descending
        tcount = [0]*n
        for t in T:
            tt = t
            while tt:
                lsb = tt & -tt
                tcount[lsb.bit_length()-1] += 1
                tt ^= lsb
        order = sorted(range(n), key=lambda v: -tcount[v])

        def dfs(idx, mask, size):
            if time.time() > deadline - 2:
                return
            if size > best[0]:
                best[0] = size
            if idx == n:
                return
            # try including order[idx] if it doesn't complete a triangle with existing
            v = order[idx]
            bit = 1 << v
            # can include v only if no triangle t has t & (mask|bit) == t and v in t and other two in mask
            # i.e., no triangle t with t & mask == t ^ bit  (the other two in mask)
            ok = True
            for t in T:
                if t & bit:
                    if (t & mask) == (t ^ bit):
                        ok = False
                        break
            if ok:
                dfs(idx+1, mask|bit, size+1)
            # exclude
            dfs(idx+1, mask, size)
        dfs(0, 0, 0)
        return best[0]

    def gen_maximal_k4free(seed):
        rnd = random.Random(seed)
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rnd.shuffle(pairs)
        adj = [0]*n
        for (i, j) in pairs:
            # check if adding edge creates K4: need a common neighbor k of both i and j such that
            # k is adjacent to a vertex l that is adjacent to both i and j (l != i,j,k) -> actually K4 = {i,j,k,l}
            # all four pairwise. Adding edge (i,j) creates K4 iff there exist k,l != i,j with
            # k,l both adjacent to i and j, AND k,l adjacent to each other.
            # i.e., common neighbors of i and j (excluding i,j) contain an edge.
            common = (adj[i] & adj[j]) & full_mask ^ ((1<<i)|(1<<j))
            # check if common induces any edge
            has_edge = False
            c = common
            while c:
                lsb = c & -c
                v = lsb.bit_length()-1
                if adj[v] & common:
                    has_edge = True
                    break
                c ^= lsb
            if not has_edge:
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
        return adj

    def edges_from_adj(adj):
        out = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    out.append([i, j])
        return out

    # Turan T_3(18): complete 3-partite, parts size 6
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    turan_edges = [[a,b] for i in range(3) for j in range(i+1,3) for a in parts[i] for b in parts[j]]

    best = 12
    best_edges = turan_edges

    t0 = time.time()
    for seed in range(10**6):
        if time.time() > deadline:
            break
        adj = gen_maximal_k4free(seed)
        # verify K4-free (should hold by construction)
        if not is_k4_free(adj):
            continue
        T = tri_masks(adj)
        s = max_tf_size(adj, T)
        if s < best:
            best = s
            best_edges = edges_from_adj(adj)
            if best <= 9:
                break
    return best_edges

# EVOLVE-BLOCK-START
# construct defined above
# EVOLVE-BLOCK-END
