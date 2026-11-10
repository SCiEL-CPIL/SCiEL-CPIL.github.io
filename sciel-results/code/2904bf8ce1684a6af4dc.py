import random
import time

N = 20
ALL = (1 << N) - 1

def _exact_k33_free(adj):
    # adj: list of int bitmasks of neighbors.
    # A K3,3 exists iff there is a set S of 3 vertices whose common-neighbor
    # set T has >=3 vertices, and those T vertices are not all adjacent to all S... 
    # Actually a K3,3 is a bipartite subgraph with parts A (3) and B (3) where
    # every a in A is adjacent to every b in B. It is NOT required that A and B
    # be independent sets. So we just need: exists A (|A|=3), exists B (|B|=3),
    # such that for all a in A, b in B: a~b.
    # Equivalent: there exist 3 vertices A such that their common-neighborhood
    # (intersection of their neighbor sets) contains 3 vertices B, AND those 3
    # B vertices are all in the common neighborhood (trivially true). But we must
    # ensure A and B are disjoint? In K3,3 the two parts are disjoint. So B must
    # be disjoint from A.
    # So: exists A (size 3) such that (common_neigh(A) & ~mask(A)) has size >= 3.
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                common = adj[i] & adj[j] & adj[k]
                mask_ijk = (1<<i) | (1<<j) | (1<<k)
                if (common & ~mask_ijk).bit_count() >= 3:
                    return False
    return True

def _is_k33_free_fast(adj):
    # Fast check using bitmasks; returns True if no K3,3.
    for i in range(N):
        ai = adj[i]
        for j in range(i+1, N):
            common_ij = ai & adj[j]
            # For any k not in {i,j}, if adj[k] contains at least 3 common neighbors of i,j
            # that are also not in {i,j,k}, then K3,3.
            # We need 3 vertices B all adjacent to i,j,k and disjoint from {i,j,k}.
            # B subset of common_ij & ~({i,j,k}).
            for k in range(j+1, N):
                common = common_ij & adj[k]
                mask = (1<<i)|(1<<j)|(1<<k)
                if (common & ~mask).bit_count() >= 3:
                    return False
    return True

def construct():
    deadline = time.time() + 110.0
    best_edges = None
    best_count = -1
    rng = random.Random(12345)

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def try_greedy(order, adj, adj_set):
        # Add edges in given order if it keeps K3,3-free.
        for (u, v) in order:
            if time.time() > deadline:
                return adj, adj_set
            if adj[u] & (1 << v):
                continue
            # Tentatively add.
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj_set.add((min(u,v), max(u,v)))
            if not _is_k33_free_fast(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj_set.discard((min(u,v), max(u,v)))
        return adj, adj_set

    def local_improve(adj, adj_set):
        # Try to add any missing edge that keeps K3,3-free; repeat until stuck.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(N):
                for v in range(u+1, N):
                    if adj[u] & (1 << v):
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj_set.add((u, v))
                    if _is_k33_free_fast(adj):
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj_set.discard((u, v))
        return adj, adj_set

    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]

    for restart in range(200):
        if time.time() > deadline - 5:
            break
        # Random permutation of edges for greedy.
        order = all_pairs[:]
        rng.shuffle(order)
        adj = [0] * N
        adj_set = set()
        adj, adj_set = try_greedy(order, adj, adj_set)
        adj, adj_set = local_improve(adj, adj_set)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = sorted([list(e) for e in adj_set])
            # Early stop if we hit a strong bound.
            if best_count >= 170:
                break

    if best_edges is None:
        # Fallback: K2,18.
        best_edges = [[a, b] for a in range(2) for b in range(2, 20)]
    return best_edges

if __name__ == "__main__":
    e = construct()
    print(len(e))
