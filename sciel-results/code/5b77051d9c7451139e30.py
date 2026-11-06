import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_vertices = list(range(n))

    def is_valid_addition(adj, pair_mask, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free using exact triple constraint."""
        if adj[u] & (1 << v):
            return False

        mask_u = adj[u]
        mask_v = adj[v]
        mask_u |= (1 << v)
        mask_v |= (1 << u)

        # Check triples {u, a, b} where v is common neighbor of a,b
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << v):
                ma = adj[a]
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << v):
                        if (mask_u & ma & adj[b]).bit_count() > 1:
                            return False

        # Check triples {v, a, b} where u is common neighbor of a,b
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << u):
                ma = adj[a]
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << u):
                        if (mask_v & ma & adj[b]).bit_count() > 1:
                            return False

        return True

    def compute_pair_masks(adj):
        """Precompute pair common-neighbor masks for all pairs."""
        pm = {}
        for i in range(n):
            for j in range(i + 1, n):
                pm[(i, j)] = adj[i] & adj[j]
        return pm

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(seed):
        random.seed(seed)
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, None, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, seed):
        """Try to add any valid edges, then try edge swaps."""
        random.seed(seed)
        cur = adj[:]
        edges = count_edges(cur)
        pm = compute_pair_masks(cur)

        # Phase 1: Add any valid edges
        missing = [(i, j) for i, j in all_pairs if not (cur[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if is_valid_addition(cur, pm, u, v):
                cur[u] |= (1 << v)
                cur[v] |= (1 << u)
                edges += 1
                pm[(u, v)] = cur[u] & cur[v]
                for w in range(n):
                    if w != u and w != v:
                        if w > u:
                            pm[(u, w)] = cur[u] & cur[w]
                        else:
                            pm[(w, u)] = cur[u] & cur[w]
                        if w > v:
                            pm[(v, w)] = cur[v] & cur[w]
                        else:
                            pm[(w, v)] = cur[v] & cur[w]

        # Phase 2: Edge swaps - remove one edge, try to add two
        if time.time() > deadline - 5:
            return cur, edges

        existing = [(i, j) for i, j in all_pairs if cur[i] & (1 << j)]
        random.shuffle(existing)
        for ru, rv in existing[:80]:  # Limit swaps tried
            if time.time() > deadline - 3:
                break
            # Remove edge (ru, rv)
            cur[ru] &= ~(1 << rv)
            cur[rv] &= ~(1 << ru)
            edges -= 1

            # Try to add two edges
            missing_after = [(i, j) for i, j in all_pairs
                           if not (cur[i] & (1 << j)) and (i, j) != (ru, rv) and (j, i) != (ru, rv)]
            random.shuffle(missing_after)
            added = 0
            added_edges = []
            for u, v in missing_after:
                if added >= 2:
                    break
                if is_valid_addition(cur, None, u, v):
                    cur[u] |= (1 << v)
                    cur[v] |= (1 << u)
                    edges += 1
                    added += 1
                    added_edges.append((u, v))

            if added < 2:
                # Revert: remove added edges, restore (ru, rv)
                for u, v in added_edges:
                    cur[u] &= ~(1 << v)
                    cur[v] &= ~(1 << u)
                    edges -= 1
                cur[ru] |= (1 << rv)
                cur[rv] |= (1 << ru)
                edges += 1

        return cur, edges

    best_adj = [0] * n
    best_edges = 0

    # Run multiple restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break

        adj = greedy_construct(i)
        adj, edges = local_search(adj, i + 10000)

        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final polish on best solution
    if time.time() < deadline - 5:
        adj, edges = local_search(best_adj, 99999)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj

    # Convert to edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])

    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
