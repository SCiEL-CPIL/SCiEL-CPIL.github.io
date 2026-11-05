import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish by 140s to be safe.
    deadline = start_time + 140.0

    n = 20
    # Precompute all pairs of vertices
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    # Precompute bit masks for each vertex
    bit = [1 << i for i in range(n)]
    full_mask = (1 << n) - 1

    # Helper: Convert adjacency list of ints to list of edges
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Helper: Count edges
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # Helper: Find a K3,3 violation (triple with >= 3 common neighbors)
    def find_violation(adj):
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k)
        return None

    # Helper: Check if adding edge (u, v) preserves K3,3-freeness
    # This is the critical performance bottleneck.
    def can_add_edge(adj, u, v):
        if adj[u] & bit[v]:
            return True
        
        old_u = adj[u]
        old_v = adj[v]
        
        # 1. Check triples involving u and v: (u, v, w)
        # Common neighbors of u and v are (old_u & old_v).
        # After adding edge, N(u) gains v, N(v) gains u.
        # Common neighbors of {u,v,w} = N(u) & N(v) & N(w).
        # Since u not in N(u) and v not in N(v), the set of common neighbors of u and v 
        # remains (old_u & old_v) intersected with N(w).
        # Wait, strictly:
        # N(u) = old_u U {v}
        # N(v) = old_v U {u}
        # N(u) & N(v) = (old_u & old_v) U (old_u & {u}) U ({v} & old_v) U ({v} & {u})
        # old_u & {u} is empty (no self-loops). {v} & old_v is empty (no self-loops).
        # So N(u) & N(v) = old_u & old_v.
        # Thus, for any w, |N(u) & N(v) & N(w)| = |(old_u & old_v) & N(w)|.
        # This is unchanged by adding the edge (u,v). So we don't need to check this if it was valid before.
        # However, we are adding an edge to a *valid* graph, so this condition is already satisfied.
        
        # 2. Check triples involving u but not v: (u, w1, w2)
        # N(u) gains v.
        # New common neighbors = (old_u & N(w1) & N(w2)) U ({v} & N(w1) & N(w2))
        # The term ({v} & N(w1) & N(w2)) is {v} if v is a common neighbor of w1 and w2, else empty.
        # So we need: |(old_u & N(w1) & N(w2))| + (1 if v in N(w1) & N(w2) else 0) < 3
        # i.e., if v in N(w1) & N(w2), then |(old_u & N(w1) & N(w2))| must be < 2.
        
        # 3. Check triples involving v but not u: (v, w1, w2)
        # Symmetric to case 2.
        
        # Optimization: Iterate over w1, w2 that are neighbors of v (for case 2)
        # and neighbors of u (for case 3).
        
        # Case 2: v is added to N(u).
        # We need to check pairs (w1, w2) such that v is a common neighbor.
        # i.e., w1 is neighbor of v, w2 is neighbor of v.
        # Let candidates_v = [w for w in range(n) if w != u and w != v and (adj[v] & bit[w])]
        # For each pair (w1, w2) in candidates_v:
        #   if (old_u & adj[w1] & adj[w2]).bit_count() >= 2: return False
        
        # Case 3: u is added to N(v).
        # Let candidates_u = [w for w in range(n) if w != u and w != v and (adj[u] & bit[w])]
        # For each pair (w1, w2) in candidates_u:
        #   if (old_v & adj[w1] & adj[w2]).bit_count() >= 2: return False
        
        # To speed this up, we can precompute lists or just iterate.
        # n=20, so lists are small.
        
        # Check Case 2
        # We only care about w1, w2 that are neighbors of v.
        # If v has degree d_v, we check O(d_v^2) pairs.
        mask_v_neighbors = adj[v] & ~bit[u] & ~bit[v]
        # Get list of neighbors of v (excluding u, v)
        neigh_v = []
        temp = mask_v_neighbors
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            neigh_v.append(idx)
            temp ^= low
            
        for idx1 in range(len(neigh_v)):
            w1 = neigh_v[idx1]
            for idx2 in range(idx1 + 1, len(neigh_v)):
                w2 = neigh_v[idx2]
                # Check if old_u & adj[w1] & adj[w2] has >= 2 bits
                common = old_u & adj[w1] & adj[w2]
                if common.bit_count() >= 2:
                    return False

        # Check Case 3
        mask_u_neighbors = adj[u] & ~bit[u] & ~bit[v]
        neigh_u = []
        temp = mask_u_neighbors
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            neigh_u.append(idx)
            temp ^= low
            
        for idx1 in range(len(neigh_u)):
            w1 = neigh_u[idx1]
            for idx2 in range(idx1 + 1, len(neigh_u)):
                w2 = neigh_u[idx2]
                common = old_v & adj[w1] & adj[w2]
                if common.bit_count() >= 2:
                    return False

        return True

    def local_search(adj, max_iter=1000):
        """
        Local search: Try to add edges greedily. If stuck, try swaps.
        """
        adj = adj[:]
        iters = 0
        while iters < max_iter and time.time() < deadline - 5:
            # 1. Try to add any valid edge
            added = False
            # Shuffle pairs to avoid bias
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if adj[u] & bit[v]:
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= bit[v]
                    adj[v] |= bit[u]
                    added = True
                    break
            
            if added:
                iters += 1
                continue
            
            # 2. If no edge can be added, try a swap
            # Find an edge (a,b) to remove and (c,d) to add
            # Heuristic: Remove an edge that is "critical" or random.
            # Add an edge that was just rejected.
            
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & bit[j]]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & bit[j])]
            
            if not edges_in or not edges_out:
                break
                
            # Try a few random swaps
            swapped = False
            # Limit swap attempts per iteration to save time
            for _ in range(50):
                e_rem = random.choice(edges_in)
                e_add = random.choice(edges_out)
                if e_rem == e_add:
                    continue
                
                u1, v1 = e_rem
                u2, v2 = e_add
                
                # Save state
                old_u1, old_v1 = adj[u1], adj[v1]
                old_u2, old_v2 = adj[u2], adj[v2]
                
                # Remove e_rem
                adj[u1] &= ~bit[v1]
                adj[v1] &= ~bit[u1]
                # Add e_add
                adj[u2] |= bit[v2]
                adj[v2] |= bit[u2]
                
                # Check validity
                if find_violation(adj) is None:
                    swapped = True
                    break
                else:
                    # Revert
                    adj[u1], adj[v1] = old_u1, old_v1
                    adj[u2], adj[v2] = old_u2, old_v2
            
            if not swapped:
                # Try a more aggressive move: Remove 2 edges, add 2?
                # Or just break to save time?
                # Let's try one more step: Remove a random edge, see if we can add 2.
                # Actually, simple swap is usually enough for this size.
                break
                
            iters += 1
            
        return adj

    best_adj = None
    best_count = -1
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: Start from Complete Graph and prune
    # This is often effective because the maximum density is high.
    adj = [full_mask ^ bit[i] for i in range(n)]
    
    # Repair loop
    for _ in range(1000):
        if time.time() > deadline - 30:
            break
        viol = find_violation(adj)
        if viol is None:
            break
        i, j, k = viol
        # Find a common neighbor x to remove an edge involving
        cij = adj[i] & adj[j] & adj[k]
        if cij == 0:
            break
        # Pick a random common neighbor to break the violation
        # Or pick one that has high degree?
        # Let's pick one that minimizes damage (low degree) or random.
        # Random is fine.
        x = (cij & (-cij)).bit_length() - 1
        # Remove edge (i, x)
        adj[i] &= ~bit[x]
        adj[x] &= ~bit[i]
        
    # Polish
    adj = local_search(adj, max_iter=500)
    update_best(adj)

    # Strategy 2: Random Restart with Greedy Construction
    # Start with empty or sparse graph and add edges.
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        
        random.seed(42 + trial * 101)
        adj = [0] * n
        
        # Initialize with random sparse graph
        p = 0.4
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= bit[j]
                adj[j] |= bit[i]
        
        # Repair
        for _ in range(500):
            if time.time() > deadline - 15:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j] & adj[k]
            if cij == 0:
                break
            x = (cij & (-cij)).bit_length() - 1
            adj[i] &= ~bit[x]
            adj[x] &= ~bit[i]
            
        # Polish
        adj = local_search(adj, max_iter=300)
        update_best(adj)

    # Strategy 3: Structured Initialization (Bipartite-like)
    # K3,3 is bipartite. A random bipartite graph might be a good start.
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        random.seed(1000 + trial * 7)
        adj = [0] * n
        # Split vertices into two sets
        # Random split
        set_a = set(random.sample(range(n), n // 2))
        set_b = set(range(n)) - set_a
        
        # Add edges between A and B with high probability
        for u in set_a:
            for v in set_b:
                if random.random() < 0.8:
                    adj[u] |= bit[v]
                    adj[v] |= bit[u]
        
        # Repair
        for _ in range(300):
            if time.time() > deadline - 5:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j] & adj[k]
            if cij == 0:
                break
            x = (cij & (-cij)).bit_length() - 1
            adj[i] &= ~bit[x]
            adj[x] &= ~bit[i]
            
        # Polish
        adj = local_search(adj, max_iter=200)
        update_best(adj)

    # Final Polish on the best solution found
    if best_adj is not None and time.time() < deadline - 2:
        adj = best_adj[:]
        adj = local_search(adj, max_iter=1000)
        update_best(adj)

    if not best_edges:
        # Fallback: Return a valid graph (e.g., empty or sparse)
        return [[i, i+1] for i in range(n-1)]
        
    return best_edges

if __name__ == "__main__":
    # Quick self-test
    edges = construct()
    print(f"Edges found: {len(edges)}")
