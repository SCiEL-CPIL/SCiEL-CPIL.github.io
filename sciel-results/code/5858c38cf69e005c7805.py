import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_edges)

    # Precompute triples (i<j<k)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))

    num_triples = len(triples)

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def try_add(adj, u, v):
        """Try to add edge (u,v). Returns True if valid after adding."""
        # Only triples containing exactly one of {u,v} can be affected.
        for a in range(n):
            if a == u:
                continue
            if not (adj[v] >> a) & 1:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[v] >> b) & 1:
                    continue
                cm = adj[u] & adj[a] & adj[b]
                mask = (1 << u) | (1 << a) | (1 << b)
                cm &= ~mask
                if bin(cm).count('1') >= 2:
                    return False
        for a in range(n):
            if a == v:
                continue
            if not (adj[u] >> a) & 1:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[u] >> b) & 1:
                    continue
                cm = adj[v] & adj[a] & adj[b]
                mask = (1 << v) | (1 << a) | (1 << b)
                cm &= ~mask
                if bin(cm).count('1') >= 2:
                    return False
        return True

    def is_valid_full(adj):
        """Check if the entire graph is K3,3-free."""
        for t in range(num_triples):
            i, j, k = triples[t]
            cm = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            cm &= ~mask
            if bin(cm).count('1') >= 3:
                return False
        return True

    def find_violation(adj):
        """Find a violating triple if any. Returns (i,j,k) or None."""
        for t in range(num_triples):
            i, j, k = triples[t]
            cm = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            cm &= ~mask
            if bin(cm).count('1') >= 3:
                return (i, j, k)
        return None

    def repair(adj, max_iters=500):
        """Remove edges until valid. Returns edge count."""
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cm = adj[i] & adj[j] & adj[k]
            mask = (1 << i) | (1 << j) | (1 << k)
            cm &= ~mask
            c_list = [x for x in range(n) if (cm >> x) & 1]
            if not c_list:
                break
            c = rng.choice(c_list)
            v = rng.choice([i, j, k])
            remove_edge(adj, v, c)
        return count_edges(adj)

    def greedy_add(adj):
        """Greedy add edges in random order."""
        edges = count_edges(adj)
        remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        while remaining:
            if time.time() > deadline - 3:
                break
            rng.shuffle(remaining)
            progress = False
            new_remaining = []
            for (u, v) in remaining:
                if (adj[u] >> v) & 1:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges += 1
                    progress = True
                else:
                    new_remaining.append((u, v))
            remaining = new_remaining
            if not progress:
                break
        return edges

    def local_search(adj, max_rounds=50):
        """
        Local search using edge swaps: remove one edge, try to add one or more.
        This allows escaping local optima of pure addition.
        """
        edges = count_edges(adj)
        present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            found_swap = False
            
            # Try removing each present edge and see if we can add more than 1 back
            rng.shuffle(present)
            for (u1, v1) in present:
                if time.time() > deadline - 5:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                
                remove_edge(adj, u1, v1)
                edges -= 1
                
                # Try to add back edges (including the original one)
                added_back = 0
                new_absent = []
                for (u2, v2) in absent:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if try_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        edges += 1
                        added_back += 1
                        # Update present/absent lists
                        if (u2, v2) in absent:
                            absent.remove((u2, v2))
                            present.append((u2, v2))
                    else:
                        new_absent.append((u2, v2))
                
                # Also try to re-add the original edge
                if not ((adj[u1] >> v1) & 1):
                    if try_add(adj, u1, v1):
                        add_edge(adj, u1, v1)
                        edges += 1
                        added_back += 1
                        present.append((u1, v1))
                    
                # If we added back more than we removed (net gain), keep it
                # If we added back exactly 1, it's a neutral swap - still useful for exploration
                # If we added back 0, revert
                if added_back == 0:
                    # Revert: add back u1,v1
                    add_edge(adj, u1, v1)
                    edges += 1
                    # Restore absent list
                    absent = new_absent + [(u1, v1)] if (u1, v1) not in new_absent else new_absent
                    # Actually this is getting complicated. Let's simplify.
                    # Just break out and try a different approach.
                    break
                
                if added_back >= 1:
                    found_swap = True
                    # Rebuild present and absent
                    present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
                    absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
                    break
            
            if not found_swap:
                break
        
        # Final polish
        edges = greedy_add(adj)
        return edges

    def polish(adj):
        """Try to add any remaining edges that can be added."""
        edges = count_edges(adj)
        remaining = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        added = True
        while added:
            if time.time() > deadline - 3:
                break
            added = False
            new_remaining = []
            for (u, v) in remaining:
                if (adj[u] >> v) & 1:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges += 1
                    added = True
                else:
                    new_remaining.append((u, v))
            remaining = new_remaining
        return edges

    def make_paley_start():
        """Paley-type graph."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 5
                if d not in (1, 4):
                    add_edge(adj, i, j)
        return adj

    def make_turan_start(num_parts):
        """Complete multipartite graph."""
        adj = [0] * n
        part_size = n // num_parts
        for i in range(n):
            for j in range(i + 1, n):
                pi = i // part_size
                pj = j // part_size
                if pi != pj:
                    add_edge(adj, i, j)
        return adj

    def make_random_start(density):
        """Random graph with given edge probability."""
        adj = [0] * n
        for (i, j) in all_edges:
            if rng.random() < density:
                add_edge(adj, i, j)
        return adj

    def make_circulant_start(skip_set):
        """Circulant graph."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d not in skip_set and (n - d) not in skip_set:
                    add_edge(adj, i, j)
        return adj

    def make_complete_start():
        """Start with complete graph, then repair."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                add_edge(adj, i, j)
        return adj

    rng = random.Random(42)

    # Define various structured starts
    start_generators = [
        make_complete_start,
        make_paley_start,
        lambda: make_turan_start(4),
        lambda: make_turan_start(5),
        lambda: make_circulant_start({1, 4, 9, 16}),
        lambda: make_circulant_start({2, 3, 8, 12, 17}),
        lambda: make_random_start(0.7),
        lambda: make_random_start(0.65),
        lambda: make_random_start(0.6),
        lambda: make_random_start(0.75),
        lambda: make_circulant_start({1, 2, 8, 9, 11, 12}),
        lambda: make_circulant_start({1, 3, 7, 13, 19}),
        lambda: make_circulant_start({2, 5, 15, 18}),
    ]

    best_edges = 0
    best_adj = None

    num_starts = 60

    for s in range(num_starts):
        if time.time() > deadline - 15:
            break

        gen = start_generators[s % len(start_generators)]
        adj = gen()

        # For some starts, vary parameters
        if s % len(start_generators) >= 6:
            adj = [0] * n
            density = 0.55 + 0.03 * (s % 8)
            for (i, j) in all_edges:
                if rng.random() < density:
                    add_edge(adj, i, j)

        # Repair to validity
        ec = repair(adj)

        # Greedy add
        ec = greedy_add(adj)

        # Local search with swaps
        ec = local_search(adj)

        # Polish
        ec = polish(adj)

        # Verify validity
        if is_valid_full(adj):
            if ec > best_edges:
                best_edges = ec
                best_adj = [a for a in adj]

    # If no valid graph found, return empty
    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges
