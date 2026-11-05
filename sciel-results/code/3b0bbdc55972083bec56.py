import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    full_mask = (1 << n) - 1

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

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

    def is_valid(adj):
        """Check if no 3 vertices have 3 common neighbors (K3,3-free)."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a K3,3 violation. Returns (i,j,k) or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k)
        return None

    def make_valid(adj):
        """Remove edges to make the graph valid."""
        adj = adj[:]
        iterations = 0
        while iterations < 5000:
            if time.time() > deadline - 5:
                break
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cij = adj[i] & adj[j]
            cijk = cij & adj[k]
            if cijk == 0:
                break
            # Pick a random common neighbor
            bits = []
            v = cijk
            while v:
                low = v & (-v)
                bits.append(low.bit_length() - 1)
                v ^= low
            x = random.choice(bits)
            # Remove edge (i, x)
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            iterations += 1
        return adj

    def greedy_add(adj):
        """Try to add edges greedily until no more can be added."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    improved = True
                    break
                else:
                    adj[u], adj[v] = old_u, old_v
        return adj

    def local_search(adj, max_iter=2000):
        """Local search: try adding edges, and if blocked, try 1-swaps and 2-swaps."""
        adj = adj[:]
        iters = 0
        while iters < max_iter and time.time() < deadline - 15:
            # Try to add an edge
            added = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    added = True
                    break
                else:
                    adj[u], adj[v] = old_u, old_v
            if added:
                iters += 1
                continue

            # Try a 1-swap: remove one edge, add another
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]

            if not edges_out:
                break

            random.shuffle(edges_in)
            random.shuffle(edges_out)

            swapped = False
            for e_rem in edges_in[:50]:
                if swapped:
                    break
                for e_add in edges_out[:50]:
                    (u1, v1) = e_rem
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if is_valid(adj):
                        swapped = True
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                if not swapped:
                    continue

            if swapped:
                iters += 1
                continue

            # Try a 2-swap: remove two edges, add two different edges
            if len(edges_in) >= 2 and len(edges_out) >= 2:
                sample_in = random.sample(edges_in, min(20, len(edges_in)))
                sample_out = random.sample(edges_out, min(20, len(edges_out)))

                two_swapped = False
                for i_rem in range(len(sample_in)):
                    if two_swapped:
                        break
                    for j_rem in range(i_rem + 1, len(sample_in)):
                        if two_swapped:
                            break
                        e_rem1 = sample_in[i_rem]
                        e_rem2 = sample_in[j_rem]
                        (u1, v1) = e_rem1
                        (u2, v2) = e_rem2

                        old_u1, old_v1, old_u2, old_v2 = adj[u1], adj[v1], adj[u2], adj[v2]

                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)

                        for i_add in range(len(sample_out)):
                            if two_swapped:
                                break
                            for j_add in range(i_add + 1, len(sample_out)):
                                if two_swapped:
                                    break
                                e_add1 = sample_out[i_add]
                                e_add2 = sample_out[j_add]
                                (w1, x1) = e_add1
                                (w2, x2) = e_add2

                                if (e_add1 == e_rem1 or e_add1 == e_rem2 or e_add2 == e_rem1 or e_add2 == e_rem2):
                                    continue

                                adj[w1] |= (1 << x1)
                                adj[x1] |= (1 << w1)
                                adj[w2] |= (1 << x2)
                                adj[x2] |= (1 << w2)

                                if is_valid(adj):
                                    two_swapped = True
                                    break
                                else:
                                    adj[w1] &= ~(1 << x1)
                                    adj[x1] &= ~(1 << w1)
                                    adj[w2] &= ~(1 << x2)
                                    adj[x2] &= ~(1 << w2)

                        if two_swapped:
                            break
                        else:
                            adj[u1] = old_u1
                            adj[v1] = old_v1
                            adj[u2] = old_u2
                            adj[v2] = old_v2

                if two_swapped:
                    iters += 1
                    continue

            if not swapped and not two_swapped:
                break
            iters += 1

        return adj

    def simulated_annealing(adj, start_temp=1.0, end_temp=0.01, max_iter=5000):
        """Simulated annealing for escaping local minima."""
        adj = adj[:]
        current_temp = start_temp
        best_adj = adj[:]
        best_count = count_edges(adj)
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Generate a neighbor: either add an edge, remove an edge, or swap
            r = random.random()
            if r < 0.3:
                # Try to add an edge
                u, v = random.choice(all_pairs)
                if adj[u] & (1 << v):
                    continue
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
            elif r < 0.5:
                # Try to remove an edge
                edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
                if not edges_in:
                    continue
                u, v = random.choice(edges_in)
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                # Try a swap
                edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
                edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
                if not edges_in or not edges_out:
                    continue
                u1, v1 = random.choice(edges_in)
                u2, v2 = random.choice(edges_out)
                if (u1, v1) == (u2, v2):
                    continue
                new_adj = adj[:]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
            
            if not is_valid(new_adj):
                continue
            
            old_count = count_edges(adj)
            new_count = count_edges(new_adj)
            delta = new_count - old_count
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / current_temp):
                adj = new_adj
                if new_count > best_count:
                    best_count = new_count
                    best_adj = new_adj[:]
            
            current_temp *= 0.999
        
        return best_adj

    # Best solution tracking
    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: Start from complete graph, make valid, then improve
    adj = [(full_mask ^ (1 << i)) for i in range(n)]
    adj = make_valid(adj)
    adj = greedy_add(adj)
    adj = local_search(adj)
    update_best(adj)

    # Strategy 2: Simulated annealing from best
    if best_edges:
        adj = simulated_annealing(best_adj, start_temp=2.0, end_temp=0.01, max_iter=10000)
        update_best(adj)

    # Strategy 3: Try various structured initializations
    for s in range(5, 12):
        if time.time() > deadline - 30:
            break
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            random.seed(42 + s * 10 + trial)
            offsets = set()
            while len(offsets) < s:
                d = random.randint(1, n - 1)
                offsets.add(d)
                offsets.add(n - d)
            offsets.discard(0)

            adj = [0] * n
            for u in range(n):
                for d in offsets:
                    v = (u + d) % n
                    if v > u:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            if not is_valid(adj):
                adj = make_valid(adj)
            adj = greedy_add(adj)
            adj = local_search(adj)
            update_best(adj)

    # Strategy 4: Random dense graphs with local search
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        random.seed(1000 + trial * 7)
        adj = [0] * n
        p = 0.4 + trial * 0.06
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj = make_valid(adj)
        adj = greedy_add(adj)
        adj = local_search(adj)
        update_best(adj)

    # Strategy 5: More aggressive local search on best
    if best_edges:
        adj = best_adj[:]
        adj = local_search(adj, max_iter=15000)
        update_best(adj)

        for _ in range(15):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            adj = local_search(adj, max_iter=5000)
            update_best(adj)

    # Fallback
    if not best_edges:
        return [[i, i + 1] for i in range(19)]

    return best_edges
