import time
import random
from itertools import combinations

N = 15
DEADLINE = 85.0  # Safety margin under the 90s limit

def _adjacency(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(N)]

def _is_valid(adj, deg):
    """Check if no 3-edge path s-u-v-t has deg[s] == deg[t]."""
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _greedy(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    adj = _adjacency(edges)
    deg = _degrees(adj)
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        # Try adding edge
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        if _is_valid(adj, deg):
            edges.append((u, v))
            present.add(key)
        else:
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
    return edges

def _local_search_swap(start_edges, max_swaps=50):
    """Try removing one edge and adding a different one."""
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    adj = _adjacency(edges)
    deg = _degrees(adj)
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    improved = True
    swaps_done = 0
    
    while improved and swaps_done < max_swaps:
        improved = False
        # Try removing each edge
        for i, (ru, rv) in enumerate(edges):
            key_r = (min(ru, rv), max(ru, rv))
            if key_r not in present:
                continue
            # Remove this edge
            adj[ru].remove(rv)
            adj[rv].remove(ru)
            deg[ru] -= 1
            deg[rv] -= 1
            present.discard(key_r)
            
            # Try adding each non-adjacent edge
            for (au, av) in all_edges:
                key_a = (min(au, av), max(au, av))
                if key_a in present:
                    continue
                if key_a == key_r:
                    continue
                # Try adding
                adj[au].add(av)
                adj[av].add(au)
                deg[au] += 1
                deg[av] += 1
                if _is_valid(adj, deg):
                    # Accept the swap
                    edges.pop(i)
                    edges.append((au, av))
                    present.add(key_a)
                    improved = True
                    swaps_done += 1
                    break
                else:
                    adj[au].remove(av)
                    adj[av].remove(au)
                    deg[au] -= 1
                    deg[av] -= 1
            
            if improved:
                break
            else:
                # Restore
                adj[ru].add(rv)
                adj[rv].add(ru)
                deg[ru] += 1
                deg[rv] += 1
                present.add(key_r)
    
    return edges

def construct():
    t0 = time.time()
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng = random.Random(20240607)

    # K_{7,8}: 56 edges, valid
    k78 = [(a, b) for a in range(7) for b in range(7, 15)]
    best = k78[:]
    best_len = len(k78)

    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand[:]

    # Phase 1: Greedy from K_{7,8} with many random orders
    for _ in range(150):
        if time.time() - t0 > DEADLINE:
            break
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, k78)
        consider(cand)
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 2: Remove 1-2 edges from K_{7,8}, then greedy add
    for _ in range(200):
        if time.time() - t0 > DEADLINE:
            break
        base = k78[:]
        n_remove = rng.choice([1, 2])
        for _ in range(n_remove):
            if base:
                idx = rng.randint(0, len(base) - 1)
                base.pop(idx)
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, base)
        consider(cand)
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy(order2, cand)
            consider(cand2)

    # Phase 3: Random dense graphs with repair + greedy
    for _ in range(80):
        if time.time() - t0 > DEADLINE:
            break
        adj = [set() for _ in range(N)]
        p = rng.uniform(0.5, 0.9)
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        edges = [[u, v] for u in range(N) for v in adj[u] if u < v]
        
        # Repair: remove edges from violations
        max_steps = 50
        steps = 0
        valid = False
        while not valid and steps < max_steps:
            deg = _degrees(adj)
            viol = None
            for u in range(N):
                au = adj[u]
                for v in au:
                    if v <= u:
                        continue
                    av = adj[v]
                    for s in au:
                        if s == v:
                            continue
                        ds = deg[s]
                        for t in av:
                            if t == u or t == s:
                                continue
                            if ds == deg[t]:
                                viol = (s, u, v, t)
                                break
                        if viol:
                            break
                    if viol:
                        break
                if viol:
                    break
            
            if viol is None:
                valid = True
                break
            
            s, u, v, t = viol
            e_options = [(s, u), (u, v), (v, t)]
            a, b = rng.choice(e_options)
            adj[a].remove(b)
            adj[b].remove(a)
            steps += 1
        
        if valid:
            cand = [[u, v] for u in range(N) for v in adj[u] if u < v]
            consider(cand)
            if time.time() - t0 < DEADLINE:
                order = all_edges[:]
                rng.shuffle(order)
                cand2 = _greedy(order, cand)
                consider(cand2)

    # Phase 4: Local search with edge swaps on best
    for _ in range(100):
        if time.time() - t0 > DEADLINE:
            break
        base = best[:]
        cand = _local_search_swap(base, max_swaps=30)
        consider(cand)
        if time.time() - t0 < DEADLINE:
            order = all_edges[:]
            rng.shuffle(order)
            cand2 = _greedy(order, cand)
            consider(cand2)

    # Phase 5: Try other bipartite splits as seeds
    for a in range(4, 11):
        b = N - a
        seed = [(u, v) for u in range(a) for v in range(a, N)]
        if time.time() - t0 > DEADLINE:
            break
        for _ in range(5):
            if time.time() - t0 > DEADLINE:
                break
            order = all_edges[:]
            rng.shuffle(order)
            cand = _greedy(order, seed)
            consider(cand)

    # Phase 6: Final polish - try local search on multiple good candidates
    if time.time() - t0 < DEADLINE:
        # Collect diverse candidates
        candidates = [best[:]]
        # Add some random perturbations of best
        for _ in range(20):
            if time.time() - t0 > DEADLINE:
                break
            perturbed = best[:]
            n_remove = rng.randint(1, 3)
            for _ in range(n_remove):
                if perturbed:
                    idx = rng.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
            order = all_edges[:]
            rng.shuffle(order)
            cand = _greedy(order, perturbed)
            if len(cand) not in [len(c) for c in candidates]:
                candidates.append(cand)
        
        # Run local search on each
        for cand in candidates:
            if time.time() - t0 > DEADLINE:
                break
            improved = _local_search_swap(cand, max_swaps=50)
            consider(improved)
            if time.time() - t0 < DEADLINE:
                order = all_edges[:]
                rng.shuffle(order)
                cand2 = _greedy(order, improved)
                consider(cand2)

    # Phase 7: Degree-sequence guided search (inspiration from Erdős–Gallai)
    # Try to find valid graphs with non-bipartite degree structures
    if time.time() - t0 < DEADLINE:
        # Generate candidate degree sequences using Erdős–Gallai
        # For a graph with n vertices and m edges, sum of degrees = 2m
        # Try degree sequences that are "almost regular" but with small perturbations
        # to break the bipartite symmetry
        
        for trial in range(30):
            if time.time() - t0 > DEADLINE:
                break
            
            # Create a degree sequence close to regular but with variations
            base_deg = 7  # Average degree for ~56 edges
            deg_seq = [base_deg] * N
            # Add small perturbations
            for _ in range(rng.randint(1, 3)):
                idx1 = rng.randint(0, N - 1)
                idx2 = rng.randint(0, N - 1)
                if idx1 != idx2:
                    if deg_seq[idx1] > 1:
                        deg_seq[idx1] -= 1
                    if deg_seq[idx2] < N - 1:
                        deg_seq[idx2] += 1
            
            # Check graphicality using Erdős–Gallai (simplified check)
            deg_sorted = sorted(deg_seq, reverse=True)
            total_deg = sum(deg_sorted)
            if total_deg % 2 != 0:
                continue
            
            # Simple Erdős–Gallai check
            valid_seq = True
            for k in range(1, N + 1):
                lhs = sum(deg_sorted[:k])
                rhs = k * (k - 1) + sum(min(deg_sorted[i], k) for i in range(k, N))
                if lhs > rhs:
                    valid_seq = False
                    break
            
            if not valid_seq:
                continue
            
            # Try to construct a graph with this degree sequence
            # Using Havel-Hakimi algorithm
            remaining = deg_seq[:]
            adj = [set() for _ in range(N)]
            success = True
            
            for _ in range(N):
                # Find vertex with highest degree
                max_idx = -1
                max_deg = -1
                for i in range(N):
                    if remaining[i] > max_deg:
                        max_deg = remaining[i]
                        max_idx = i
                
                if max_deg < 0:
                    break
                if max_deg == 0:
                    continue
                
                # Connect to max_deg other vertices with positive degree
                candidates = [i for i in range(N) if i != max_idx and remaining[i] > 0]
                if len(candidates) < max_deg:
                    success = False
                    break
                
                # Choose max_deg candidates (prefer those with highest degree)
                candidates.sort(key=lambda x: -remaining[x])
                for j in range(max_deg):
                    v = candidates[j]
                    adj[max_idx].add(v)
                    adj[v].add(max_idx)
                    remaining[v] -= 1
                
                remaining[max_idx] = 0
            
            if not success:
                continue
            
            # Check if the constructed graph is valid
            edges = [[u, v] for u in range(N) for v in adj[u] if u < v]
            if _valid(edges):
                consider(edges)
                if time.time() - t0 < DEADLINE:
                    order = all_edges[:]
                    rng.shuffle(order)
                    cand2 = _greedy(order, edges)
                    consider(cand2)

    result = [[int(u), int(v)] for (u, v) in best]
    return result
