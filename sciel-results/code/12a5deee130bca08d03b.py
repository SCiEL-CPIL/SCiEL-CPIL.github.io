"""Joint simulated annealing over value permutations and edge sets."""
import time
import itertools
import random

DEADLINE = 82.0  # safety margin before 90s kill
N = 14
M = 35

def _raw_score(values, edges):
    """Compute max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[b]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _count_sums_prods(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return len(sums), len(prods)

def _all_edges(n):
    return list(itertools.combinations(range(n), 2))

ALL_EDGES = _all_edges(N)
EDGE_TO_IDX = {e: i for i, e in enumerate(ALL_EDGES)}

def _greedy_edge_init(values):
    """Fast greedy: pick edges that minimize max(new_sums, new_prods)."""
    sums = set()
    prods = set()
    edges = []
    used = set()
    for _ in range(M):
        best_edge = None
        best_key = None
        for (u, v) in ALL_EDGES:
            if (u, v) in used:
                continue
            s = values[u] + values[v]
            p = values[u] * values[v]
            ns = 1 if s not in sums else 0
            np_ = 1 if p not in prods else 0
            key = (max(len(sums) + ns, len(prods) + np_), len(sums) + ns, len(prods) + np_, s, p)
            if best_key is None or key < best_key:
                best_key = key
                best_edge = (u, v)
        if best_edge is None:
            break
        used.add(best_edge)
        edges.append(best_edge)
        sums.add(values[best_edge[0]] + values[best_edge[1]])
        prods.add(values[best_edge[0]] * values[best_edge[1]])
    return edges

def _anneal(values, edges, start_temp, n_steps, deadline):
    """Simulated annealing on edge set for fixed values."""
    cur_edges = list(edges)
    cur_ns, cur_np = _count_sums_prods(values, cur_edges)
    cur_score = max(cur_ns, cur_np)
    best_edges = list(cur_edges)
    best_ns, best_np = cur_ns, cur_np
    best_score = cur_score
    edge_set = set(cur_edges)
    temp = start_temp
    cooling = 0.995
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        # Pick a random edge to remove and a random non-edge to add
        idx = random.randrange(len(cur_edges))
        remove_edge = cur_edges[idx]
        # Find a non-edge
        while True:
            cand = random.choice(ALL_EDGES)
            if cand not in edge_set:
                break
        # Compute new state
        new_edges = cur_edges[:idx] + cur_edges[idx+1:] + [cand]
        new_ns, new_np = _count_sums_prods(values, new_edges)
        new_score = max(new_ns, new_np)
        delta = new_score - cur_score
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-9)):
            cur_edges = new_edges
            cur_ns, cur_np = new_ns, new_np
            cur_score = new_score
            edge_set.discard(remove_edge)
            edge_set.add(cand)
            if cur_score < best_score:
                best_score = cur_score
                best_edges = list(cur_edges)
                best_ns, best_np = cur_ns, cur_np
        temp *= cooling
    
    return best_edges, best_score

def _anneal_values(values, edges, start_temp, n_steps, deadline):
    """Annealing that also permutes values."""
    cur_vals = list(values)
    cur_edges = list(edges)
    cur_ns, cur_np = _count_sums_prods(cur_vals, cur_edges)
    cur_score = max(cur_ns, cur_np)
    best_vals = list(cur_vals)
    best_edges = list(cur_edges)
    best_score = cur_score
    edge_set = set(cur_edges)
    temp = start_temp
    cooling = 0.997
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.4:
            # Swap two values
            i, j = random.sample(range(N), 2)
            cur_vals[i], cur_vals[j] = cur_vals[j], cur_vals[i]
            new_ns, new_np = _count_sums_prods(cur_vals, cur_edges)
            new_score = max(new_ns, new_np)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-9)):
                cur_ns, cur_np = new_ns, new_np
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(cur_vals)
                    best_edges = list(cur_edges)
            else:
                cur_vals[i], cur_vals[j] = cur_vals[j], cur_vals[i]
        elif r < 0.8:
            # Edge swap
            idx = random.randrange(len(cur_edges))
            remove_edge = cur_edges[idx]
            while True:
                cand = random.choice(ALL_EDGES)
                if cand not in edge_set:
                    break
            new_edges = cur_edges[:idx] + cur_edges[idx+1:] + [cand]
            new_ns, new_np = _count_sums_prods(cur_vals, new_edges)
            new_score = max(new_ns, new_np)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-9)):
                cur_edges = new_edges
                cur_ns, cur_np = new_ns, new_np
                cur_score = new_score
                edge_set.discard(remove_edge)
                edge_set.add(cand)
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(cur_vals)
                    best_edges = list(cur_edges)
            else:
                pass
        else:
            # Rotate three values
            i, j, k = random.sample(range(N), 3)
            vi, vj, vk = cur_vals[i], cur_vals[j], cur_vals[k]
            cur_vals[i], cur_vals[j], cur_vals[k] = vk, vi, vj
            new_ns, new_np = _count_sums_prods(cur_vals, cur_edges)
            new_score = max(new_ns, new_np)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-9)):
                cur_ns, cur_np = new_ns, new_np
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(cur_vals)
                    best_edges = list(cur_edges)
            else:
                cur_vals[i], cur_vals[j], cur_vals[k] = vi, vj, vk
        temp *= cooling
    
    return best_vals, best_edges, best_score

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_vals = None
    best_edges = None
    random.seed(12345)
    
    # Candidate value sets: small ranges maximize collisions
    candidate_sets = []
    
    # 1-14 (maximal collision for small products)
    candidate_sets.append(list(range(1, 15)))
    # 2-15
    candidate_sets.append(list(range(2, 16)))
    # 1-14 with some perturbation
    for _ in range(10):
        base = list(range(1, 15))
        i = random.randrange(N)
        base[i] = random.randint(15, 30)
        if len(set(base)) == N:
            candidate_sets.append(base)
    
    # Multiples of small numbers
    for g in [2, 3]:
        for base_start in [1, 2]:
            vals = [g * (base_start + i) for i in range(N)]
            if max(vals) <= 10000 and len(set(vals)) == N:
                candidate_sets.append(vals)
    
    # Random small sets
    for max_v in [18, 20, 25, 30]:
        for _ in range(5):
            vals = random.sample(range(1, max_v + 1), N)
            candidate_sets.append(vals)
    
    # Process each candidate with annealing
    for vals in candidate_sets:
        if time.time() > DEADLINE:
            break
        if len(set(vals)) != N:
            continue
        if any(not isinstance(v, int) or v < 1 or v > 10000 for v in vals):
            continue
        
        # Greedy init
        edges = _greedy_edge_init(vals)
        if len(edges) < M:
            continue
        ns, np_ = _count_sums_prods(vals, edges)
        score = max(ns, np_)
        
        if score < best_score:
            best_score = score
            best_vals = list(vals)
            best_edges = list(edges)
        
        # Anneal edges only (fast)
        if time.time() < DEADLINE - 3:
            e2, s2 = _anneal(vals, edges, start_temp=2.0, n_steps=500, deadline=DEADLINE - 2)
            if s2 < best_score:
                best_score = s2
                best_vals = list(vals)
                best_edges = list(e2)
        
        # Joint anneal (slower)
        if time.time() < DEADLINE - 5 and score <= best_score + 2:
            v2, e2, s2 = _anneal_values(vals, edges, start_temp=3.0, n_steps=800, deadline=DEADLINE - 2)
            if s2 < best_score:
                best_score = s2
                best_vals = list(v2)
                best_edges = list(e2)
    
    # Final polish on best
    if best_vals is not None and best_edges is not None and time.time() < DEADLINE - 1:
        v2, e2, s2 = _anneal_values(best_vals, best_edges, start_temp=1.0, n_steps=600, deadline=DEADLINE - 0.5)
        if s2 < best_score:
            best_score = s2
            best_vals = list(v2)
            best_edges = list(e2)
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(itertools.combinations(range(N), 2))[:M]
    
    best_vals = [int(v) for v in best_vals]
    best_edges = [list(e) for e in best_edges]
    return {'values': best_vals, 'edges': best_edges}
