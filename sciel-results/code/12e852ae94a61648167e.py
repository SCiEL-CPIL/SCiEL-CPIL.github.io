import time
import random
import math
import numpy as np

N = 15
DEADLINE = 85.0

def _is_valid(adj, n):
    """
    Check if no simple path s-u-v-t (s != t) has deg(s) == deg(t).
    For every edge (u, v), and every s in N(u)\{v}, t in N(v)\{u} with s != t, deg(s) != deg(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:
                degs_u = set()
                for s in adj[u]:
                    if s != v:
                        degs_u.add(deg[s])
                degs_v = set()
                for t in adj[v]:
                    if t != u:
                        degs_v.add(deg[t])
                common = degs_u & degs_v
                if common:
                    for d in common:
                        s_list = [s for s in adj[u] if s != v and deg[s] == d]
                        t_list = [t for t in adj[v] if t != u and deg[t] == d]
                        found = False
                        for s in s_list:
                            for t in t_list:
                                if s != t:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            return False
    return True

def _is_valid_fast(adj_list, n):
    """Faster validity check using precomputed neighbor degree sets."""
    deg = [len(adj_list[i]) for i in range(n)]
    for u in range(n):
        neighbors_u = list(adj_list[u])
        for v in neighbors_u:
            if u >= v:
                continue
            degs_u = set(deg[s] for s in neighbors_u if s != v)
            degs_v = set(deg[t] for t in adj_list[v] if t != u)
            common = degs_u & degs_v
            if not common:
                continue
            for d in common:
                s_list = [s for s in neighbors_u if s != v and deg[s] == d]
                t_list = [t for t in adj_list[v] if t != u and deg[t] == d]
                for s in s_list:
                    for t in t_list:
                        if s != t:
                            return False
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def _make_bipartite(n, a, b):
    """Create K_{a,b} on n vertices."""
    adj = [set() for _ in range(n)]
    for i in range(a):
        for j in range(a, n):
            adj[i].add(j)
            adj[j].add(i)
    return adj

def _make_disjoint_bipartite(n, partitions):
    """
    Create disjoint union of K_{a_i, b_i} components.
    partitions: list of (a_i, b_i) where sum(a_i + b_i) == n
    Vertices are assigned sequentially: first component gets vertices 0..a_1+b_1-1, etc.
    """
    adj = [set() for _ in range(n)]
    offset = 0
    for a, b in partitions:
        for i in range(a):
            for j in range(a, a + b):
                adj[offset + i].add(offset + j)
                adj[offset + j].add(offset + i)
        offset += a + b
    return adj

def _search_disjoint_bipartite(n):
    """
    Search over all ways to partition n into pairs (a_i, b_i) with a_i, b_i >= 1,
    such that the degree sets {a_i, b_i} for different components are disjoint.
    Maximize total edges = sum(a_i * b_i).
    """
    best_edges = 0
    best_partitions = None
    
    def _rec(remaining, start_part, current_parts):
        nonlocal best_edges, best_partitions
        
        if remaining == 0:
            deg_sets = []
            for a, b in current_parts:
                deg_sets.append(frozenset({a, b}))
            valid = True
            for i in range(len(deg_sets)):
                for j in range(i + 1, len(deg_sets)):
                    if deg_sets[i] & deg_sets[j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                edges = sum(a * b for a, b in current_parts)
                if edges > best_edges:
                    best_edges = edges
                    best_partitions = list(current_parts)
            return
        
        for k in range(2, remaining + 1):
            for a in range(1, k):
                b = k - a
                current_parts.append((a, b))
                _rec(remaining - k, k, current_parts)
                current_parts.pop()
    
    _rec(n, 0, [])
    return best_edges, best_partitions

def _local_search(adj, n, seed, max_iter, deadline):
    """Local search: add/remove edges, accept if valid and improves."""
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    current_valid = _is_valid(adj, n)
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges if current_valid else 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        if j in adj[i]:
            adj[i].discard(j)
            adj[j].discard(i)
        else:
            adj[i].add(j)
            adj[j].add(i)
            
        if _is_valid(adj, n):
            new_edges = _edge_count(adj, n)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [set(s) for s in adj]
        else:
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
                
    return best_adj, best_edges

def _simulated_annealing(adj, n, seed, deadline, initial_temp=10.0, cooling_rate=0.999):
    """Simulated annealing to escape local optima."""
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    current_valid = _is_valid(adj, n)
    
    best_adj = [set(s) for s in adj]
    best_edges = current_edges if current_valid else 0
    
    temp = initial_temp
    no_improve_count = 0
    
    while time.time() < deadline:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        
        if j in adj[i]:
            adj[i].discard(j)
            adj[j].discard(i)
        else:
            adj[i].add(j)
            adj[j].add(i)
        
        new_valid = _is_valid(adj, n)
        new_edges = _edge_count(adj, n)
        
        if new_valid:
            delta = new_edges - current_edges
            if delta >= 0:
                current_edges = new_edges
                no_improve_count = 0
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [set(s) for s in adj]
            else:
                # Accept worse valid solution with probability exp(delta/temp)
                if rng.random() < math.exp(delta / temp):
                    current_edges = new_edges
                    no_improve_count += 1
                else:
                    # Revert
                    if j in adj[i]:
                        adj[i].discard(j)
                        adj[j].discard(i)
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
        else:
            # Revert invalid solution
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
        
        temp *= cooling_rate
        if temp < 0.1:
            temp = 0.1
        
        # If stuck for too long, restart with higher temp
        if no_improve_count > 500:
            temp = initial_temp
            no_improve_count = 0
    
    return best_adj, best_edges

def construct():
    start = time.time()
    
    # Strategy 1: Disjoint bipartite components
    bp_edges, bp_partitions = _search_disjoint_bipartite(N)
    if bp_partitions:
        best_adj = _make_disjoint_bipartite(N, bp_partitions)
        best_count = _edge_count(best_adj, N)
    else:
        best_adj = _make_bipartite(N, 7, 8)
        best_count = _edge_count(best_adj, N)
    
    # Verify
    if not _is_valid(best_adj, N):
        best_adj = _make_bipartite(N, 7, 8)
        best_count = _edge_count(best_adj, N)
        if not _is_valid(best_adj, N):
            best_adj = [set() for _ in range(N)]
            for v in range(1, N):
                best_adj[0].add(v)
                best_adj[v].add(0)
            best_count = 14
    
    # Strategy 2: Local search from multiple starts
    deadline = start + DEADLINE - 5
    
    # Start from best_adj with local search
    for seed in range(30):
        if time.time() > deadline:
            break
            
        perturbed = [set(s) for s in best_adj]
        rng = random.Random(seed)
        for _ in range(rng.randint(1, 4)):
            all_edges = []
            for i in range(N):
                for j in perturbed[i]:
                    if i < j:
                        all_edges.append((i, j))
            if all_edges:
                a, b = rng.choice(all_edges)
                perturbed[a].discard(b)
                perturbed[b].discard(a)
        
        result, cnt = _local_search(perturbed, N, 1000 + seed, 3000, deadline)
        
        if cnt > best_count and _is_valid(result, N):
            best_count = cnt
            best_adj = result
    
    # Strategy 3: Simulated annealing from best solution
    sa_deadline = deadline - 2
    for seed in range(10):
        if time.time() > sa_deadline:
            break
        
        result, cnt = _simulated_annealing(best_adj, N, 5000 + seed, sa_deadline)
        
        if cnt > best_count and _is_valid(result, N):
            best_count = cnt
            best_adj = result
    
    # Strategy 4: Try starting from K_{7,8}
    k78 = _make_bipartite(N, 7, 8)
    if _is_valid(k78, N):
        for seed in range(15):
            if time.time() > deadline:
                break
            perturbed = [set(s) for s in k78]
            rng = random.Random(10000 + seed)
            for _ in range(rng.randint(1, 3)):
                all_edges = []
                for i in range(N):
                    for j in perturbed[i]:
                        if i < j:
                            all_edges.append((i, j))
                if all_edges:
                    a, b = rng.choice(all_edges)
                    perturbed[a].discard(b)
                    perturbed[b].discard(a)
            result, cnt = _local_search(perturbed, N, 20000 + seed, 2000, deadline)
            if cnt > best_count and _is_valid(result, N):
                best_count = cnt
                best_adj = result
    
    # Final check
    if not _is_valid(best_adj, N):
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14

    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges
