import time
import random
import math

N = 15
DEADLINE = 85.0

def _is_valid(adj, n):
    """
    Check if no simple path s-u-v-t (s != t) has deg(s) == deg(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # Collect s_list and t_list
            s_list = [s for s in adj[u] if s != v]
            t_list = [t for t in adj[v] if t != u]
            # Build degree-to-vertex mapping for each side
            # For each degree d in s_list, if there's a vertex with same degree in t_list that is different
            deg_s = {}
            for s in s_list:
                d = deg[s]
                if d not in deg_s:
                    deg_s[d] = []
                deg_s[d].append(s)
            
            deg_t = {}
            for t in t_list:
                d = deg[t]
                if d not in deg_t:
                    deg_t[d] = []
                deg_t[d].append(t)
            
            # Check common degrees
            for d in deg_s:
                if d in deg_t:
                    # Check if there exist s in deg_s[d], t in deg_t[d] with s != t
                    found = False
                    for s in deg_s[d]:
                        for t in deg_t[d]:
                            if s != t:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
    return True

def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def _make_disjoint_bipartite(n, partitions):
    """
    Create disjoint union of K_{a_i, b_i} components.
    partitions: list of (a_i, b_i) where sum(a_i + b_i) == n
    Vertices are assigned sequentially: first component gets vertices 0..a_1+b_1-1, etc.
    """
    adj = [set() for _ in range(n)]
    offset = 0
    for a, b in partitions:
        # Component: vertices offset..offset+a-1 (part A) and offset+a..offset+a+b-1 (part B)
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
    
    def _rec(remaining, current_parts):
        nonlocal best_edges, best_partitions
        
        if remaining == 0:
            # Check validity: degree sets must be disjoint
            deg_sets = []
            for a, b in current_parts:
                deg_sets.append(frozenset({a, b}))
            # Check all pairs are disjoint
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
        
        # Try adding a part of size k (2 <= k <= remaining)
        for k in range(2, remaining + 1):
            # Split k into (a, b) with 1 <= a, b, a + b = k
            for a in range(1, k):
                b = k - a
                current_parts.append((a, b))
                _rec(remaining - k, current_parts)
                current_parts.pop()
    
    _rec(n, [])
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
            # Revert
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
                
    return best_adj, best_edges

def construct():
    start = time.time()
    
    # Strategy 1: Disjoint bipartite components
    bp_edges, bp_partitions = _search_disjoint_bipartite(N)
    if bp_partitions:
        best_adj = _make_disjoint_bipartite(N, bp_partitions)
        best_count = _edge_count(best_adj, N)
    else:
        # Fallback: star graph
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14
    
    # Verify
    if not _is_valid(best_adj, N):
        best_adj = [set() for _ in range(N)]
        for v in range(1, N):
            best_adj[0].add(v)
            best_adj[v].add(0)
        best_count = 14
    
    # Strategy 2: Local search from multiple starts
    deadline = start + DEADLINE - 5
    
    # Start from best_adj
    for seed in range(50):
        if time.time() > deadline:
            break
            
        perturbed = [set(s) for s in best_adj]
        rng = random.Random(seed)
        # Remove 1-3 random edges
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
