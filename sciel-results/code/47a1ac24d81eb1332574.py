import time
import random
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def get_bitmasks(adj, n):
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def has_k4_fast(adj, n, masks):
    """Check if graph contains a K4 using precomputed bitmasks."""
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if (masks[v] >> w) & 1:
                    common = masks[v] & masks[w] & masks[u]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if common:
                        return True
    return False

def count_k4s_fast(adj, n, masks):
    """Count K4s in the graph."""
    count = 0
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if (masks[v] >> w) & 1:
                    common = masks[v] & masks[w] & masks[u]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    count += bin(common).count('1')
    return count

def has_independent_set_fast(adj, n, k, masks):
    """Check if graph contains an independent set of size k."""
    if n < k:
        return False
    
    def backtrack(start, current_mask, size):
        if size == k:
            return True
        if start >= n:
            return False
        if (n - start) < (k - size):
            return False
        
        candidates = []
        for v in range(start, n):
            if not (current_mask & (1 << v)):
                candidates.append(v)
        candidates.sort(key=lambda v: len(adj[v]))
        
        for v in candidates:
            if backtrack(v + 1, current_mask | masks[v], size + 1):
                return True
        return False
    
    return backtrack(0, 0, 0)

def get_independent_sets_size_k_fast(adj, n, k, masks):
    """Return list of independent sets of size k."""
    if n < k:
        return []
    
    sets = []
    
    def backtrack(start, current_set, neighbor_mask):
        if len(current_set) == k:
            sets.append(tuple(current_set))
            return
        remaining = k - len(current_set)
        if n - start < remaining:
            return
        
        for v in range(start, n):
            if not (neighbor_mask & (1 << v)):
                current_set.append(v)
                backtrack(v + 1, current_set, neighbor_mask | masks[v])
                current_set.pop()
                
    backtrack(0, [], 0)
    return sets

def try_add_vertex_walksat(adj, n, deadline, masks):
    """
    WalkSAT-style approach to find valid neighbor set for new vertex.
    """
    ind_sets_5 = get_independent_sets_size_k_fast(adj, n, 5, masks)
    
    if not ind_sets_5:
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining:
            neigh_in_N = [u for u in N if u in adj[v]]
            triangle_formed = False
            for i in range(len(neigh_in_N)):
                for j in range(i+1, len(neigh_in_N)):
                    if neigh_in_N[j] in adj[neigh_in_N[i]]:
                        triangle_formed = True
                        break
                if triangle_formed:
                    break
            if not triangle_formed:
                N.add(v)
        return N
    
    # WalkSAT: maintain a candidate set N, flip vertices to reduce violations
    best_N = None
    best_violations = float('inf')
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize N randomly
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining[:n//3]:
            N.add(v)
        
        # WalkSAT iterations
        for it in range(500):
            if time.time() > deadline:
                break
                
            # Count violations: independent sets of size 5 with no vertex in N
            violations = 0
            violating_sets = []
            for s in ind_sets_5:
                if not (set(s) & N):
                    violations += 1
                    violating_sets.append(s)
            
            if violations == 0:
                # Check triangle constraint
                valid = True
                for v in N:
                    neigh_in_N = [u for u in N if u in adj[v]]
                    for i in range(len(neigh_in_N)):
                        for j in range(i+1, len(neigh_in_N)):
                            if neigh_in_N[j] in adj[neigh_in_N[i]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return N.copy()
            
            if violations < best_violations:
                best_violations = violations
                best_N = N.copy()
            
            if not violating_sets:
                break
            
            # Pick a random violating set
            target_set = random.choice(violating_sets)
            
            # Decide to add or remove a vertex
            if random.random() < 0.7:
                # Add a vertex from target_set
                candidates = [v for v in target_set if v not in N]
                if candidates:
                    v = random.choice(candidates)
                    N.add(v)
            else:
                # Remove a vertex from N that's in target_set
                in_target = [v for v in N if v in target_set]
                if in_target:
                    v = random.choice(in_target)
                    N.discard(v)
                else:
                    # Add a random vertex from target_set
                    candidates = list(target_set)
                    if candidates:
                        v = random.choice(candidates)
                        N.add(v)
        
        if best_N is not None and best_violations == 0:
            return best_N
    
    return best_N

def local_search_walksat(adj, n, masks, deadline):
    """
    WalkSAT-style local search on edge flips to enable vertex additions.
    """
    best_n = n
    best_adj = [set(s) for s in adj]
    best_masks = list(masks)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Randomly perturb edges
        cur_adj = [set(s) for s in adj]
        cur_masks = list(masks)
        
        for _ in range(random.randint(5, 20)):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if v in cur_adj[u]:
                cur_adj[u].remove(v)
                cur_adj[v].remove(u)
                cur_masks[u] &= ~(1 << v)
                cur_masks[v] &= ~(1 << u)
            else:
                cur_adj[u].add(v)
                cur_adj[v].add(u)
                cur_masks[u] |= (1 << v)
                cur_masks[v] |= (1 << u)
        
        # WalkSAT on edge flips
        for it in range(200):
            if time.time() > deadline - 5:
                break
            
            # Check current validity
            if has_k4_fast(cur_adj, n, cur_masks) or has_independent_set_fast(cur_adj, n, 6, cur_masks):
                # Find a violation and flip an edge to fix it
                if has_k4_fast(cur_adj, n, cur_masks):
                    # Find a K4 and flip an edge within it
                    found = False
                    for u in range(n):
                        neighbors = list(cur_adj[u])
                        for i in range(len(neighbors)):
                            v = neighbors[i]
                            for j in range(i + 1, len(neighbors)):
                                w = neighbors[j]
                                if (cur_masks[v] >> w) & 1:
                                    common = cur_masks[v] & cur_masks[w] & cur_masks[u]
                                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                                    if common:
                                        x = common.bit_length() - 1
                                        # Flip edge (u,x) or (v,x) or (w,x)
                                        flip_pairs = [(u, x), (v, x), (w, x)]
                                        random.shuffle(flip_pairs)
                                        for a, b in flip_pairs:
                                            if b in cur_adj[a]:
                                                cur_adj[a].remove(b)
                                                cur_adj[b].remove(a)
                                                cur_masks[a] &= ~(1 << b)
                                                cur_masks[b] &= ~(1 << a)
                                            else:
                                                cur_adj[a].add(b)
                                                cur_adj[b].add(a)
                                                cur_masks[a] |= (1 << b)
                                                cur_masks[b] |= (1 << a)
                                            found = True
                                            break
                                        if found:
                                            break
                                    if found:
                                        break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        break
                else:
                    # Find an independent set of size 6 and add an edge within it
                    ind_sets = get_independent_sets_size_k_fast(cur_adj, n, 6, cur_masks)
                    if not ind_sets:
                        break
                    s = random.choice(ind_sets)
                    u, v = random.sample(s, 2)
                    cur_adj[u].add(v)
                    cur_adj[v].add(u)
                    cur_masks[u] |= (1 << v)
                    cur_masks[v] |= (1 << u)
            else:
                # Valid! Try to add a vertex
                new_N = try_add_vertex_walksat(cur_adj, n, deadline, cur_masks)
                if new_N is not None:
                    cur_adj.append(set(new_N))
                    for u2 in new_N:
                        cur_adj[u2].add(n)
                    new_mask = 0
                    for u2 in new_N:
                        new_mask |= (1 << u2)
                    cur_masks.append(new_mask)
                    for u2 in new_N:
                        cur_masks[u2] |= (1 << n)
                    n += 1
                    best_n = n
                    best_adj = [set(s) for s in cur_adj]
                    best_masks = list(cur_masks)
                    break
        
        if n == best_n:
            break
    
    return best_n, best_adj, best_masks

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    g17 = generate_paley(17)
    if not g17:
        return {'n': 1, 'edges': []}
        
    adj = build_adjacency(17, g17['edges'])
    n = 17
    masks = get_bitmasks(adj, n)
    
    if has_k4_fast(adj, n, masks) or has_independent_set_fast(adj, n, 6, masks):
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_adj = [set(s) for s in adj]
    best_masks = list(masks)
    
    while n < 35 and time.time() < deadline - 10:
        added = False
        
        new_N = try_add_vertex_walksat(adj, n, deadline, masks)
        
        if new_N is not None:
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            
            new_mask = 0
            for u in new_N:
                new_mask |= (1 << u)
            masks.append(new_mask)
            
            for u in new_N:
                masks[u] |= (1 << n)
                
            n += 1
            added = True
            best_n = n
            best_adj = [set(s) for s in adj]
            best_masks = list(masks)
            
        if not added:
            if time.time() < deadline - 15:
                new_n, new_adj, new_masks = local_search_walksat(adj, n, masks, deadline)
                if new_n > n:
                    n = new_n
                    adj = new_adj
                    masks = new_masks
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    best_masks = list(masks)
                    continue
            
            break
    
    n = best_n
    adj = best_adj
    masks = best_masks
    
    if has_k4_fast(adj, n, masks) or has_independent_set_fast(adj, n, 6, masks):
        return {'n': 1, 'edges': []}
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
