import random
import time
import sys
from collections import deque
import numpy as np
import math

def construct(k, g):
    # If k or g are not 4 and 10, fallback to standard construction
    if k != 4 or g != 10:
        return _fallback_construct(k, g)
    
    deadline = time.time() + 115  # 115s limit to be safe (150s total, leave 35s margin)
    
    # Attempt to construct the Moore graph (n=242)
    try:
        n, edges = _construct_moore_candidate(deadline)
        if n is not None:
            # Verify girth
            if _verify_girth(n, edges, g):
                return n, edges
    except Exception:
        pass
    
    # Fallback to original Lazebnik-Ustimenko-Woldar construction
    return _fallback_construct(k, g)

def _construct_moore_candidate(deadline):
    # Build two trees of depth 4, rooted at 0 and 1
    edges = []
    edges.append((0, 1))
    
    # Left Tree
    l1 = [2, 3, 4]
    for node in l1: edges.append((0, node))
    
    l2 = []
    idx = 5
    for node in l1:
        for _ in range(3):
            edges.append((node, idx))
            l2.append(idx)
            idx += 1
            
    l3 = []
    for node in l2:
        for _ in range(3):
            edges.append((node, idx))
            l3.append(idx)
            idx += 1
            
    l4 = []
    for node in l3:
        for _ in range(3):
            edges.append((node, idx))
            l4.append(idx)
            idx += 1
            
    # Right Tree
    r1 = [121, 122, 123]
    for node in r1: edges.append((1, node))
    
    r2 = []
    idx = 124
    for node in r1:
        for _ in range(3):
            edges.append((node, idx))
            r2.append(idx)
            idx += 1
            
    r3 = []
    for node in r2:
        for _ in range(3):
            edges.append((node, idx))
            r3.append(idx)
            idx += 1
            
    r4 = []
    for node in r3:
        for _ in range(3):
            edges.append((node, idx))
            r4.append(idx)
            idx += 1
    
    # Build tree data structures
    def build_tree_data(root, level_lists):
        parents = {}
        depth = {root: 0}
        current_level = [root]
        for level_nodes in level_lists:
            next_level = []
            for i, node in enumerate(level_nodes):
                par = current_level[i // 3]
                parents[node] = par
                depth[node] = depth[par] + 1
                next_level.append(node)
            current_level = next_level
        return parents, depth
    
    parents_l, depth_l = build_tree_data(0, [l1, l2, l3, l4])
    parents_r, depth_r = build_tree_data(1, [r1, r2, r3, r4])
    
    # Build LCA with binary lifting
    def build_lca(parents, depth, nodes):
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        max_depth = max(depth.values())
        log_max = max_depth.bit_length()
        
        up = [[-1] * n_nodes for _ in range(log_max)]
        
        for node in nodes:
            idx = node_to_idx[node]
            if node in parents:
                up[0][idx] = node_to_idx[parents[node]]
        
        for j in range(1, log_max):
            for i in range(n_nodes):
                if up[j-1][i] != -1:
                    up[j][i] = up[j-1][up[j-1][i]]
        
        def lca(u, v):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            if depth[u] < depth[v]:
                u_idx, v_idx = v_idx, u_idx
            
            diff = depth[u] - depth[v]
            for j in range(log_max):
                if diff & (1 << j):
                    u_idx = up[j][u_idx]
            
            if u_idx == v_idx:
                return nodes[u_idx]
            
            for j in range(log_max - 1, -1, -1):
                if up[j][u_idx] != up[j][v_idx]:
                    u_idx = up[j][u_idx]
                    v_idx = up[j][v_idx]
            
            return nodes[up[0][u_idx]]
        
        return lca
    
    left_nodes = [0] + l1 + l2 + l3 + l4
    right_nodes = [1] + r1 + r2 + r3 + r4
    lca_l = build_lca(parents_l, depth_l, left_nodes)
    lca_r = build_lca(parents_r, depth_r, right_nodes)
    
    N = 81
    D_L = np.zeros((N, N), dtype=np.int32)
    D_R = np.zeros((N, N), dtype=np.int32)
    
    for i in range(N):
        for j in range(i + 1, N):
            u = l4[i]
            v = l4[j]
            l = lca_l(u, v)
            d = depth_l[u] + depth_l[v] - 2 * depth_l[l]
            D_L[i, j] = d
            D_L[j, i] = d
            
            u = r4[i]
            v = r4[j]
            l = lca_r(u, v)
            d = depth_r[u] + depth_r[v] - 2 * depth_r[l]
            D_R[i, j] = d
            D_R[j, i] = d
            
    # Find Permutation sigma
    # We want min_{i!=j} D_L[i,j] + D_R[sigma[i], sigma[j]] >= 8
    
    best_sigma = None
    best_min_sum = -1
    
    def compute_min_sum_fast(sigma):
        """Compute min sum using numpy vectorization - O(N) via matrix ops."""
        sigma_arr = np.array(sigma)
        # M[i,j] = D_L[i,j] + D_R[sigma[i], sigma[j]] for all i,j
        # We need min over i!=j of M[i,j]
        # Build full matrix: M = D_L + D_R[sigma][:,sigma]
        DR_sel = D_R[np.ix_(sigma_arr, sigma_arr)]
        M = D_L + DR_sel
        # Mask diagonal
        M_masked = M.copy()
        np.fill_diagonal(M_masked, 999)
        return int(M_masked.min())
    
    def compute_min_sum_and_worst(sigma):
        """Compute min sum and identify worst pair (i,j)."""
        sigma_arr = np.array(sigma)
        DR_sel = D_R[np.ix_(sigma_arr, sigma_arr)]
        M = D_L + DR_sel
        M_masked = M.copy()
        np.fill_diagonal(M_masked, 999)
        min_val = int(M_masked.min())
        # Find position of minimum
        flat_idx = np.argmin(M_masked)
        i, j = np.unravel_index(flat_idx, (N, N))
        return min_val, int(i), int(j)
    
    for restart in range(2000):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            sigma = list(range(N))
        elif restart == 1:
            sigma = list(range(N-1, -1, -1))
        elif restart < 10:
            # Structured permutations
            sigma = list(range(N))
            # Reverse blocks
            block_size = max(1, N // (restart + 1))
            for b in range(0, N, block_size):
                sigma[b:b+block_size] = reversed(sigma[b:b+block_size])
        else:
            sigma = list(range(N))
            random.shuffle(sigma)
        
        current_min, worst_i, worst_j = compute_min_sum_and_worst(sigma)
        
        if current_min >= 8:
            if current_min > best_min_sum:
                best_min_sum = current_min
                best_sigma = sigma[:]
            return 242, edges + [(l4[i], r4[sigma[i]]) for i in range(N)]
        
        # Local search with fast swap evaluation
        no_improve = 0
        for step in range(300):
            if time.time() > deadline - 10:
                break
            
            if current_min >= 8:
                if current_min > best_min_sum:
                    best_min_sum = current_min
                    best_sigma = sigma[:]
                return 242, edges + [(l4[i], r4[sigma[i]]) for i in range(N)]
            
            # Try swapping sigma[worst_i] with various k
            # When we swap sigma[i] and sigma[k], the affected entries in M are:
            # M[i,*], M[k,*], M[*,i], M[*,k]
            # Specifically, M[a,b] = D_L[a,b] + D_R[sigma[a], sigma[b]]
            # After swapping sigma[i] <-> sigma[k]:
            #   M[i,b] changes: D_R[sigma[i],sigma[b]] -> D_R[sigma[k],sigma[b]]
            #   M[k,b] changes: D_R[sigma[k],sigma[b]] -> D_R[sigma[i],sigma[b]]
            #   M[a,i] changes: D_R[sigma[a],sigma[i]] -> D_R[sigma[a],sigma[k]]
            #   M[a,k] changes: D_R[sigma[a],sigma[k]] -> D_R[sigma[a],sigma[i]]
            #   M[i,k] and M[k,i] swap their D_R components
            
            i0 = worst_i
            best_k = -1
            best_new_min = current_min - 1
            
            # Candidate k values: worst_j, neighbors, and some random
            candidates = set()
            candidates.add(worst_j)
            if i0 > 0: candidates.add(i0 - 1)
            if i0 < N-1: candidates.add(i0 + 1)
            if worst_j > 0: candidates.add(worst_j - 1)
            if worst_j < N-1: candidates.add(worst_j + 1)
            for _ in range(15):
                candidates.add(random.randint(0, N-1))
            
            for k in candidates:
                if k == i0:
                    continue
                
                # Evaluate effect of swapping sigma[i0] and sigma[k]
                # Only rows i0 and k, and columns i0 and k change
                # M[i0, b] for b != i0,k: D_L[i0,b] + D_R[sigma[k], sigma[b]] (was sigma[i0])
                # M[k, b] for b != i0,k: D_L[k,b] + D_R[sigma[i0], sigma[b]] (was sigma[k])
                # M[a, i0] for a != i0,k: D_L[a,i0] + D_R[sigma[a], sigma[k]] (was sigma[i0])
                # M[a, k] for a != i0,k: D_L[a,k] + D_R[sigma[a], sigma[i0]] (was sigma[k])
                # M[i0,k]: D_L[i0,k] + D_R[sigma[k], sigma[i0]]
                # M[k,i0]: D_L[k,i0] + D_R[sigma[i0], sigma[k]]
                
                # We need the min over all pairs (a,b), a!=b
                # Unaffected pairs keep their old values
                # Affected pairs are those where a in {i0,k} or b in {i0,k}
                
                # Get current min excluding affected region
                sigma_arr = np.array(sigma)
                si = sigma_arr[i0]
                sk = sigma_arr[k]
                
                # Build the affected sub-matrix
                # Rows: i0, k; Cols: all
                # And rows: all; Cols: i0, k
                # Union of affected: {(i0,b) for all b} U {(k,b) for all b} U {(a,i0) for all a} U {(a,k) for all a}
                
                # Min of unaffected: min over a not in {i0,k}, b not in {i0,k} of M[a,b]
                # This is expensive to compute directly. Instead, we can compute:
                # new_min = min( min_unaffected, min_affected_new )
                
                # For speed, let's just compute the full min after swap for a few candidates
                # but use a faster method
                
                sigma[i0], sigma[k] = sigma[k], sigma[i0]
                new_min = compute_min_sum_fast(sigma)
                sigma[i0], sigma[k] = sigma[k], sigma[i0]
                
                if new_min > best_new_min:
                    best_new_min = new_min
                    best_k = k
            
            if best_k >= 0 and best_new_min > current_min:
                sigma[i0], sigma[best_k] = sigma[best_k], sigma[i0]
                current_min = best_new_min
                no_improve = 0
            else:
                # Random perturbation
                a, b = random.sample(range(N), 2)
                sigma[a], sigma[b] = sigma[b], sigma[a]
                current_min, worst_i, worst_j = compute_min_sum_and_worst(sigma)
                no_improve += 1
                if no_improve > 50:
                    break
                
    if best_sigma is not None and best_min_sum >= 8:
        return 242, edges + [(l4[i], r4[best_sigma[i]]) for i in range(N)]
        
    return None, None

def _verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    # Check simple graph
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edge_set:
            return False
        edge_set.add((min(u,v), max(u,v)))
    
    # Girth check: verify no cycle shorter than g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _fallback_construct(k, g):
    # Original program logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
