"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random
import numpy as np

def construct():
    _t0 = time.time()
    _DEADLINE = 135.0  # leave 15s margin

    def _instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _build_conflict_bitmasks(edges, adj, n_edges):
        """Build numpy bitmasks for conflict checking.
        conflict_mask[i] is a bitmask where bit j is set if edge i conflicts with edge j.
        """
        if n_edges <= 64:
            dtype = np.uint64
            n_words = 1
        elif n_edges <= 128:
            dtype = np.uint64
            n_words = 2
        else:
            dtype = np.uint64
            n_words = (n_edges + 63) // 64
        
        masks = np.zeros((n_edges, n_words), dtype=dtype)
        
        for i in range(n_edges):
            u1, v1 = edges[i]
            for j in range(i + 1, n_edges):
                u2, v2 = edges[j]
                # Check conflict: share vertex or joined by edge
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    masks[i, j // 64] |= (1 << (j % 64))
                    masks[j, i // 64] |= (1 << (i % 64))
                elif u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    masks[i, j // 64] |= (1 << (j % 64))
                    masks[j, i // 64] |= (1 << (i % 64))
        
        return masks, n_words

    def _dsatur_color(edges, adj, masks, n_words, n_edges):
        """DSATUR algorithm for strong edge coloring.
        At each step, pick the uncolored edge with the most saturated color set,
        then assign the lowest available color.
        """
        color = [-1] * n_edges
        sat = [set() for _ in range(n_edges)]
        colored = [False] * n_edges
        
        # Precompute uncolored neighbor count for tie-breaking
        # For each edge, count how many other edges conflict with it
        conflict_degree = [0] * n_edges
        for i in range(n_edges):
            for w in range(n_words):
                word = masks[i, w]
                conflict_degree[i] = np.bit_count(word)
        
        for step in range(n_edges):
            # Pick uncolored edge with max saturation, tie-break by conflict degree
            best_i = -1
            best_sat = -1
            best_deg = -1
            for i in range(n_edges):
                if colored[i]:
                    continue
                s = len(sat[i])
                if s > best_sat or (s == best_sat and conflict_degree[i] > best_deg):
                    best_sat = s
                    best_deg = conflict_degree[i]
                    best_i = i
            
            if best_i == -1:
                break
            
            # Find lowest available color
            forbidden = set()
            for j in range(n_edges):
                if colored[j] and (masks[best_i, j // 64] & (1 << (j % 64))):
                    forbidden.add(color[j])
            
            c = 0
            while c in forbidden:
                c += 1
            
            color[best_i] = c
            colored[best_i] = True
            
            # Update saturation for all uncolored edges that conflict with best_i
            for j in range(n_edges):
                if not colored[j] and (masks[best_i, j // 64] & (1 << (j % 64))):
                    sat[j].add(c)
        
        return color

    def _sa_color_reduce(colours, edges, adj, masks, n_words, n_edges, time_limit=10.0):
        """Simulated annealing to reduce color count."""
        t_start = time.time()
        if n_edges == 0:
            return colours
        
        col = colours[:]
        max_c = max(col) + 1 if col else 0
        
        # Color class membership
        classes = [set() for _ in range(max_c)]
        for i, c in enumerate(col):
            classes[c].add(i)
        
        best_col = col[:]
        best_nc = max_c
        T = 3.0
        cooling = 0.999
        min_T = 0.05
        rng = Random(12345)
        
        while T > min_T and time.time() - t_start < time_limit:
            # Pick a random edge
            i = rng.randrange(n_edges)
            old_c = col[i]
            
            # Pick a new color
            if rng.random() < 0.2:
                new_c = max_c  # try new color (explore)
            else:
                new_c = rng.randrange(max_c)
            
            if new_c == old_c:
                T *= cooling
                continue
            
            # Ensure classes list is large enough
            while len(classes) <= new_c:
                classes.append(set())
            
            # Remove i from old_c class
            classes[old_c].discard(i)
            
            # Check conflicts: how many edges in new_c conflict with i?
            conflicts_new = 0
            for j in classes[new_c]:
                if (masks[i, j // 64] & (1 << (j % 64))):
                    conflicts_new += 1
            
            # How many edges in old_c (now without i) conflict with i?
            conflicts_old = 0
            for j in classes[old_c]:
                if (masks[i, j // 64] & (1 << (j % 64))):
                    conflicts_old += 1
            
            delta = conflicts_new - conflicts_old
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                # Accept move
                col[i] = new_c
                classes[new_c].add(i)
                
                # Try to reduce max_c: check if top colors are empty
                while max_c > 0 and len(classes[max_c - 1]) == 0:
                    max_c -= 1
                
                # Update best
                current_nc = max_c
                if current_nc < best_nc:
                    best_nc = current_nc
                    best_col = col[:]
            else:
                # Reject: put i back in old_c
                classes[old_c].add(i)
            
            T *= cooling
        
        return best_col

    def _local_search_reduce(colours, edges, adj, masks, n_words, n_edges, time_limit=5.0):
        """Greedy local search: repeatedly move edges to lower colors if possible."""
        t_start = time.time()
        if n_edges == 0:
            return colours
        
        col = colours[:]
        max_c = max(col) + 1 if col else 0
        classes = [set() for _ in range(max_c)]
        for i, c in enumerate(col):
            classes[c].add(i)
        
        improved = True
        while improved and time.time() - t_start < time_limit:
            improved = False
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                for c in range(cur_c):
                    # Check if i can move to color c
                    ok = True
                    for j in classes[c]:
                        if (masks[i, j // 64] & (1 << (j % 64))):
                            ok = False
                            break
                    if ok:
                        classes[cur_c].discard(i)
                        classes[c].add(i)
                        col[i] = c
                        improved = True
                        break
            
            # Compact
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    max_c = len(present)
                    new_classes = [set() for _ in range(max_c)]
                    for i, c in enumerate(col):
                        new_classes[c].add(i)
                    classes = new_classes
        
        return col

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = _build_adj(edges)
        n_edges = len(edges)
        
        if n_edges == 0:
            out.append([])
            continue
        
        time_left = _DEADLINE - time.time()
        if time_left < 3:
            # Fallback: simple greedy
            color = [-1] * n_edges
            for i in range(n_edges):
                color[i] = i  # worst case: unique color
            out.append(color)
            continue
        
        # Build conflict masks
        masks, n_words = _build_conflict_bitmasks(edges, adj, n_edges)
        
        # DSATUR initial coloring
        dsatur_time = min(15, time_left * 0.4)
        colours = _dsatur_color(edges, adj, masks, n_words, n_edges)
        
        # Local search reduction
        ls_time = min(10, _DEADLINE - time.time() - 15)
        if ls_time > 1:
            colours = _local_search_reduce(colours, edges, adj, masks, n_words, n_edges, time_limit=ls_time)
        
        # Simulated annealing
        sa_time = _DEADLINE - time.time() - 5
        if sa_time > 2:
            colours = _sa_color_reduce(colours, edges, adj, masks, n_words, n_edges, time_limit=sa_time)
        
        out.append(colours)

    return out
# EVOLVE-BLOCK-END
