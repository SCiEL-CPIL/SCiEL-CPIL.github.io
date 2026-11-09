"""Find longest simple path in pinned graph using skeleton compression + tree DP + Warnsdorff + time-bounded DFS."""
import time
from collections import deque
import heapq
import sys
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def build_skeleton(comp):
    """Compress degree-2 chains in comp into weighted edges on the core graph.
    Returns:
      core_nodes: list of core vertex ids (degree != 2 in comp, or isolated)
      core_adj: dict core_node -> list of (other_core, chain_length)
      chain_map: dict (core_a, core_b) -> list of vertices in chain (inclusive of both endpoints)
      isolated: list of vertices with degree 0 in comp
    """
    # Degree within comp
    deg = {v: len(ADJ[v] & comp) for v in comp}
    core = set(v for v in comp if deg[v] != 2)
    isolated = [v for v in comp if deg[v] == 0]
    
    if not core:
        # All degree 2: it's a cycle. Longest path = comp_size - 1 (break the cycle)
        return None, None, None, isolated  # signal cycle
    
    core_adj = {v: [] for v in core}
    chain_map = {}
    visited_core = set()
    
    for start in core:
        if start in visited_core:
            continue
        # For each core node, follow each chain to the next core node
        for nbr in ADJ[start] & comp:
            if nbr in visited_core:
                continue
            # Walk the chain from start through nbr until we hit another core node
            chain = [start]
            prev = start
            cur = nbr
            while cur not in core:
                chain.append(cur)
                # Next vertex: the one that's not prev and is in comp
                nxts = [x for x in ADJ[cur] & comp if x != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
            chain.append(cur)
            # cur is now a core node
            key = tuple(sorted((start, cur)))
            if key not in chain_map:
                chain_map[key] = chain
                core_adj[start].append((cur, len(chain) - 1))
                core_adj[cur].append((start, len(chain) - 1))
            visited_core.add(cur)
    
    return sorted(core), core_adj, chain_map, isolated

def warnsdorff_score(v, seen):
    """Prefer vertices with fewer unseen neighbors (Warnsdorff's rule)."""
    return (len(ADJ[v] - seen), v)

def longest_path_on_weighted_tree(core_nodes, core_adj, comp_size):
    """Find longest path in a weighted tree (core graph). Returns (length, path_in_core).
    If core graph is not a tree, fall back to DFS.
    """
    if not core_nodes:
        return 0, []
    
    n = len(core_nodes)
    # Check if it's a tree: n-1 edges
    edge_count = sum(len(core_adj[v]) for v in core_nodes) // 2
    is_tree = (edge_count == n - 1)
    
    if not is_tree:
        # Has cycles; use DFS on the small core
        best = [0, []]
        def dfs_core(path, seen):
            if len(path) > best[0]:
                best[0] = len(path)
                best[1] = list(path)
            for nbr, w in core_adj[path[-1]]:
                if nbr not in seen:
                    path.append(nbr)
                    seen.add(nbr)
                    dfs_core(path, seen)
                    path.pop()
                    seen.remove(nbr)
        for s in core_nodes:
            dfs_core([s], {s})
        # Convert core path length to vertex count: sum of chain lengths along the path
        # Actually we need to expand: the path in core has k nodes, the vertex count is sum of (chain_len - 1) for each edge + 1
        if best[1]:
            vp = best[1]
            total = 1
            for i in range(len(vp) - 1):
                key = tuple(sorted((vp[i], vp[i+1])))
                total += len(chain_map_ref[key]) - 1
            return total, vp
        return 0, []
    
    # Tree DP: for each node, compute the two longest downward paths
    # longest_path = max over all nodes of (top1 + top2) where top1, top2 are the two longest chains from that node
    parent = {v: -1 for v in core_nodes}
    order = []
    # BFS from arbitrary root
    root = core_nodes[0]
    queue = deque([root])
    parent[root] = -2
    while queue:
        u = queue.popleft()
        order.append(u)
        for nbr, w in core_adj[u]:
            if parent[nbr] == -1:
                parent[nbr] = u
                queue.append(nbr)
    
    # DP: down[v] = longest path starting at v going down (sum of weights)
    down = {v: 0 for v in core_nodes}
    best = [0, []]
    
    for u in reversed(order):
        # Collect children's contributions
        child_vals = []
        for nbr, w in core_adj[u]:
            if parent[nbr] == u:
                child_vals.append((w + down[nbr], nbr))
        child_vals.sort(reverse=True)
        if child_vals:
            down[u] = child_vals[0][0]
        # Path through u: top1 + top2
        if len(child_vals) >= 2:
            total = child_vals[0][0] + child_vals[1][0]
            if total > best[0]:
                best[0] = total
                # Reconstruct path: go up from u through child_vals[0]'s child, and through child_vals[1]'s child
                # We'll store the two endpoints
                best[1] = (u, child_vals[0][1], child_vals[1][1])
        elif len(child_vals) == 1:
            total = child_vals[0][0]
            if total > best[0]:
                best[0] = total
                best[1] = (u, child_vals[0][1], None)
    
    # Also check single-node paths (shouldn't be better than above for trees with edges)
    # Reconstruct the actual core path from the best
    if not best[1]:
        return 0, core_nodes[:1]
    
    u, c1, c2 = best[1]
    # Build path from c1 up to u, then from u down to c2
    def path_to_ancestor(desc, anc):
        p = [desc]
        while desc != anc:
            desc = parent[desc]
            p.append(desc)
        return p  # desc -> anc
    
    if c1 is not None and c2 is not None:
        p1 = path_to_ancestor(c1, u)
        p2 = path_to_ancestor(c2, u)
        core_path = p1 + p2[1:]
    elif c1 is not None:
        core_path = path_to_ancestor(c1, u)
    else:
        core_path = [u]
    
    # Convert to vertex count
    total = 1
    for i in range(len(core_path) - 1):
        key = tuple(sorted((core_path[i], core_path[i+1])))
        total += len(chain_map_ref[key]) - 1
    return total, core_path

def expand_core_path(core_path, chain_map):
    """Expand a core path into a full vertex path."""
    if not core_path:
        return []
    if len(core_path) == 1:
        return list(core_path)
    full = []
    for i in range(len(core_path) - 1):
        key = tuple(sorted((core_path[i], core_path[i+1])))
        chain = chain_map[key]
        # Ensure chain goes from core_path[i] to core_path[i+1]
        if chain[0] != core_path[i]:
            chain = chain[::-1]
        if not full:
            full.extend(chain)
        else:
            full.extend(chain[1:])
    return full

def greedy_path_from(start, seen=None):
    """Greedy Warnsdorff path from start."""
    if seen is None:
        seen = {start}
    else:
        seen = set(seen)
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: warnsdorff_score(w, seen))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen):
    """Extend path greedily from both ends."""
    path = list(path)
    seen = set(seen)
    changed = True
    while changed:
        changed = False
        # Extend front
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def time_bounded_dfs(start, comp_set, deadline, best_path, best_len):
    """Time-bounded DFS to find longest path starting from start within comp."""
    # Use a list to track best found
    best = [best_path[:], best_len]
    
    def dfs(path, seen):
        if time.time() > deadline:
            return
        if len(path) > best[1]:
            best[0] = path[:]
            best[1] = len(path)
        # If we've reached the full component, we're done
        if len(path) == len(comp_set):
            return
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            return
        # Sort by Warnsdorff heuristic
        cand_sorted = sorted(cand, key=lambda w: warnsdorff_score(w, seen))
        for v in cand_sorted:
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
    
    dfs([start], {start})
    return best[0]

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    if not comps:
        return [0]
    
    best_path = []
    
    for largest_comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(largest_comp)
        comp_list = sorted(largest_comp)
        
        # Skeleton approach
        core_nodes, core_adj, chain_map, isolated = build_skeleton(largest_comp)
        
        if core_nodes is None:
            # It's a pure cycle: longest path = comp_size - 1
            path = comp_list  # any ordering works for a cycle minus one edge
            # Actually we need a valid path: just take all vertices in cycle order
            # The cycle is comp_list in some order; let's just use all vertices
            # For a cycle, any Hamiltonian path is valid
            if len(path) > len(best_path):
                best_path = list(path)
            continue
        
        # Try skeleton-based longest path
        global chain_map_ref
        chain_map_ref = chain_map
        if core_nodes and core_adj:
            skel_len, skel_core_path = longest_path_on_weighted_tree(core_nodes, core_adj, comp_size)
            if skel_core_path:
                full_path = expand_core_path(skel_core_path, chain_map)
                # Validate
                valid = True
                for i in range(len(full_path) - 1):
                    if full_path[i+1] not in ADJ[full_path[i]]:
                        valid = False
                        break
                if valid and len(set(full_path)) == len(full_path):
                    if len(full_path) > len(best_path):
                        best_path = full_path
        
        # Phase 1: Greedy + extend from every vertex in largest component
        for start in comp_list:
            if time.time() > deadline:
                break
            p = greedy_path_from(start)
            p = extend_both_ends(p, set(p))
            p = [v for v in p if v in largest_comp]
            if len(p) > len(best_path):
                best_path = list(p)
        
        # If we found Hamiltonian path in largest component, done
        if len(best_path) >= comp_size:
            for i in range(len(best_path) - 1):
                if best_path[i+1] not in ADJ[best_path[i]]:
                    break
            else:
                return best_path
        
        # Phase 2: Time-bounded DFS from promising starts
        promising_starts = sorted(comp_list, key=lambda x: (-len(ADJ[x] & largest_comp), x))
        
        if best_path:
            for v in [best_path[0], best_path[-1], best_path[len(best_path)//2]]:
                if v not in promising_starts:
                    promising_starts.insert(0, v)
        
        for start in promising_starts:
            if time.time() > deadline:
                break
            if len(best_path) >= comp_size:
                break
            
            remaining = deadline - time.time()
            if remaining < 0.5:
                break
            budget = min(3.0, remaining * 0.4)
            dl = time.time() + budget
            
            local_best = time_bounded_dfs(start, largest_comp, dl, best_path, len(best_path))
            if len(local_best) > len(best_path):
                best_path = local_best
        
        # Phase 3: Try to extend the best path found
        if best_path:
            best_path = extend_both_ends(best_path, set(best_path))
            best_path = [v for v in best_path if v in largest_comp]
    
    # Final validation
    if not best_path:
        best_path = [0]
    
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path
