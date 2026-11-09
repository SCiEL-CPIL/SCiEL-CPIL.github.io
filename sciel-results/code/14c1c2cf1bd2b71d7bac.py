import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    def check_time():
        return time.time() > deadline

    # ---- Find connected components ----
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited_comp[v]:
                        visited_comp[v] = True
                        stack.append(v)
            components.append(comp)

    best_path = []

    def update_best(path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]

    # ---- BFS diameter (optimal for trees; good baseline for general graphs) ----
    def bfs_diameter_path(comp, local_adj):
        if not comp:
            return []
        comp_set = set(comp)

        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([start])
            farthest = start
            while queue:
                u = queue.popleft()
                for v in local_adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                        if dist[v] > dist[farthest]:
                            farthest = v
            return farthest, parent

        a, _ = bfs_farthest(comp[0])
        b, parent = bfs_farthest(a)
        path = [b]
        cur = b
        while cur != a:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    # ---- Skeleton decomposition: compress degree-2 chains into weighted edges ----
    def build_skeleton(comp):
        comp_set = set(comp)
        local_adj = {u: [v for v in adj[u] if v in comp_set] for u in comp}
        deg = {u: len(local_adj[u]) for u in comp}

        # Branching nodes: degree >= 3 (or degree-0 isolated). Degree-2 nodes are "chain" nodes.
        branching = [u for u in comp if deg[u] >= 3]
        chain_nodes = [u for u in comp if deg[u] == 2]

        # Build skeleton: nodes = branching nodes (plus isolated deg-0/1 endpoints handled below)
        # For each branching node, find the chain going out to the next branching node.
        # A chain is a maximal path of degree-2 nodes between two branching nodes,
        # or from a branching node to a leaf (degree 1), or a pure cycle of degree-2 nodes.

        # Map: for each branching node, list of (neighbor_branching_or_leaf, chain_vertices, chain_length)
        # chain_length = number of vertices on the chain (including the branching nodes? we count internal)
        # Let's define: edge weight = number of vertices along the path between the two endpoints
        # (inclusive of endpoints in the full path, but for skeleton we store the internal chain).

        # For each branching node, for each neighbor:
        #   if neighbor is branching: direct edge, weight = 1 (just the two nodes, 0 internal)
        #   if neighbor is chain node (deg 2): walk the chain until we hit a branching or leaf
        skeleton_adj = {u: [] for u in branching}  # list of (other_node, internal_chain_list, weight)
        weight = {u: 0 for u in branching}  # not used directly

        # Also track leaves (degree 1) and their chain back to a branching node
        leaves = [u for u in comp if deg[u] == 1]

        processed_edges = set()  # to avoid double-processing

        for b in branching:
            for nb in local_adj[b]:
                key = (min(b, nb), max(b, nb))
                if key in processed_edges:
                    continue
                processed_edges.add(key)

                if deg[nb] >= 3:
                    # Direct branching-branching edge
                    skeleton_adj[b].append((nb, [], 1))
                    skeleton_adj[nb].append((b, [], 1))
                elif deg[nb] == 2:
                    # Walk the chain starting from b -> nb -> ... until hitting branching or leaf
                    chain = [nb]
                    prev = b
                    cur = nb
                    while True:
                        nxt_candidates = [x for x in local_adj[cur] if x != prev]
                        if not nxt_candidates:
                            break
                        nxt = nxt_candidates[0]
                        if deg[nxt] >= 3:
                            # Hit a branching node
                            skeleton_adj[b].append((nxt, chain, len(chain) + 1))
                            skeleton_adj[nxt].append((b, chain[:], len(chain) + 1))
                            break
                        elif deg[nxt] == 1:
                            # Hit a leaf
                            chain.append(nxt)
                            # This chain ends at a leaf. We represent the leaf as a pseudo-node.
                            # Add a pseudo-node for the leaf.
                            leaf_id = ('leaf', nxt)
                            skeleton_adj[b].append((leaf_id, chain, len(chain) + 1))
                            break
                        else:
                            # deg 2, continue
                            chain.append(nxt)
                            prev = cur
                            cur = nxt
                            if check_time():
                                break
                else:
                    # deg 0 or deg 1 directly adjacent (leaf)
                    if deg[nb] == 1:
                        leaf_id = ('leaf', nb)
                        skeleton_adj[b].append((leaf_id, [nb], 2))
                    # deg 0: isolated, ignore

        # Handle pure cycles of degree-2 nodes (no branching node in the cycle)
        # Find unvisited degree-2 nodes that form a cycle
        visited_chain = set()
        for u in chain_nodes:
            if u in visited_chain:
                continue
            # Walk the cycle
            cycle = [u]
            visited_chain.add(u)
            prev = None
            cur = u
            while True:
                nxt_candidates = [x for x in local_adj[cur] if x != prev]
                if not nxt_candidates:
                    break
                nxt = nxt_candidates[0]
                if nxt == u:
                    # Closed the cycle
                    break
                if deg[nxt] >= 3 or deg[nxt] <= 1:
                    # Shouldn't happen if it's a pure cycle, but break to be safe
                    break
                cycle.append(nxt)
                visited_chain.add(nxt)
                prev = cur
                cur = nxt
                if check_time():
                    break
            # This is a cycle of degree-2 nodes. The longest path in a cycle of length L is L (all nodes).
            # Represent as a pseudo-cycle node.
            cycle_id = ('cycle', tuple(cycle))
            # Add as a self-contained component; we'll handle it separately.
            if len(cycle) > len(best_path):
                update_best(cycle)

        return branching, skeleton_adj, local_adj, deg

    # ---- Solve longest path on skeleton via DFS over branching nodes ----
    def solve_skeleton(branching, skeleton_adj, local_adj, deg, comp):
        if not branching:
            return []

        # Map branching node ids to indices
        bnodes = list(branching)
        bidx = {b: i for i, b in enumerate(bnodes)}
        m = len(bnodes)

        # Build skeleton edges: for each branching node, list of (neighbor_id, internal_chain, weight)
        # neighbor_id can be a branching node index or a pseudo-node ('leaf', v) or ('cycle', ...)
        # We'll treat leaf pseudo-nodes as terminal nodes in the skeleton.

        # For DFS, we track which branching nodes are visited.
        # A path in the skeleton is a sequence of branching nodes (and possibly leaf endpoints).
        # The total vertex count = sum of weights along the path.

        # To handle leaf endpoints: a leaf is attached to exactly one branching node,
        # so it can only be used at the end of a path.

        # DFS state: (current_branching, visited_set, current_path_vertices_count, current_vertex_sequence)
        # We want to maximize current_path_vertices_count.

        best_local = []

        def dfs_skel(cur_b, visited, count, seq):
            nonlocal best_local
            if check_time():
                return
            if len(seq) > len(best_local):
                best_local = seq[:]

            # Try extending to neighboring branching nodes
            for (nb, chain, w) in skeleton_adj[cur_b]:
                if isinstance(nb, tuple):
                    # Leaf endpoint: can only be used once, and only as an end
                    # Since we're extending from cur_b, adding a leaf makes this the end of the path.
                    # A leaf is always safe to add (it's degree 1, so no further extension).
                    new_seq = seq + chain + [nb[1]] if chain else seq + [nb[1]]
                    # Actually chain already includes the leaf? Let's check:
                    # In build_skeleton, for a leaf hit: chain.append(nxt) where nxt is the leaf.
                    # So chain includes the leaf vertex. weight = len(chain)+1 includes cur_b? No.
                    # Let me re-examine: weight = len(chain) + 1. The +1 accounts for... 
                    # Actually, the path from cur_b to the leaf goes: cur_b -> chain[0] -> ... -> chain[-1] (leaf).
                    # Number of vertices = 1 (cur_b) + len(chain). But cur_b is already in seq.
                    # So new vertices = len(chain).
                    new_seq = seq + chain
                    if len(new_seq) > len(best_local):
                        best_local = new_seq[:]
                    continue

                # nb is a branching node
                if nb in visited:
                    continue
                visited.add(nb)
                new_seq = seq + chain + [nb]
                dfs_skel(nb, visited, count + w, new_seq)
                visited.remove(nb)
                if check_time():
                    return

        # Try starting from each branching node
        # Sort by degree in skeleton (fewer neighbors first) for better pruning
        bnodes_sorted = sorted(bnodes, key=lambda b: len(skeleton_adj[b]))
        for start in bnodes_sorted:
            if check_time():
                break
            visited = {start}
            dfs_skel(start, visited, 1, [start])
            if len(best_local) >= len(comp):
                break  # Hamiltonian path found

        return best_local

    # ---- IDDFS with Warnsdorff for general graphs (refinement) ----
    def iddfs_warnsdorff(comp, local_adj, max_iter=None):
        if max_iter is None:
            max_iter = len(comp)
        comp_set = set(comp)
        best_local = []

        def unvisited_degree(v, visited):
            return sum(1 for w in local_adj[v] if w not in visited)

        for target_len in range(max_iter, 1, -1):
            if check_time():
                break
            found = False
            starts = sorted(comp, key=lambda x: len(local_adj[x]))
            for start in starts:
                if found or check_time():
                    break

                def dfs_iter(u, visited, path, remaining):
                    nonlocal found
                    if check_time():
                        return True
                    if len(path) == target_len:
                        if len(path) > len(best_local):
                            best_local = path[:]
                        found = True
                        return True
                    if remaining == 0:
                        return False
                    neighbors = [v for v in local_adj[u] if v not in visited]
                    if not neighbors:
                        return False
                    if len(path) + remaining < target_len:
                        return False
                    neighbors.sort(key=lambda v: unvisited_degree(v, visited))
                    for v in neighbors:
                        visited.add(v)
                        path.append(v)
                        result = dfs_iter(v, visited, path, remaining - 1)
                        path.pop()
                        visited.remove(v)
                        if result:
                            return True
                    return False

                visited = {start}
                path = [start]
                dfs_iter(start, visited, path, target_len - 1)
            if found:
                break
        return best_local

    # ---- Process each component ----
    for comp in sorted(components, key=len, reverse=True):
        if check_time():
            break

        local_adj = {u: [v for v in adj[u] if v in set(comp)] for u in comp}
        deg = {u: len(local_adj[u]) for u in comp}

        # Baseline: BFS diameter
        diam_path = bfs_diameter_path(comp, local_adj)
        update_best(diam_path)

        # Check if tree
        edge_count = sum(deg[u] for u in comp) // 2
        is_tree = (edge_count == len(comp) - 1)

        if is_tree:
            # BFS diameter is optimal for trees
            continue

        # Try skeleton approach
        branching, skeleton_adj, _, _ = build_skeleton(comp)
        if branching:
            skel_path = solve_skeleton(branching, skeleton_adj, local_adj, deg, comp)
            update_best(skel_path)

        # If skeleton didn't find a Hamiltonian path and component is small enough, try IDDFS
        if len(best_path) < len(comp) and len(comp) <= 30:
            iddfs_path = iddfs_warnsdorff(comp, local_adj)
            update_best(iddfs_path)

        if len(best_path) >= len(comp):
            break  # Hamiltonian path found

    # ---- Final refinement: try to extend the best path ----
    if best_path and len(best_path) < n:
        path_set = set(best_path)

        def try_extend(end_vertex, path, is_front):
            for v in adj[end_vertex]:
                if v not in path_set:
                    new_path = [v] + path if is_front else path + [v]
                    return new_path
            return None

        extended = try_extend(best_path[0], best_path, True)
        if extended:
            best_path = extended
        extended = try_extend(best_path[-1], best_path, False)
        if extended:
            best_path = extended

    if not best_path:
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]

    return best_path
# EVOLVE-BLOCK-END
