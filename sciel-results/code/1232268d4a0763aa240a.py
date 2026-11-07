"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 130
    trees = instance_trees(20, 611)
    n = 20

    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Precompute tree structures
    tree_info = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': {}, 'verts': [0], 'ordered': [0]})
            continue
        tree_verts = set()
        for u, v in tree_edges:
            tree_verts.add(u)
            tree_verts.add(v)
        tree_verts = sorted(tree_verts)
        tree_adj = {v: [] for v in tree_verts}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        ordered_tv = sorted(tree_verts, key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append({
            'k': k,
            'edges': tree_edges,
            'verts': tree_verts,
            'adj': tree_adj,
            'ordered': ordered_tv
        })

    def embed_tree(t_idx, available_edges, rng, deadline):
        """Embed tree t_idx into available_edges. Returns list of host vertices or None."""
        if time.time() > deadline:
            return None
        info = tree_info[t_idx]
        k = info['k']
        if k == 1:
            return [rng.randint(0, n - 1)]
        tree_adj = info['adj']
        ordered_tv = info['ordered']
        tree_edges = info['edges']

        # Build host adjacency from available edges
        host_adj = {h: set() for h in range(n)}
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)

        host_map = {}
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(ordered_tv):
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True

            tv = ordered_tv[pos]
            tv_neighbors = tree_adj[tv]
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors if nb in host_map]

            if assigned_neighbors:
                candidates = set(host_adj[assigned_neighbors[0][1]])
                for _, hn in assigned_neighbors[1:]:
                    candidates &= host_adj[hn]
            else:
                candidates = set(range(n))
            candidates -= used_hosts

            if not candidates:
                return False

            # Sort by degree in available graph, descending
            candidates_list = sorted(candidates, key=lambda h: -len(host_adj[h]))

            # Add randomness for diversity
            if len(candidates_list) > 4:
                top4 = candidates_list[:4]
                rest = candidates_list[4:]
                rng.shuffle(rest)
                candidates_list = top4 + rest
            elif len(candidates_list) > 1:
                rng.shuffle(candidates_list)

            for h in candidates_list:
                host_map[tv] = h
                used_hosts.add(h)

                # Forward checking
                valid = True
                for w in tv_neighbors:
                    if w not in host_map:
                        other_assigned = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                        if other_assigned:
                            inter = set(host_adj[h])
                            for oh in other_assigned:
                                inter &= host_adj[oh]
                            inter -= used_hosts
                        else:
                            inter = set(host_adj[h])
                            inter -= used_hosts
                        if not inter:
                            valid = False
                            break

                if valid:
                    if backtrack(pos + 1):
                        return True

                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    def check_and_embed_t20(remaining_edges, rng, deadline):
        """Check if remaining edges form a tree isomorphic to T20, and embed it."""
        if time.time() > deadline:
            return None
        if len(remaining_edges) != 19:
            return None

        # Build adjacency of remaining graph
        rem_adj = {h: set() for h in range(n)}
        for (a, b) in remaining_edges:
            rem_adj[a].add(b)
            rem_adj[b].add(a)

        # Check if it's a tree: connected and exactly 19 edges (already checked)
        # BFS from vertex 0
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            u = queue.pop(0)
            for v in rem_adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        if len(visited) != 20:
            return None

        # Now find isomorphic embedding of T20 into this tree
        t20_info = tree_info[19]
        t20_adj = t20_info['adj']
        t20_ordered = t20_info['ordered']
        t20_edges = t20_info['edges']

        # Use backtracking to find isomorphism
        host_map = {}
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(t20_ordered):
                for u, v in t20_edges:
                    hu, hv = host_map[u], host_map[v]
                    if hv not in rem_adj[hu]:
                        return False
                return True

            tv = t20_ordered[pos]
            tv_neighbors = t20_adj[tv]
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors if nb in host_map]

            if assigned_neighbors:
                candidates = set(rem_adj[assigned_neighbors[0][1]])
                for _, hn in assigned_neighbors[1:]:
                    candidates &= rem_adj[hn]
            else:
                candidates = set(range(n))
            candidates -= used_hosts

            if not candidates:
                return False

            candidates_list = sorted(candidates, key=lambda h: -len(rem_adj[h]))
            if len(candidates_list) > 3:
                top3 = candidates_list[:3]
                rest = candidates_list[3:]
                rng.shuffle(rest)
                candidates_list = top3 + rest
            elif len(candidates_list) > 1:
                rng.shuffle(candidates_list)

            for h in candidates_list:
                host_map[tv] = h
                used_hosts.add(h)

                valid = True
                for w in tv_neighbors:
                    if w not in host_map:
                        other_assigned = [host_map[nb] for nb in t20_adj[w] if nb in host_map and nb != tv]
                        if other_assigned:
                            inter = set(rem_adj[h])
                            for oh in other_assigned:
                                inter &= rem_adj[oh]
                            inter -= used_hosts
                        else:
                            inter = set(rem_adj[h])
                            inter -= used_hosts
                        if not inter:
                            valid = False
                            break

                if valid:
                    if backtrack(pos + 1):
                        return True

                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(20)]
        return None

    best_result = None
    best_edges = 0

    # Try multiple strategies
    for attempt in range(500):
        if time.time() > deadline:
            break

        rng = Random(attempt * 104729 + 1)
        available = set(all_edges)
        result = [None] * 20
        success = True

        # Tree order: largest first, with some randomization
        tree_order = list(range(20))
        # Keep T20 (index 19) last, T1 (index 0) can be anywhere
        # Sort T2..T19 by size descending, then randomize ties
        mid_trees = [i for i in range(20) if i != 0 and i != 19]
        mid_trees.sort(key=lambda idx: (-len(trees[idx]), idx))
        # Shuffle within same size groups for diversity
        i = 0
        while i < len(mid_trees):
            j = i
            while j < len(mid_trees) and len(trees[mid_trees[j]]) == len(trees[mid_trees[i]]):
                j += 1
            rng.shuffle(mid_trees[i:j])
            i = j

        # Place T1 and T20
        t1_pos = rng.randint(0, len(mid_trees))
        tree_order = mid_trees[:t1_pos] + [0] + mid_trees[t1_pos:] + [19]

        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break

            k = t_idx + 1
            if k == 1:
                result[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = embed_tree(t_idx, available, rng, deadline)

            if embedded is None:
                success = False
                break

            result[t_idx] = embedded
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)

        if success:
            # All trees embedded!
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges == 190:
                best_result = result
                break

    if best_result is None:
        # Try partial solutions
        for attempt in range(300):
            if time.time() > deadline:
                break
            rng = Random(attempt * 999999 + 7)
            available = set(all_edges)
            result = [None] * 20
            current_edges = 0

            tree_order = list(range(20))
            tree_order.sort(key=lambda idx: (-len(trees[idx]), idx))

            for t_idx in tree_order:
                if time.time() > deadline:
                    break
                k = t_idx + 1
                if k == 1:
                    result[t_idx] = [rng.randint(0, n - 1)]
                    continue
                embedded = embed_tree(t_idx, available, rng, deadline)
                if embedded is None:
                    break
                result[t_idx] = embedded
                current_edges += len(trees[t_idx])
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    available.discard(edge)

            if current_edges > best_edges:
                best_edges = current_edges
                best_result = result
                if best_edges == 190:
                    break

    if best_result is None:
        # Fallback: just return T20 as spanning tree, others None
        result = [None] * 20
        result[19] = list(range(20))
        return result

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
