"""Packing trees T1..T20 into K20 - simulated annealing approach."""
import time
import math
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140
    trees = instance_trees(20, 611)
    n = 20
    num_trees = 20
    
    # Precompute tree edge lists
    tree_edges = []
    for t_idx in range(num_trees):
        k = t_idx + 1
        if k <= 2:
            tree_edges.append([])
        else:
            tree_edges.append(trees[t_idx])
    
    # Total edges in all trees = 190
    total_tree_edges = sum(len(e) for e in tree_edges)
    
    # Build all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Process trees from largest to smallest for better constraint propagation
    tree_order = list(range(num_trees))
    tree_order.sort(key=lambda idx: -(idx + 1))
    
    # For each tree, precompute adjacency and vertex ordering
    tree_info = []
    for t_idx in range(num_trees):
        k = t_idx + 1
        if k <= 2:
            tree_info.append(None)
            continue
        edges = trees[t_idx]
        tree_adj = {v: [] for v in range(k)}
        for u, v in edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        # Order by degree descending, then by vertex index
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append((edges, tree_adj, ordered_verts))
    
    def try_embed_backtrack(t_idx, avail, rng):
        """Try to embed tree t_idx into available edges using backtracking."""
        k = t_idx + 1
        if k == 1:
            return [rng.randint(0, n - 1)]
        if k == 2:
            for u, v in avail:
                return [u, v]
            return None

        tree_edges_list, tree_adj, ordered_verts = tree_info[t_idx]
        host_map = {}
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                return True

            tv = ordered_verts[pos]
            candidates = []

            for h in range(n):
                if h in used_hosts:
                    continue
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in avail:
                            ok = False
                            break
                if ok:
                    remaining = sum(1 for j in range(n) if j != h and (min(h, j), max(h, j)) in avail)
                    candidates.append((remaining, h))

            if not candidates:
                return False

            candidates.sort(key=lambda x: x[0])
            if len(candidates) > 1:
                top_n = min(len(candidates), 5)
                top = candidates[:top_n]
                rng.shuffle(top)
                rest = candidates[top_n:]
                candidates = top + rest

            for _, h in candidates:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    # Main search: try backtracking first, then simulated annealing if needed
    best_result = None
    best_edges_used = -1
    
    # Phase 1: Try backtracking with multiple restarts
    for attempt in range(500):
        if time.time() > deadline:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * num_trees
        success = True

        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            for trial in range(3):
                if time.time() > deadline:
                    success = False
                    break
                embedded = try_embed_backtrack(t_idx, avail, rng)
                if embedded is not None:
                    break

            if not success:
                break
            if embedded is None:
                success = False
                break

            res[t_idx] = embedded
            if k >= 2:
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)

        if success:
            edges_used = len(all_edges) - len(avail)
            if edges_used > best_edges_used:
                best_edges_used = edges_used
                best_result = res
            if edges_used == 190:
                break

    # Phase 2: If we didn't get a perfect packing, try simulated annealing
    if best_edges_used < 190:
        rng = Random(12345)
        T_start = 1.0
        T_end = 0.001
        cooling = 0.9995
        max_iter = 500000
        
        # Initialize with random embeddings
        current = []
        for t_idx in range(num_trees):
            k = t_idx + 1
            if k == 1:
                current.append([rng.randint(0, n - 1)])
            else:
                hosts = rng.sample(range(n), k)
                current.append(hosts)
        
        def compute_penalty(embeddings):
            """Compute penalty: count edge conflicts (edges used more than once)"""
            edge_count = {}
            for t_idx in range(num_trees):
                k = t_idx + 1
                if k <= 2:
                    continue
                hosts = embeddings[t_idx]
                for u, v in trees[t_idx]:
                    hu, hv = hosts[u], hosts[v]
                    edge = (min(hu, hv), max(hu, hv))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            
            penalty = 0
            for edge, count in edge_count.items():
                if count > 1:
                    penalty += count - 1
            
            # Also penalize if not all tree edges are mapped to valid edges
            # (this shouldn't happen since we only use edges from K20)
            
            # Bonus for using more distinct edges
            distinct_edges = len(edge_count)
            penalty -= 0.1 * distinct_edges
            
            return penalty
        
        current_penalty = compute_penalty(current)
        best_penalty = current_penalty
        best_embeddings = [list(e) for e in current]
        
        T = T_start
        iter_count = 0
        
        while T > T_end and iter_count < max_iter:
            if time.time() > deadline:
                break
            
            # Pick a random tree and a random vertex to reassign
            t_idx = rng.randint(0, num_trees - 1)
            k = t_idx + 1
            if k <= 2:
                iter_count += 1
                T *= cooling
                continue
            
            # Pick a random vertex in the tree
            v = rng.randint(0, k - 1)
            old_host = current[t_idx][v]
            
            # Try a new random host
            new_host = rng.randint(0, n - 1)
            if new_host == old_host:
                iter_count += 1
                T *= cooling
                continue
            
            # Check if new_host is already used by another vertex in the same tree
            if new_host in current[t_idx] and current[t_idx].index(new_host) != v:
                iter_count += 1
                T *= cooling
                continue
            
            # Make the move
            current[t_idx][v] = new_host
            new_penalty = compute_penalty(current)
            
            # Accept or reject
            if new_penalty < current_penalty or rng.random() < math.exp(-(new_penalty - current_penalty) / T):
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_embeddings = [list(e) for e in current]
            else:
                current[t_idx][v] = old_host
            
            iter_count += 1
            T *= cooling
        
        # Convert best_embeddings to valid format if penalty is 0
        if best_penalty <= 0.01:
            # Verify the solution
            avail = set(all_edges)
            valid = True
            for t_idx in range(num_trees):
                k = t_idx + 1
                if k <= 2:
                    continue
                hosts = best_embeddings[t_idx]
                # Check distinct hosts
                if len(set(hosts)) != k:
                    valid = False
                    break
                for u, v in trees[t_idx]:
                    hu, hv = hosts[u], hosts[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in avail:
                        valid = False
                        break
                    avail.discard(edge)
                if not valid:
                    break
            
            if valid:
                edges_used = len(all_edges) - len(avail)
                if edges_used > best_edges_used:
                    best_edges_used = edges_used
                    # Build result
                    res = [None] * num_trees
                    for t_idx in range(num_trees):
                        k = t_idx + 1
                        if k == 1:
                            res[t_idx] = [best_embeddings[t_idx][0]]
                        else:
                            res[t_idx] = best_embeddings[t_idx]
                    best_result = res

    if best_result is None:
        # Fallback: return a trivial (invalid but well-formed) result
        return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
