import time
import random
from collections import deque
import sys

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    # 150s limit. Use 145s as deadline to be safe.
    deadline = start_time + 145
    
    trees = instance_trees(20, 611)
    
    # Precompute tree properties
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'num_edges': len(edges)
        })
    
    # Sort by size descending (largest first) to constrain early
    order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    # Precompute a DFS order for each tree.
    # We want an order where parents come before children.
    # Using DFS ensures that when we place a node, its parent is already placed.
    # We'll use a specific root (0) and traverse.
    def get_dfs_order(adj, k, root=0):
        visited = [False] * k
        order = []
        def dfs(u):
            visited[u] = True
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
        dfs(root)
        return order

    dfs_orders = []
    dfs_pos = []
    for i in range(20):
        k = tree_info[i]['k']
        if k == 1:
            dfs_orders.append([0])
            dfs_pos.append([0])
        else:
            ord_list = get_dfs_order(tree_info[i]['adj'], k, 0)
            dfs_orders.append(ord_list)
            pos_map = [-1] * k
            for j, v in enumerate(ord_list):
                pos_map[v] = j
            dfs_pos.append(pos_map)

    def find_embedding(t_idx, used_edges, rng):
        """
        Finds a valid embedding for tree t_idx into K20 avoiding used_edges.
        Returns list of length k where mapping[i] is host vertex for tree vertex i.
        """
        info = tree_info[t_idx]
        k = info['k']
        adj_t = info['adj']
        deg_t = info['deg']
        dfs_order = dfs_orders[t_idx]
        pos_map = dfs_pos[t_idx]
        
        if k == 1:
            return [0]

        mapping = [-1] * k
        used_hosts = set()
        
        def count_available_hosts_for_neighbor(nb, current_mapping, current_used_hosts, current_used_edges):
            """Count how many host vertices are available for tree vertex nb."""
            count = 0
            for h in range(20):
                if h in current_used_hosts:
                    continue
                valid = True
                for mnb in adj_t[nb]:
                    if pos_map[mnb] < pos_map[nb]:  # already mapped
                        hn = current_mapping[mnb]
                        if hn == -1:
                            continue
                        e = (min(h, hn), max(h, hn))
                        if e in current_used_edges:
                            valid = False
                            break
                if valid:
                    count += 1
            return count
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == k:
                return True
            
            tree_v = dfs_order[pos]
            
            # Identify neighbors of tree_v that are already mapped (parents in DFS tree)
            mapped_neighbors = []
            for nb in adj_t[tree_v]:
                if pos_map[nb] < pos:
                    mapped_neighbors.append(nb)
            
            # Candidates: Host vertices not in used_hosts
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                valid = True
                for nb in mapped_neighbors:
                    hn = mapping[nb]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
                if valid:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Heuristic: Score each candidate by how many options it leaves for unmapped neighbors
            # This is a "least constraining value" heuristic
            scored_candidates = []
            for h in candidates:
                # Count available hosts for each unmapped neighbor of tree_v
                total_options = 0
                for nb in adj_t[tree_v]:
                    if pos_map[nb] >= pos:  # unmapped
                        # Check if h can connect to nb's potential hosts
                        # For now, just count available hosts for nb
                        count = 0
                        for h2 in range(20):
                            if h2 in used_hosts or h2 == h:
                                continue
                            # Check edge h-h2 is not used
                            e = (min(h, h2), max(h, h2))
                            if e not in used_edges:
                                count += 1
                        total_options += count
                scored_candidates.append((total_options, h))
            
            # Sort by descending total_options (most options first), with random tie-breaking
            rng.shuffle(scored_candidates)
            scored_candidates.sort(key=lambda x: -x[0])
            
            for _, h in scored_candidates:
                mapping[tree_v] = h
                used_hosts.add(h)
                
                # Forward Checking: verify unmapped neighbors still have options
                fc_valid = True
                for nb in adj_t[tree_v]:
                    if pos_map[nb] >= pos:  # unmapped
                        # Check if nb has at least one available host
                        has_option = False
                        for h2 in range(20):
                            if h2 in used_hosts:
                                continue
                            e = (min(h, h2), max(h, h2))
                            if e not in used_edges:
                                has_option = True
                                break
                        if not has_option:
                            fc_valid = False
                            break
                
                if fc_valid and backtrack(pos + 1):
                    return True
                
                mapping[tree_v] = -1
                used_hosts.remove(h)
            
            return False
        
        for _ in range(30):
            mapping = [-1] * k
            used_hosts = set()
            if backtrack(0):
                return mapping
        return None

    def try_pack(attempt_seed):
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        # Vary the packing order slightly for diversity
        pack_order = order[:]
        if attempt_seed % 3 == 0:
            # Occasionally try a different order: smallest first
            pack_order = list(reversed(order))
        elif attempt_seed % 3 == 1:
            # Random shuffle
            local_rng.shuffle(pack_order)
        
        for t_idx in pack_order:
            if time.time() > deadline:
                break
            
            mapping = find_embedding(t_idx, used_edges, local_rng)
            if mapping is None:
                # Retry with fresh state
                found = False
                for _ in range(10):
                    mapping = find_embedding(t_idx, used_edges, local_rng)
                    if mapping is not None:
                        found = True
                        break
                if not found:
                    continue
            mappings[t_idx] = mapping
            for u, v in trees[t_idx]:
                hu = mapping[u]
                hv = mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
        
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += tree_info[i]['num_edges']
        
        return score, mappings

    best_score = 0
    best_mappings = [None] * 20
    
    # Allocate time: first 60% for many quick attempts, last 40% for focused attempts
    total_time = 145
    quick_phase_end = start_time + total_time * 0.6
    
    attempt = 0
    while time.time() < deadline:
        # Use different strategies in different phases
        if time.time() < quick_phase_end:
            # Quick attempts with more diversity
            score, mappings = try_pack(attempt)
        else:
            # Focused attempts: reuse best mappings as starting point
            score, mappings = try_pack(attempt + 1000)
        
        if score > best_score:
            best_score = score
            best_mappings = mappings[:]
        attempt += 1
        if best_score == 190:
            break
        
        # Safety: if we're getting close to deadline, stop early
        if time.time() > deadline - 5:
            break

    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            # Ensure all values are Python ints
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    
    # Final validation and cleanup
    used_edges_check = set()
    valid = True
    for i in range(20):
        if result[i] is None:
            continue
        mapping = result[i]
        k = i + 1
        if len(mapping) != k:
            valid = False
            break
        if len(set(mapping)) != k:
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges_check:
                valid = False
                break
            used_edges_check.add(e)
        if not valid:
            break
    
    if not valid:
        # Fallback: return empty result (all None) to avoid invalid output
        return [None] * 20
    
    return result

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    trees = instance_trees(20, 611)
    used_edges = set()
    total_edges = 0
    valid = True
    for i in range(20):
        if result[i] is None:
            continue
        mapping = result[i]
        k = i + 1
        if len(mapping) != k:
            print(f"Tree {i}: wrong length {len(mapping)} vs {k}")
            valid = False
            break
        if len(set(mapping)) != k:
            print(f"Tree {i}: duplicate host vertices")
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                print(f"Tree {i}: invalid vertex {v}")
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges:
                print(f"Tree {i}: edge conflict {e}")
                valid = False
                break
            used_edges.add(e)
            total_edges += 1
        if not valid:
            break
    
    print(f"Valid: {valid}")
    print(f"Total edges: {total_edges}")
    if valid:
        print(f"Score: {total_edges}")
    else:
        print("INVALID RESULT")
