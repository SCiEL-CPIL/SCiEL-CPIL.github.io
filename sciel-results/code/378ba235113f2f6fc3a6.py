import time
import random
from collections import deque

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
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        deg = [len(adj[v]) for v in range(k)]
        tree_info.append({'k': k, 'edges': edges, 'adj': adj, 'num_edges': len(edges), 'deg': deg})
    
    def try_pack(attempt_seed, order):
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        for t_idx in order:
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            
            if k == 1:
                mappings[t_idx] = [0]
                continue
            
            found = False
            for _ in range(100):
                mapping = [-1] * k
                used_hosts = set()
                
                root_t = local_rng.randint(0, k - 1)
                mapping[root_t] = local_rng.randint(0, 19)
                used_hosts.add(mapping[root_t])
                
                q = deque([root_t])
                visited = {root_t}
                success = True
                
                while q and success:
                    u = q.popleft()
                    hu = mapping[u]
                    nbrs = [v for v in adj_t[u] if v not in visited]
                    nbrs.sort(key=lambda x: -info['deg'][x])
                    
                    for v in nbrs:
                        visited.add(v)
                        candidates = [hv for hv in range(20) if hv not in used_hosts and (min(hu,hv),max(hu,hv)) not in used_edges]
                        if not candidates:
                            success = False
                            break
                        candidates.sort(key=lambda hv: sum(1 for o in range(20) if o!=hv and (min(hv,o),max(hv,o)) not in used_edges))
                        hv = local_rng.choice(candidates[:min(3,len(candidates))])
                        mapping[v] = hv
                        used_hosts.add(hv)
                        used_edges.add((min(hu,hv),max(hu,hv)))
                        q.append(v)
                
                if success:
                    valid = all((min(mapping[u2],mapping[v2]),max(mapping[u2],mapping[v2])) in used_edges for u2,v2 in info['edges'])
                    if valid:
                        mappings[t_idx] = mapping
                        found = True
                        break
                else:
                    for u2,v2 in info['edges']:
                        if mapping[u2]!=-1 and mapping[v2]!=-1:
                            used_edges.discard((min(mapping[u2],mapping[v2]),max(mapping[u2],mapping[v2])))
            
            # If not found, skip this tree and continue with others
        
        score = sum(tree_info[i]['num_edges'] for i in range(20) if mappings[i] is not None)
        return mappings, score

    best_mappings = [None] * 20
    best_score = 0
    
    base_order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    orders = [base_order]
    for _ in range(5):
        s = base_order[:]
        random.Random().shuffle(s)
        orders.append(s)
    
    attempt = 0
    while time.time() < deadline:
        order = orders[attempt % len(orders)]
        mappings, score = try_pack(attempt, order)
        if mappings is not None and score > best_score:
            best_score = score
            best_mappings = mappings[:]
            if best_score == 190:
                break
        attempt += 1
    
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    return result

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    # Verify
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
