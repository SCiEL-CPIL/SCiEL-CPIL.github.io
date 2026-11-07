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
    deadline = start_time + 145  # 5s margin

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
    
    # Sort by size descending
    order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    def find_embedding(t_idx, used_edges, rng, time_limit):
        """
        Find an embedding of tree t_idx into K20 avoiding used_edges.
        Returns a list of host vertices for tree vertices 0..k-1, or None.
        """
        info = tree_info[t_idx]
        k = info['k']
        adj_t = info['adj']
        deg_t = info['deg']
        
        if k == 1:
            # Find any unused host? No, T1 has no edges, so any host works.
            # But we must return a valid vertex.
            return [0] # Symmetry allows us to fix this? No, T1 is isolated.
                       # Wait, T1 is a single vertex. It consumes no edges.
                       # It can be mapped to ANY vertex 0..19.
                       # To be safe and distinct from others? No, vertices can be shared
                       # across different trees? 
                       # "Entry k-1 is None or a list of k distinct int host vertices".
                       # "Embedded trees must be edge-disjoint".
                       # Vertices CAN be shared. Only edges must be disjoint.
                       # So T1 can be [0].
        
        # Order vertices by degree descending for backtracking
        # This helps fail fast if a high-degree node has no options
        verts = list(range(k))
        verts.sort(key=lambda v: -deg_t[v])
        
        # We need to map these vertices to hosts 0..19
        # mapping[tree_v] = host_v
        mapping = [-1] * k
        used_hosts = set() # For this tree, hosts must be distinct?
        # "list of k distinct int host vertices" -> Yes, within a tree, hosts must be distinct.
        
        # Backtracking
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            
            if idx == k:
                # All mapped. Check all edges.
                # We checked edges during assignment, so this should be valid.
                return True
            
            tv = verts[idx]
            
            # Candidates: hosts 0..19 not in used_hosts
            # Additional constraint: if tv has neighbors already mapped, 
            # the host must be connected to their hosts via unused edges.
            
            # Find mapped neighbors
            mapped_neighbors = []
            for nb in adj_t[tv]:
                if mapping[nb] != -1:
                    mapped_neighbors.append(mapping[nb])
            
            candidates = []
            for h in range(20):
                if h in used_hosts:
                    continue
                ok = True
                for nh in mapped_neighbors:
                    e_key = min(h, nh) * 20 + max(h, nh)
                    if e_key in used_edges:
                        ok = False
                        break
                if ok:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Shuffle candidates for randomness
            rng.shuffle(candidates)
            
            for h in candidates:
                # Forward checking:
                # For each unmapped neighbor of tv, is there at least one available host?
                # This is expensive. Maybe skip for now, rely on backtracking.
                
                mapping[tv] = h
                used_hosts.add(h)
                
                if backtrack(idx + 1):
                    return True
                
                mapping[tv] = -1
                used_hosts.remove(h)
            
            return False
        
        # Try a few times with different shuffles if it fails?
        # The shuffle is inside the recursion, so it varies per node.
        # We can just call it once, but if it fails, it might be due to bad initial choice.
        # Let's try multiple times with different random seeds for the shuffle logic?
        # No, the shuffle uses the passed rng.
        
        # To improve success rate, we can try different root mappings?
        # The root is the first vertex in `verts` (highest degree).
        # We can try all 20 hosts for the root?
        # That multiplies the search space by 20.
        
        # Let's just run the backtracking. If it fails, return None.
        # The caller can retry with a different global state or seed.
        
        if backtrack(0):
            return mapping[:]
        
        return None

    def try_pack(seed):
        rng = random.Random(seed)
        used_edges = set()
        mappings = [None] * 20
        
        for t_idx in order:
            if time.time() > deadline:
                break
            
            mapping = find_embedding(t_idx, used_edges, rng, deadline)
            if mapping is None:
                # Try a few more times with fresh rng state?
                # Or just fail this attempt.
                return None, mappings
            
            mappings[t_idx] = mapping
            
            for u, v in trees[t_idx]:
                hu = mapping[u]
                hv = mapping[v]
                e_key = min(hu, hv) * 20 + max(hu, hv)
                used_edges.add(e_key)
        
        # Calculate score
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += tree_info[i]['num_edges']
        
        return score, mappings

    best_score = 0
    best_mappings = [None] * 20
    
    # Try multiple seeds
    # If we can pack all, score is 190.
    # We should try to find a perfect packing.
    
    seed = 0
    while time.time() < deadline:
        score, mappings = try_pack(seed)
        if score > best_score:
            best_score = score
            best_mappings = mappings[:]
            if best_score == 190:
                break
        seed += 1
        
        # If we are stuck at a low score, maybe we need a different strategy?
        # For now, just restart with different randomization.
        
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
            
    return result

# Test
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
            print(f"Tree {i}: wrong length")
            valid = False
            break
        if len(set(mapping)) != k:
            print(f"Tree {i}: duplicate hosts")
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                print(f"Tree {i}: invalid vertex")
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e_key = min(hu, hv) * 20 + max(hu, hv)
            if e_key in used_edges:
                print(f"Tree {i}: edge conflict")
                valid = False
                break
            used_edges.add(e_key)
            total_edges += 1
        if not valid:
            break
    
    print(f"Valid: {valid}")
    print(f"Total edges: {total_edges}")
