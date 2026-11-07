"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
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
    deadline = start_time + 115.0
    
    trees = instance_trees(20, 611)
    num_trees = 20
    N = 20
    
    # Pre-compute tree adjacency
    tree_adjs = []
    for i in range(num_trees):
        k = i + 1
        if k <= 2:
            tree_adjs.append([] if k == 1 else [[1], [0]])
            continue
        adj = [[] for _ in range(k)]
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adjs.append(adj)
    
    # Order trees by decreasing size
    order = list(range(num_trees - 1, -1, -1))
    
    def embed_tree_csp(tree_idx, used_edges, rng, time_limit):
        """
        Embeds tree tree_idx into K20 avoiding used_edges.
        Uses CSP backtracking with MRV and forward checking.
        Returns mapping list or None.
        """
        k = tree_idx + 1
        if k == 1:
            for v in range(N):
                return [v]
            return None
        
        if k == 2:
            for u in range(N):
                for v in range(u + 1, N):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        return [u, v]
            return None
        
        edges = trees[tree_idx]
        adj = tree_adjs[tree_idx]
        
        # Select root: vertex with max degree
        degrees = [len(adj[i]) for i in range(k)]
        root = max(range(k), key=lambda i: degrees[i])
        
        # State: assigned dict, used_hosts set
        # We use a stack-based DFS to avoid recursion limits if needed, but recursion is fine for k<=20
        # However, for speed, we will use a simple recursive function with a global time check
        
        best_mapping = None
        
        def backtrack(assigned, used_hosts):
            nonlocal best_mapping
            if time.time() > time_limit:
                return False
            
            if len(assigned) == k:
                # Verify all edges
                for u, v in edges:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return False
                best_mapping = list(assigned.values())
                return True
            
            # MRV: Select unassigned vertex with smallest domain
            unassigned = [v for v in range(k) if v not in assigned]
            best_v = -1
            min_domain_size = N + 1
            best_domain = []
            
            for v in unassigned:
                valid_hosts = []
                for h in range(N):
                    if h in used_hosts:
                        continue
                    # Check against assigned neighbors
                    ok = True
                    for u in adj[v]:
                        if u in assigned:
                            hu = assigned[u]
                            e_key = (min(h, hu), max(h, hu))
                            if e_key in used_edges:
                                ok = False
                                break
                    if ok:
                        valid_hosts.append(h)
                
                if len(valid_hosts) < min_domain_size:
                    min_domain_size = len(valid_hosts)
                    best_v = v
                    best_domain = valid_hosts
                    if min_domain_size == 0:
                        break
            
            if best_v == -1 or min_domain_size == 0:
                return False
            
            # Shuffle domain to introduce randomness/restarts
            rng.shuffle(best_domain)
            
            for h in best_domain:
                assigned[best_v] = h
                used_hosts.add(h)
                
                if backtrack(assigned, used_hosts):
                    return True
                
                del assigned[best_v]
                used_hosts.remove(h)
            
            return False
        
        # Try a few random restarts for this specific tree embedding
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            assigned = {root: rng.randint(0, N-1)}
            # Actually, let's just let MRV pick the first one? 
            # Or fix root to a random vertex?
            # Let's fix root to a random valid vertex to diversify
            root_hosts = [h for h in range(N) if h not in used_hosts] # used_hosts is empty initially
            if not root_hosts:
                continue
            root_h = rng.choice(root_hosts)
            assigned = {root: root_h}
            used_hosts = {root_h}
            
            if backtrack(assigned, used_hosts):
                return best_mapping
        
        return None

    # Main loop: try to pack all trees
    best_mappings = [None] * num_trees
    best_score = 0
    
    # We will try multiple full restarts of the entire packing process
    # Since the trees are fixed, we just change the order of trial for embeddings or the random seed
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        used_edges = set()
        mappings = [None] * num_trees
        current_score = 0
        success = True
        rng = Random(restart * 12345 + 67890)
        
        # Embed trees in descending order
        for tree_idx in order:
            if time.time() > deadline:
                success = False
                break
            
            # Allocate time for this embedding
            # Simple heuristic: give each tree a fixed slice of remaining time
            # Or just let it run until it finds a solution or times out
            # Since we have 50 restarts, let's limit each embedding attempt to ~2s
            # But the first few (largest) might need more.
            # Let's just use the global deadline.
            
            mapping = embed_tree_csp(tree_idx, used_edges, rng, deadline)
            
            if mapping is None:
                success = False
                break
            
            mappings[tree_idx] = mapping
            current_score += len(trees[tree_idx])
            
            # Update used_edges
            for u, v in trees[tree_idx]:
                hu = mapping[u]
                hv = mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                used_edges.add(e_key)
        
        if success:
            # We packed all trees!
            if current_score > best_score:
                best_score = current_score
                best_mappings = mappings[:]
            break # Found a perfect packing
        
        # If we didn't pack all, we might still have a partial solution
        # But we only count valid packed trees.
        # The current code breaks on first failure, so we only have a prefix of the order.
        # This is suboptimal. We should try to pack as many as possible, even if one fails.
        # Let's modify the loop to continue even if one fails?
        # No, the contract says "Embedded trees must be edge-disjoint". It doesn't say we must pack them in a specific order.
        # But if we fail to pack T20, we can still try to pack T19, etc.
        # However, the current `embed_tree_csp` modifies `used_edges` only if it returns a mapping.
        # If it returns None, `used_edges` is unchanged.
        # So we can just continue to the next tree in the order.
        
        # Let's re-implement the main loop to be more robust:
        # Iterate through trees in a specific order. For each, try to embed.
        # If successful, commit. If not, skip (mark as None).
        # This allows packing a subset.
        
        # Actually, the previous block already did `if mapping is None: success = False; break`.
        # Let's change it to just skip and continue.
        
        # Wait, the code above is inside the `for restart` loop.
        # Let's refine the logic inside the restart loop.
        
        # --- Revised Inner Loop Logic ---
        # (The code above is already written, but I need to make sure it doesn't break on failure)
        # I will rewrite the inner loop to be more robust.
        
        pass # Placeholder, will rewrite below

    # --- Robust Packing Loop ---
    best_mappings = [None] * num_trees
    best_score = 0
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        used_edges = set()
        mappings = [None] * num_trees
        current_score = 0
        rng = Random(restart * 12345 + 67890)
        
        # Try to embed trees in descending order
        for tree_idx in order:
            if time.time() > deadline:
                break
            
            # Limit time per embedding to ensure we don't get stuck on one tree
            # Give each tree ~1 second? 20 trees * 1s = 20s per restart.
            # 100 restarts * 20s = 2000s. Too long.
            # We need to be faster.
            # The CSP solver is fast for small trees.
            # Let's just rely on the global deadline.
            
            mapping = embed_tree_csp(tree_idx, used_edges, rng, deadline)
            
            if mapping is not None:
                mappings[tree_idx] = mapping
                current_score += len(trees[tree_idx])
                
                # Update used_edges
                for u, v in trees[tree_idx]:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    used_edges.add(e_key)
        
        if current_score > best_score:
            best_score = current_score
            best_mappings = mappings[:]
            
            if best_score == 190:
                break
    
    # Final result construction
    final_result = []
    for i in range(num_trees):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
    
    return final_result

# EVOLVE-BLOCK-END
