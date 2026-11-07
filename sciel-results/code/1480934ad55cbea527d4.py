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
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    num_trees = 20
    
    # Pre-compute tree adjacency and edge lists
    tree_data = []
    for i in range(num_trees):
        k = i + 1
        if k == 1:
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            tree_data.append({'edges': edges, 'adj': adj, 'size': k})
            
    # Strategy: Deterministic Greedy Edge-Embedding
    # 1. Order trees by "difficulty": higher total degree or larger size first.
    #    Actually, larger trees are harder to fit if the graph is fragmented.
    #    Let's try embedding largest trees first (20, 19, ...) as they need the most contiguous resources.
    #    However, small trees are more flexible. 
    #    A common heuristic for packing is to place the most constrained items first.
    #    In K20, all vertices are symmetric initially. 
    #    Let's order by size descending: T20, T19, ..., T1.
    
    order = list(range(19, -1, -1))
    
    # State:
    # used_edges: set of tuples (u, v) with u < v, representing edges already used in K20.
    # mappings: list of 20 entries. Entry i is None or list of 20 ints (mapping for T_{i+1}).
    
    best_score = 0
    best_mappings = [None] * 20
    used_edges = set()
    
    # To ensure we get a valid result even if greedy fails, we keep track of the best partial solution.
    # But the task requires returning 20 entries. If a tree can't be embedded, it's None.
    # The score is the total number of embedded tree edges.
    
    # We will try multiple random restarts with different random tie-breakings to escape local optima of the greedy choice.
    # Since the graph is K20, the "greedy" choice of which specific host vertices to use matters.
    
    def try_greedy_embedding(rng):
        """
        Attempts to embed all trees in 'order' using a greedy strategy with random tie-breaking.
        Returns (score, mappings)
        """
        local_used_edges = set()
        local_mappings = [None] * num_trees
        current_score = 0
        
        # For each tree in order
        for tree_idx in order:
            k = tree_data[tree_idx]['size']
            if k == 1:
                # T1 has 0 edges. Map vertex 0 to 0.
                local_mappings[tree_idx] = [0]
                continue
                
            if k == 2:
                # T2 has 1 edge. Find any available edge in K20.
                found = False
                for u in range(20):
                    for v in range(u + 1, 20):
                        e_key = (u, v)
                        if e_key not in local_used_edges:
                            local_mappings[tree_idx] = [u, v]
                            local_used_edges.add(e_key)
                            current_score += 1
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # No edge available. This shouldn't happen if we embed large trees first and there's plenty of space.
                    # But if it does, we stop.
                    return current_score, local_mappings
                continue
            
            # For k > 2:
            # We need to map k vertices of the tree to k distinct host vertices in 0..19.
            # The edges of the tree must map to distinct edges in K20 not in local_used_edges.
            
            # Greedy vertex assignment:
            # 1. Identify the "center" or high-degree vertices of the tree.
            # 2. Assign them to host vertices with the most remaining capacity (or just randomly from available).
            # 3. Assign leaves to available neighbors.
            
            # A simpler greedy approach: Randomly select a subset of k host vertices from 0..19.
            # Check if the tree can be embedded into this induced subgraph K_k such that edges are disjoint.
            # This is equivalent to checking if the tree is a subgraph of the available edges within that set.
            # Since K20 is dense, most random subsets of size k will work if we haven't used too many edges.
            
            # Better: Construct the embedding incrementally.
            # 1. Pick a root vertex r in the tree.
            # 2. Assign r to a random host vertex h_r.
            # 3. For each neighbor of r, assign to a random available host vertex connected to h_r (which is all others in K20, minus used).
            # 4. Continue BFS.
            
            # Let's use a BFS-based greedy embedding with backtracking limited to depth 2 or 3 if stuck?
            # No, let's stick to a simpler heuristic:
            # Try to find a random mapping of the k tree vertices to k host vertices.
            # Verify edge disjointness.
            # If it works, accept.
            # If not, try again (up to a limit).
            
            found_embedding = False
            max_attempts = 500  # Limit attempts per tree to keep speed high
            
            for _ in range(max_attempts):
                # Select k distinct host vertices
                host_candidates = list(range(20))
                rng.shuffle(host_candidates)
                selected_hosts = host_candidates[:k]
                
                # Try to map tree vertices to these selected hosts
                # We need a bijection from tree vertices (0..k-1) to selected_hosts.
                # Let's try a random permutation of selected_hosts.
                mapping = list(selected_hosts)
                rng.shuffle(mapping)
                
                # Check if this mapping is valid (edge disjointness)
                valid = True
                new_edges = []
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in local_used_edges:
                        valid = False
                        break
                    new_edges.append(e_key)
                
                if valid:
                    # Also check that the mapping is injective (guaranteed by distinct selected_hosts)
                    local_mappings[tree_idx] = mapping
                    local_used_edges.update(new_edges)
                    current_score += len(tree_data[tree_idx]['edges'])
                    found_embedding = True
                    break
            
            if not found_embedding:
                # Failed to embed this tree.
                # We could try a more sophisticated search here, but for now, we stop.
                # The score is current_score.
                return current_score, local_mappings
                
        return current_score, local_mappings

    # Run multiple restarts with different seeds
    best_score = 0
    best_mappings = [None] * 20
    
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        rng = Random(restart * 12345 + 6789)
        score, mappings = try_greedy_embedding(rng)
        
        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in mappings]
            
        if best_score == 190:
            break
            
    # Final check and conversion
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            # Ensure integers are Python int
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
