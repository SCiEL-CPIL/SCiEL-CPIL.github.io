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
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
            
    # Total edges in K20 is 190.
    # Sum of edges in trees is 190.
    # We need a perfect decomposition.
    
    # Strategy: 
    # 1. Order trees by difficulty (largest first, or those with high max degree first).
    #    T20 (spanning tree) is most constrained in terms of size but flexible in shape.
    #    Actually, embedding large trees first is generally better because they consume 
    #    more "resources" (edges) and leave less flexibility for smaller ones if we 
    #    fail. However, small trees are easier to place anywhere.
    #    Let's try embedding in order of decreasing size: 20, 19, ..., 1.
    
    order = list(range(19, -1, -1))
    
    best_score = 0
    best_mappings = [None] * 20
    
    # We will use a randomized restart strategy with a sophisticated backtracker.
    # The backtracker assigns host vertices to tree vertices one by one.
    # To make it fast, we use:
    # 1. MRV (Minimum Remaining Values): Pick the tree vertex with the fewest available host vertices.
    # 2. Forward Checking: After assigning a vertex, update the domains of its neighbors.
    # 3. Conflict-Directed Backjumping (simplified): If a conflict occurs, backtrack to the last 
    #    assignment that could have caused it. (Simplified to standard backtracking for now, 
    #    but with strong pruning).
    
    # Given the small size (20 vertices), a well-ordered backtracking search might solve it quickly.
    # But 20! is huge. We need strong heuristics.
    
    # Alternative: Since the graph is K20, the only constraint is edge-disjointness.
    # This is equivalent to finding a set of edge-disjoint trees.
    # We can model this as a matching problem in a hypergraph? No, too complex.
    
    # Let's stick to the CSP approach but optimize it.
    # Key insight: The trees are random. K20 is highly symmetric.
    # We can fix the first vertex of the largest tree to 0 to break symmetry?
    # No, the host graph is fixed, but we can permute host labels.
    # We can fix the mapping of the first few vertices of the largest tree to specific host vertices
    # to reduce the search space by a factor of 20*19*...
    
    # Let's try a different structural approach:
    # **Simulated Annealing / Local Search** on the set of mappings.
    # Start with a random valid packing (or partial packing).
    # Define energy as the number of conflicts (overlapping edges).
    # Move: Swap two host vertices in one of the tree mappings, or re-map a single vertex.
    # If energy decreases, accept. If energy increases, accept with probability exp(-dE/T).
    # This might find a perfect packing (energy 0) quickly.
    
    # However, ensuring "distinct int host vertices" within a tree is a hard constraint.
    # We can enforce this in the moves: only swap if the new vertex is not already in the tree.
    
    # Let's implement a Local Search approach.
    
    def get_initial_mappings():
        """Generate random initial mappings for all trees."""
        mappings = [None] * num_trees
        used_edges = set()
        
        # Embed trees in random order? Or size order?
        # Let's embed in size order, picking random available edges/vertices.
        for tree_idx in order:
            k = tree_data[tree_idx]['size']
            if k == 1:
                mappings[tree_idx] = [0]
                continue
            if k == 2:
                # Find any available edge
                available = [(u, v) for u in range(20) for v in range(u+1, 20) if (u,v) not in used_edges]
                if not available:
                    return None, 0, used_edges
                u, v = available[0]
                mappings[tree_idx] = [u, v]
                used_edges.add((u, v))
                continue
            
            # For k > 2, try to find a random embedding
            found = False
            for _ in range(100):
                host_verts = list(range(20))
                Random().shuffle(host_verts)
                mapping = host_verts[:k]
                valid = True
                edges_to_add = []
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    edges_to_add.append(e_key)
                
                if valid:
                    mappings[tree_idx] = mapping
                    used_edges.update(edges_to_add)
                    found = True
                    break
            
            if not found:
                # If we can't find one quickly, leave as None (partial solution)
                # Or fail? For local search, we need a complete assignment to start.
                # If we can't embed a tree, the initial state is "invalid" with high energy.
                # Let's just return None if we can't get a complete initial packing.
                return None, 0, used_edges
                
        return mappings, len(used_edges), used_edges

    # Local Search Parameters
    max_iterations = 50000
    initial_temp = 1.0
    cooling_factor = 0.999
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        mappings, score, used_edges = get_initial_mappings()
        if mappings is None:
            continue
            
        if score == 190:
            best_score = 190
            best_mappings = [m[:] if m else None for m in mappings]
            break
            
        # Calculate initial energy (number of conflicts)
        # Since we built it greedily, there are no conflicts by construction.
        # So energy = 0? No, we might have failed to embed some trees (None).
        # Let's define energy as:
        # 1. Number of None mappings * 100 (penalty for missing trees)
        # 2. Number of overlapping edges (should be 0 if constructed carefully)
        
        # Wait, the greedy construction above ensures no overlaps for the trees it embeds.
        # But it might leave some trees as None.
        # We need to handle None mappings in local search.
        # If a tree is None, we need to assign it a mapping.
        
        # Let's refine the initial state generation to ensure ALL trees are mapped,
        # even if it creates conflicts. Then local search removes conflicts.
        
        def get_conflicting_initial_mappings():
            mappings = [None] * num_trees
            used_edges = {} # edge -> count
            for tree_idx in range(num_trees):
                k = tree_data[tree_idx]['size']
                if k == 1:
                    mappings[tree_idx] = [0]
                    continue
                host_verts = list(range(20))
                Random().shuffle(host_verts)
                mapping = host_verts[:k]
                mappings[tree_idx] = mapping
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    used_edges[e_key] = used_edges.get(e_key, 0) + 1
            return mappings, used_edges

        mappings, used_edges = get_conflicting_initial_mappings()
        
        def calculate_energy():
            energy = 0
            # Count conflicts
            for e_key, count in used_edges.items():
                if count > 1:
                    energy += count - 1
            # Penalty for None mappings (shouldn't happen here)
            for m in mappings:
                if m is None:
                    energy += 1000
            return energy

        current_energy = calculate_energy()
        if current_energy == 0:
            best_score = 190
            best_mappings = [m[:] if m else None for m in mappings]
            break
            
        temp = initial_temp
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Pick a random tree to modify
            tree_idx = Random().randint(0, num_trees - 1)
            k = tree_data[tree_idx]['size']
            if k == 1:
                continue
                
            old_mapping = mappings[tree_idx]
            
            # Generate a new mapping for this tree
            # Strategy: Perturb the existing mapping or generate a new random one.
            # Let's generate a new random one, biased towards keeping some vertices?
            # Or just random.
            
            new_mapping = list(range(20))
            Random().shuffle(new_mapping)
            new_mapping = new_mapping[:k]
            
            # Calculate energy change
            # Remove old edges from used_edges
            old_edges = []
            for u, v in tree_data[tree_idx]['edges']:
                hu = old_mapping[u]
                hv = old_mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                old_edges.append(e_key)
                used_edges[e_key] -= 1
                if used_edges[e_key] == 0:
                    del used_edges[e_key]
                    
            # Add new edges
            new_edges = []
            for u, v in tree_data[tree_idx]['edges']:
                hu = new_mapping[u]
                hv = new_mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                new_edges.append(e_key)
                used_edges[e_key] = used_edges.get(e_key, 0) + 1
                
            new_energy = calculate_energy()
            
            # Accept or reject
            delta = new_energy - current_energy
            if delta <= 0 or Random() < pow(2.71828, -delta / temp):
                # Accept
                mappings[tree_idx] = new_mapping
                current_energy = new_energy
                if current_energy == 0:
                    best_score = 190
                    best_mappings = [m[:] if m else None for m in mappings]
                    break
            else:
                # Reject: Undo changes
                for e_key in new_edges:
                    used_edges[e_key] -= 1
                    if used_edges[e_key] == 0:
                        del used_edges[e_key]
                for e_key in old_edges:
                    used_edges[e_key] = used_edges.get(e_key, 0) + 1
                    
            temp *= cooling_factor
            
        if current_energy == 0:
            break

    # If local search didn't find a perfect packing, try the backtracking approach as a fallback
    # or just return the best found.
    # Actually, the local search might get stuck. Let's add a simple backtracking for the last few trees if stuck.
    
    # For now, return the best mapping found.
    # If best_score is 0, we need a fallback.
    if best_score == 0:
        # Fallback: Greedy embedding
        mappings = [None] * num_trees
        used_edges = set()
        for tree_idx in order:
            k = tree_data[tree_idx]['size']
            if k == 1:
                mappings[tree_idx] = [0]
                continue
            if k == 2:
                available = [(u, v) for u in range(20) for v in range(u+1, 20) if (u,v) not in used_edges]
                if available:
                    u, v = available[0]
                    mappings[tree_idx] = [u, v]
                    used_edges.add((u, v))
                continue
            found = False
            for _ in range(1000):
                host_verts = list(range(20))
                Random().shuffle(host_verts)
                mapping = host_verts[:k]
                valid = True
                edges_to_add = []
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    edges_to_add.append(e_key)
                if valid:
                    mappings[tree_idx] = mapping
                    used_edges.update(edges_to_add)
                    found = True
                    break
            if found:
                best_score += len(tree_data[tree_idx]['edges'])
                best_mappings = [m[:] if m else None for m in mappings]

    # Final check and conversion
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
