"""Improved K2,3-free graph construction for n=20 vertices."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    def is_valid_addition(adj, common, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # Check 1: u and v must not have 3+ common neighbors
        if common[u][v] >= 3:
            return False
        # Check 2: For each neighbor w of v, common[u][w] must be < 2
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w] and common[u][w] >= 2:
                return False
        # Check 3: For each neighbor w of u, common[v][w] must be < 2
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w] and common[v][w] >= 2:
                return False
        return True

    def add_edge(adj, common, u, v):
        """Add edge (u,v) and update common neighbor counts."""
        adj[u][v] = True
        adj[v][u] = True
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1

    def remove_edge(adj, common, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        adj[u][v] = False
        adj[v][u] = False
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
            if adj[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1

    def get_edge_list(adj):
        """Extract edge list from adjacency matrix."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def greedy_build(seed, order_strategy='random'):
        """Build graph using greedy edge addition with specified order strategy."""
        rng = random.Random(seed)
        adj = [[False]*n for _ in range(n)]
        common = [[0]*n for _ in range(n)]
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        if order_strategy == 'random':
            rng.shuffle(all_pairs)
        elif order_strategy == 'degree':
            # Prioritize pairs where both vertices currently have low degree
            # This is a heuristic - we'll use a static random order but could refine
            rng.shuffle(all_pairs)
        else:
            rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, common, u, v):
                add_edge(adj, common, u, v)
        
        return adj, common

    def local_search(adj, common, max_iterations=500):
        """
        Swap-based local search: try removing one edge and adding up to 2 others.
        Returns improved (adj, common) if found.
        """
        if time.time() > deadline:
            return adj, common
        
        current_edges = get_edge_list(adj)
        current_count = len(current_edges)
        
        # Build set for quick lookup
        edge_set = set()
        for e in current_edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        best_adj = [row[:] for row in adj]
        best_common = [row[:] for row in common]
        best_count = current_count
        
        rng = random.Random(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random existing edge to potentially remove
            if not current_edges:
                break
            rem_u, rem_v = rng.choice(current_edges)
            rem_u, rem_v = min(rem_u, rem_v), max(rem_u, rem_v)
            
            # Try removing this edge
            remove_edge(adj, common, rem_u, rem_v)
            
            # Now try to add up to 2 new edges
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(all_pairs)
            
            added_count = 0
            added_edges = []
            
            for u, v in all_pairs:
                if (u, v) == (rem_u, rem_v):
                    continue
                if (v, u) == (rem_u, rem_v):
                    continue
                if (u, v) in edge_set:
                    continue
                if (v, u) in edge_set:
                    continue
                
                if is_valid_addition(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    added_count += 1
                    added_edges.append((u, v))
                    if added_count >= 2:
                        break
            
            # Evaluate: if we added more edges than we removed (1), we improved
            new_count = current_count - 1 + added_count
            
            if new_count > best_count:
                best_count = new_count
                best_adj = [row[:] for row in adj]
                best_common = [row[:] for row in common]
                # Update current state to the improved one
                current_count = new_count
                current_edges = get_edge_list(adj)
                edge_set = set()
                for e in current_edges:
                    edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
                
                # Reset to continue from this better state
                adj = [row[:] for row in best_adj]
                common = [row[:] for row in best_common]
            else:
                # Revert: remove the added edges and re-add the removed edge
                for u, v in added_edges:
                    remove_edge(adj, common, u, v)
                add_edge(adj, common, rem_u, rem_v)
        
        return best_adj, best_common

    def multi_start_local_search(adj, common, num_rounds=3):
        """Run multiple rounds of local search with perturbations."""
        best_adj = [row[:] for row in adj]
        best_common = [row[:] for row in common]
        best_count = len(get_edge_list(adj))
        
        for rnd in range(num_rounds):
            if time.time() > deadline:
                break
            
            # Perturb: remove a few random edges
            adj_pert = [row[:] for row in adj]
            common_pert = [row[:] for row in common]
            edges = get_edge_list(adj_pert)
            
            rng = random.Random(100 + rnd)
            num_remove = rng.randint(1, 3)
            for _ in range(min(num_remove, len(edges))):
                if not edges:
                    break
                u, v = rng.choice(edges)
                u, v = min(u, v), max(u, v)
                if adj_pert[u][v]:
                    remove_edge(adj_pert, common_pert, u, v)
                    edges = [e for e in edges if not (min(e[0],e[1])==u and max(e[0],e[1])==v)]
            
            # Run local search on perturbed graph
            adj_ls, common_ls = local_search(adj_pert, common_pert, max_iterations=200)
            
            count_ls = len(get_edge_list(adj_ls))
            if count_ls > best_count:
                best_count = count_ls
                best_adj = [row[:] for row in adj_ls]
                best_common = [row[:] for row in common_ls]
                adj = best_adj
                common = best_common
        
        return best_adj, best_common

    # Main search loop
    best_adj = None
    best_common = None
    best_count = 0
    
    # Strategy 1: Multiple greedy restarts with local search
    num_restarts = 30
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj, common = greedy_build(seed, order_strategy='random')
        count = len(get_edge_list(adj))
        
        # Apply local search
        adj, common = local_search(adj, common, max_iterations=300)
        count = len(get_edge_list(adj))
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_common = [row[:] for row in common]
    
    # Strategy 2: Multi-start with perturbations on the best found
    if best_adj is not None and time.time() < deadline:
        best_adj, best_common = multi_start_local_search(best_adj, best_common, num_rounds=5)
        best_count = len(get_edge_list(best_adj))
    
    # Strategy 3: More greedy restarts if time permits
    if time.time() < deadline:
        for seed in range(num_restarts, num_restarts + 30):
            if time.time() > deadline:
                break
            adj, common = greedy_build(seed, order_strategy='random')
            count = len(get_edge_list(adj))
            
            if count + 5 > best_count:  # Only do local search if promising
                adj, common = local_search(adj, common, max_iterations=200)
                count = len(get_edge_list(adj))
            
            if count > best_count:
                best_count = count
                best_adj = [row[:] for row in adj]
                best_common = [row[:] for row in common]
    
    if best_adj is not None:
        return get_edge_list(best_adj)
    
    # Fallback: disjoint K4s
    edges = []
    for k in range(5):
        for a in range(4):
            for b in range(a+1, 4):
                edges.append([4*k+a, 4*k+b])
    return edges
