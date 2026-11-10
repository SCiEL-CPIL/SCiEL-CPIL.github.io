import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Combines:
    - Bitmask-based fast C6 detection (can_add)
    - Priority ranking for edge addition order
    - Targeted kicks: remove edges from the blocking C6
    - Simulated annealing to escape local optima
    - Final polish pass to squeeze out extra edges
    - Multi-restart with different seeds
    """
    N = 40
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Bitmask adjacency: adj_bits[u] is a bitmask of neighbors of u
        adj_bits = [0] * N
        edges_set = set()  # Set for O(1) removal
        edges_list = []  # list of (u, v) with u < v
        
        def add_edge(u, v):
            if u > v:
                u, v = v, u
            if adj_bits[u] & (1 << v):
                return False
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges_set.add((u, v))
            edges_list.append((u, v))
            return True
        
        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            if not (adj_bits[u] & (1 << v)):
                return False
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            edges_set.discard((u, v))
            if (u, v) in edges_set:
                pass
            # Remove from list - we'll rebuild if needed
            edges_list.remove((u, v))
            return True
        
        def can_add(u, v):
            """
            Check if adding edge (u, v) creates a C6.
            A C6 exists if there is a path of length 4 from u to some node w
            (w != v, w != u, all internal nodes distinct and not u/v),
            and w is adjacent to v.
            Uses iterative DFS with bitmask for visited nodes.
            """
            # Quick filter: check if any neighbor of v has a length-3 path to u
            # through nodes not including u or v
            # This is a necessary but not sufficient condition
            
            # Use iterative DFS with (node, depth, visited_mask)
            start_mask = 1 << u
            stack = [(u, 0, start_mask)]
            
            while stack:
                node, depth, mask = stack.pop()
                
                if depth == 4:
                    # Check if node is adjacent to v
                    if adj_bits[node] & (1 << v):
                        return False
                    continue
                
                # Get neighbors of node, excluding v and nodes already in path
                neighbors = adj_bits[node] & ~(1 << v) & ~mask
                
                while neighbors:
                    low = neighbors & (-neighbors)
                    nb = low.bit_length() - 1
                    neighbors ^= low
                    stack.append((nb, depth + 1, mask | (1 << nb)))
            
            return True
        
        def find_blocking_c6(u, v):
            """
            Find a C6 that would be created by adding (u, v).
            Returns list of edges forming the C6, or None.
            """
            start_mask = 1 << u
            stack = [(u, 0, start_mask, [u])]
            
            while stack:
                node, depth, mask, path = stack.pop()
                
                if depth == 4:
                    if adj_bits[node] & (1 << v):
                        # Found: path u..node, and node-v edge
                        c6_edges = []
                        for i in range(len(path) - 1):
                            c6_edges.append((path[i], path[i + 1]))
                        c6_edges.append((node, v))
                        return c6_edges
                    continue
                
                neighbors = adj_bits[node] & ~(1 << v) & ~mask
                while neighbors:
                    low = neighbors & (-neighbors)
                    nb = low.bit_length() - 1
                    neighbors ^= low
                    stack.append((nb, depth + 1, mask | (1 << nb), path + [nb]))
            
            return None
        
        def compute_priority():
            """
            For each missing edge, compute a priority score.
            Higher score = more likely to be addable and useful.
            """
            missing = []
            for u, v in all_pairs:
                if not (adj_bits[u] & (1 << v)):
                    missing.append((u, v))
            
            if not missing:
                return []
            
            scores = []
            for u, v in missing:
                du = (adj_bits[u]).bit_count()
                dv = (adj_bits[v]).bit_count()
                # Common neighbors: proxy for C4 density
                common = (adj_bits[u] & adj_bits[v]).bit_count()
                # Prefer edges between high-degree nodes, penalize common neighbors
                score = (du + dv) * 1.0 - common * 2.5
                score += random.random() * 0.1
                scores.append((score, u, v))
            
            scores.sort(reverse=True)
            return [(u, v) for _, u, v in scores]
        
        def get_degree(u):
            return adj_bits[u].bit_count()
        
        def get_edge_count():
            return len(edges_list)
        
        # Initial graph: 8 disjoint K5s (104 edges)
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)
        
        local_best = len(edges_list)
        local_best_bits = adj_bits[:]
        local_best_edges = edges_list[:]
        
        # Simulated annealing parameters
        temperature = 5.0
        cooling_rate = 0.98
        min_temperature = 0.1
        
        # Local search loop
        while time.time() < deadline:
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            # Shuffle for diversity
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if adj_bits[u] & (1 << v):
                    continue
                
                if can_add(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges_list) > local_best:
                        local_best = len(edges_list)
                        local_best_bits = adj_bits[:]
                        local_best_edges = edges_list[:]
            
            # Final polish pass: try all missing edges once more
            if added_in_pass > 0:
                for u, v in all_pairs:
                    if time.time() > deadline:
                        break
                    if adj_bits[u] & (1 << v):
                        continue
                    if can_add(u, v):
                        add_edge(u, v)
                        if len(edges_list) > local_best:
                            local_best = len(edges_list)
                            local_best_bits = adj_bits[:]
                            local_best_edges = edges_list[:]
            
            if added_in_pass == 0:
                # Stuck: targeted kick with simulated annealing
                if ranked_missing and edges_list:
                    # Try multiple kick strategies
                    kick_success = False
                    
                    # Strategy 1: Remove edges from blocking C6
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = find_blocking_c6(target_u, target_v)
                    if block_c6:
                        # Remove 1-3 edges from the blocking C6
                        num_removals = random.randint(1, 3)
                        random.shuffle(block_c6)
                        old_count = len(edges_list)
                        for _ in range(num_removals):
                            if not edges_list:
                                break
                            e_to_remove = block_c6.pop()
                            remove_edge(e_to_remove[0], e_to_remove[1])
                        
                        # Check if we can now add more edges
                        new_added = 0
                        for u2, v2 in ranked_missing:
                            if time.time() > deadline:
                                break
                            if adj_bits[u2] & (1 << v2):
                                continue
                            if can_add(u2, v2):
                                add_edge(u2, v2)
                                new_added += 1
                        
                        # Simulated annealing acceptance
                        delta = len(edges_list) - old_count
                        if delta >= 0 or random.random() < pow(2.718, delta / max(temperature, 0.1)):
                            kick_success = True
                            if len(edges_list) > local_best:
                                local_best = len(edges_list)
                                local_best_bits = adj_bits[:]
                                local_best_edges = edges_list[:]
                        else:
                            # Reject: we'd need to undo, but for simplicity,
                            # we'll just accept if it doesn't make things too bad
                            if delta > -3:
                                kick_success = True
                            else:
                                # Revert by rebuilding from local_best
                                # For simplicity, just continue
                                pass
                    
                    if not kick_success and edges_list:
                        # Fallback: remove a random edge
                        e = random.choice(edges_list)
                        remove_edge(e[0], e[1])
                    
                    # Cool down
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature
        
        # Update global best
        if local_best > best_count:
            best_count = local_best
            final_edges = []
            for u in range(N):
                mask = local_best_bits[u]
                while mask:
                    low = mask & (-mask)
                    v = low.bit_length() - 1
                    mask ^= low
                    if u < v:
                        final_edges.append([u, v])
            best_edges_list = final_edges
    
    # Run multiple restarts
    num_restarts = 4
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        run_once(42 + i * 1000)
    
    if not best_edges_list:
        return []
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
