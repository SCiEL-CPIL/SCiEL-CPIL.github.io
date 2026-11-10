import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Combines:
    - Bitmask-based fast C6 detection (can_add)
    - Priority ranking for edge addition order
    - Targeted kicks: remove edges from the blocking C6
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
        edges = []  # list of (u, v) with u < v
        
        def add_edge(u, v):
            if u > v:
                u, v = v, u
            if adj_bits[u] & (1 << v):
                return False
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            if not (adj_bits[u] & (1 << v)):
                return False
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            edges.remove((u, v))
            return True
        
        def get_neighbors(u, exclude_mask=0):
            """Return list of neighbors of u not in exclude_mask."""
            mask = adj_bits[u] & ~exclude_mask
            result = []
            while mask:
                low = mask & (-mask)
                node = low.bit_length() - 1
                mask ^= low
                result.append(node)
            return result
        
        def can_add(u, v):
            """
            Check if adding edge (u, v) creates a C6.
            A C6 exists if there is a path of length 4 from u to some node w
            (w != v, w != u, all internal nodes distinct and not u/v),
            and w is adjacent to v.
            Uses iterative DFS with bitmask for visited nodes.
            """
            # We need a path u - a - b - c - d where d is adjacent to v,
            # with all of a,b,c,d distinct and not u or v.
            # Equivalently: path of length 4 from u avoiding v, ending at a node adjacent to v.
            
            # Use iterative DFS with (node, depth, visited_mask)
            # visited_mask includes all nodes on the current path
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
        
        # Initial graph: 8 disjoint K5s (104 edges)
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)
        
        local_best = len(edges)
        local_best_bits = adj_bits[:]
        
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
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_bits = adj_bits[:]
            
            # Final polish pass: try all missing edges once more
            if added_in_pass > 0:
                for u, v in all_pairs:
                    if time.time() > deadline:
                        break
                    if adj_bits[u] & (1 << v):
                        continue
                    if can_add(u, v):
                        add_edge(u, v)
                        if len(edges) > local_best:
                            local_best = len(edges)
                            local_best_bits = adj_bits[:]
            
            if added_in_pass == 0:
                # Stuck: targeted kick
                if ranked_missing and edges:
                    # Pick a random missing edge and find its blocking C6
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = find_blocking_c6(target_u, target_v)
                    if block_c6:
                        # Remove 1-2 edges from the blocking C6
                        num_removals = random.randint(1, 2)
                        random.shuffle(block_c6)
                        for _ in range(num_removals):
                            if not edges:
                                break
                            e_to_remove = block_c6.pop()
                            remove_edge(e_to_remove[0], e_to_remove[1])
                    elif edges:
                        # Fallback: remove a random edge
                        e = random.choice(edges)
                        remove_edge(e[0], e[1])
        
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
