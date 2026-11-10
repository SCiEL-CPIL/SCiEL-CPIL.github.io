import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 118  # Safety margin (evaluator kills at 150s)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Adjacency as sets and lists
        adj_set = [set() for _ in range(N)]
        adj_list = [[] for _ in range(N)]
        edges = []
        
        def add_edge(u, v):
            if v in adj_set[u]:
                return False
            adj_list[u].append(v)
            adj_list[v].append(u)
            adj_set[u].add(v)
            adj_set[v].add(u)
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if v not in adj_set[u]:
                return False
            adj_set[u].remove(v)
            adj_set[v].remove(u)
            adj_list[u].remove(v)
            adj_list[v].remove(u)
            try:
                edges.remove((u, v))
            except ValueError:
                edges.remove((v, u))
            return True
        
        def get_blocking_c6(u, v):
            """
            Returns a list of 5 edges forming a C6 with (u,v), or None.
            Fast path: bitmask BFS to check reachability at depth 5,
            then targeted DFS to extract the actual path.
            """
            # Bitmask BFS: reachable[d] = bitmask of nodes reachable in exactly d steps
            # from u (without going through v)
            # We need a path u -> a -> b -> c -> d -> v of length 5
            # So we need d reachable from u in 4 steps, and d adjacent to v.
            
            # Build neighbor bitmasks (excluding v)
            def neighbors_mask(node):
                m = 0
                for nb in adj_list[node]:
                    if nb != v:
                        m |= (1 << nb)
                return m
            
            # BFS layers from u
            layers = [0] * 6  # layers[d] = bitmask of nodes at distance d
            layers[0] = 1 << u
            
            for d in range(1, 5):
                mask = 0
                prev = layers[d-1]
                # For each node in prev, OR its neighbors
                # Iterate set bits
                m = prev
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    mask |= neighbors_mask(node)
                # Remove nodes already in earlier layers (to keep it a simple path)
                # Actually for C6 we need a simple path, so we should not revisit
                # But bitmask BFS without simple-path constraint may over-estimate.
                # For a quick check, let's just see if any neighbor of v is in layers[4]
                layers[d] = mask
            
            # Check: is there a node d in layers[4] that is adjacent to v?
            v_neighbors = 0
            for nb in adj_list[v]:
                if nb != u:  # u is the start, not a valid intermediate
                    v_neighbors |= (1 << nb)
            
            if (layers[4] & v_neighbors) == 0:
                return None  # No C6 possible
            
            # Now do a targeted DFS to find the actual path
            # DFS from u, depth 0..4, avoid v until the end
            stack = [(u, 0, [u])]
            while stack:
                node, depth, path = stack.pop()
                if depth == 4:
                    if v in adj_set[node]:
                        # Found path
                        c6_edges = []
                        for i in range(len(path) - 1):
                            c6_edges.append((path[i], path[i+1]))
                        c6_edges.append((node, v))
                        return c6_edges
                    continue
                for nb in adj_list[node]:
                    if nb == v:
                        continue
                    if nb in path:
                        continue
                    stack.append((nb, depth + 1, path + [nb]))
            return None
        
        def is_c6_free(u, v):
            return get_blocking_c6(u, v) is None
        
        # --- Greedy phase with spectral ranking ---
        def compute_spectral_priority():
            """
            For each missing edge, estimate a spectral score.
            We use the fact that adding edge (u,v) changes the adjacency matrix.
            Approximate score: degree-based proxy that correlates with spectral gap.
            Higher degree difference = more "hub" = potentially more spectral gap.
            Also penalize edges that would create many C4s (proxy for cycle density).
            """
            missing = [(u, v) for u, v in all_pairs if v not in adj_set[u]]
            if not missing:
                return []
            
            scores = []
            for u, v in missing:
                du = len(adj_set[u])
                dv = len(adj_set[v])
                # Common neighbors (proxy for C4 creation)
                common = len(adj_set[u] & adj_set[v])
                # Score: prefer edges between high-degree nodes, penalize common neighbors
                score = (du + dv) * 1.0 - common * 2.0
                # Small random tiebreaker
                score += random.random() * 0.1
                scores.append((score, u, v))
            
            scores.sort(reverse=True)
            return [(u, v) for _, u, v in scores]
        
        # Initial graph: 8 disjoint K5s
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(base + a, base + b)
        
        local_best = len(edges)
        local_best_adj = [set(s) for s in adj_set]
        
        # Local search loop
        while time.time() < deadline:
            # Get ranked missing edges
            ranked_missing = compute_spectral_priority()
            if not ranked_missing:
                break
            
            # Shuffle within score bands for diversity
            # Just shuffle the whole list for now
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if v in adj_set[u]:
                    continue
                
                block_c6 = get_blocking_c6(u, v)
                if block_c6 is None:
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = [set(s) for s in adj_set]
            
            if added_in_pass == 0:
                # Stuck: directed kick
                if ranked_missing:
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = get_blocking_c6(target_u, target_v)
                    if block_c6:
                        e_to_remove = random.choice(block_c6)
                        remove_edge(e_to_remove[0], e_to_remove[1])
                    elif edges:
                        e = random.choice(edges)
                        remove_edge(e[0], e[1])
        
        # Update global best
        if local_best > best_count:
            best_count = local_best
            best_edges_list = []
            for u in range(N):
                for v in local_best_adj[u]:
                    if u < v:
                        best_edges_list.append([u, v])
        
        return local_best
    
    # Multi-restart
    seeds = [42, 123, 7, 999, 31415]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        run_once(s)
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
