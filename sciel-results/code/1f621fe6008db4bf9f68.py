import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Adjacency as bitmasks
        adj_mask = [0] * N
        adj_set = [set() for _ in range(N)]
        edges = []
        
        def add_edge(u, v):
            if (adj_mask[u] >> v) & 1:
                return False
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_set[u].add(v)
            adj_set[v].add(u)
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if not ((adj_mask[u] >> v) & 1):
                return False
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_set[u].discard(v)
            adj_set[v].discard(u)
            try:
                edges.remove((u, v))
            except ValueError:
                edges.remove((v, u))
            return True
        
        def has_c6_fast(u, v):
            """
            Fast check: does adding edge (u,v) create a C6?
            Need simple path u-n1-n2-n3-n4-v.
            """
            # Bit-parallel BFS over-approximation (ignore simple path constraint)
            # If over-approx says no, definitely no C6
            layer1 = 0
            m = 1 << u
            while m:
                low = m & (-m)
                node = low.bit_length() - 1
                m ^= low
                layer1 |= (adj_mask[node] & ~(1 << v))
            layer1 &= ~(1 << u)
            
            if layer1 == 0:
                return False
            
            layer2 = 0
            m = layer1
            while m:
                low = m & (-m)
                node = low.bit_length() - 1
                m ^= low
                layer2 |= (adj_mask[node] & ~(1 << v))
            layer2 &= ~(layer1 | (1 << u))
            
            if layer2 == 0:
                return False
            
            layer3 = 0
            m = layer2
            while m:
                low = m & (-m)
                node = low.bit_length() - 1
                m ^= low
                layer3 |= (adj_mask[node] & ~(1 << v))
            layer3 &= ~(layer1 | layer2 | (1 << u))
            
            if layer3 == 0:
                return False
            
            layer4 = 0
            m = layer3
            while m:
                low = m & (-m)
                node = low.bit_length() - 1
                m ^= low
                layer4 |= (adj_mask[node] & ~(1 << v))
            layer4 &= ~(layer1 | layer2 | layer3 | (1 << u))
            
            if layer4 == 0:
                return False
            
            v_neigh = adj_mask[v] & ~(1 << u)
            if (layer4 & v_neigh) == 0:
                return False
            
            # Over-approximation says possible. Do targeted DFS for simple path.
            # Find path u -> n1 -> n2 -> n3 -> n4 where n4 adjacent to v, all distinct
            def dfs(node, depth, visited):
                if depth == 4:
                    return (adj_mask[node] & (1 << v)) != 0
                nm = adj_mask[node] & ~visited & ~(1 << v)
                m = nm
                while m:
                    low = m & (-m)
                    nb = low.bit_length() - 1
                    m ^= low
                    if dfs(nb, depth + 1, visited | (1 << nb)):
                        return True
                return False
            
            return dfs(u, 0, 1 << u)
        
        def is_c6_free(u, v):
            return not has_c6_fast(u, v)
        
        # --- Greedy phase ---
        def compute_priority():
            missing = [(u, v) for u, v in all_pairs if not ((adj_mask[u] >> v) & 1)]
            if not missing:
                return []
            
            scores = []
            for u, v in missing:
                du = bin(adj_mask[u]).count('1')
                dv = bin(adj_mask[v]).count('1')
                common = bin(adj_mask[u] & adj_mask[v]).count('1')
                score = (du + dv) * 1.0 - common * 3.0
                score += random.random() * 0.5
                scores.append((score, u, v))
            
            scores.sort(reverse=True)
            return [(u, v) for _, u, v in scores]
        
        # Initial: random sparse graph
        for _ in range(20):
            u, v = random.sample(range(N), 2)
            if is_c6_free(u, v):
                add_edge(u, v)
        
        local_best = len(edges)
        local_best_adj = [set(s) for s in adj_set]
        
        # Greedy local search
        while time.time() < deadline - 15:
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline - 15:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                if is_c6_free(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = [set(s) for s in adj_set]
            
            if added_in_pass == 0:
                # Perturbation: remove random edge(s)
                if edges:
                    num_rem = random.randint(1, 3)
                    for _ in range(num_rem):
                        if edges:
                            e = random.choice(edges)
                            remove_edge(e[0], e[1])
        
        # --- Simulated Annealing phase ---
        # Proper SA: accept worse moves with probability exp(-delta/T)
        if time.time() < deadline - 5:
            T = 3.0
            cooling = 0.997
            min_T = 0.05
            current_edges = len(edges)
            
            while T > min_T:
                if time.time() > deadline - 5:
                    break
                
                # Generate a random move
                move_type = random.random()
                if move_type < 0.4:
                    # Try to add a random missing edge
                    u, v = random.sample(range(N), 2)
                    if (adj_mask[u] >> v) & 1:
                        T *= cooling
                        continue
                    if is_c6_free(u, v):
                        add_edge(u, v)
                        delta = 1  # gained an edge
                        # Always accept improvement
                        if len(edges) > local_best:
                            local_best = len(edges)
                            local_best_adj = [set(s) for s in adj_set]
                    else:
                        # Would create C6, reject
                        pass
                elif move_type < 0.8:
                    # Remove a random edge
                    if not edges:
                        T *= cooling
                        continue
                    e = random.choice(edges)
                    remove_edge(e[0], e[1])
                    delta = -1  # lost an edge
                    # Accept with probability exp(-delta/T) = exp(1/T)
                    # Since delta is negative, -delta is positive, so exp(positive) > 1, always accept
                    # Wait, standard SA: accept if exp(-delta/T) > random, where delta = new_energy - old_energy
                    # Here energy = -num_edges (we want to maximize edges)
                    # So delta = new_energy - old_energy = -new_edges - (-old_edges) = old_edges - new_edges
                    # If we remove an edge: new_edges = old_edges - 1, delta = old_edges - (old_edges-1) = 1
                    # Accept if exp(-1/T) > random
                    if random.random() > np.exp(-1.0 / T):
                        # Reject: re-add the edge
                        add_edge(e[0], e[1])
                else:
                    # Swap: remove one edge, try to add another
                    if not edges:
                        T *= cooling
                        continue
                    e1 = random.choice(edges)
                    remove_edge(e1[0], e1[1])
                    # Try to add a random missing edge
                    added = False
                    for _ in range(10):
                        u, v = random.sample(range(N), 2)
                        if (adj_mask[u] >> v) & 1:
                            continue
                        if is_c6_free(u, v):
                            add_edge(u, v)
                            added = True
                            break
                    if not added:
                        # Re-add the removed edge
                        add_edge(e1[0], e1[1])
                
                T *= cooling
        
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
    seeds = [42, 123, 7, 999, 31415, 271, 808, 555, 111, 222]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        run_once(s)
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
