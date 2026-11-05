import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_eulerian_circuit(adj, start):
        # adj: dict of list of neighbors (with duplicates for multi-edges)
        # Returns list of vertices
        stack = [start]
        circuit = []
        adj_copy = {u: list(v) for u, v in adj.items()}
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                w = adj_copy[v].pop()
                # Remove reverse edge
                # Find w in adj_copy[v] is not possible, we popped it.
                # We need to remove one w from adj_copy[w]
                try:
                    adj_copy[w].remove(v)
                except ValueError:
                    # Should not happen if consistent
                    pass
                stack.append(w)
            else:
                circuit.append(stack.pop())
                
        circuit.reverse()
        return circuit

    def decompose_graph(edges, n, max_time_per_instance=30.0):
        if not edges:
            return []
            
        adj = defaultdict(list)
        degrees = defaultdict(int)
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            degrees[u] += 1
            degrees[v] += 1
            edge_set.add(tuple(sorted((u, v))))
            
        vertices = list(range(n))
        odd_vertices = [v for v in vertices if degrees[v] % 2 != 0]
        
        if len(odd_vertices) > 0:
            K_min = len(odd_vertices) // 2
        else:
            K_min = 2 # 1 is impossible
            
        best_paths = None
        best_count = float('inf')
        
        for K in range(K_min, K_min + 5):
            if time.time() > deadline - 10:
                break
                
            # Construct S
            S = []
            if len(odd_vertices) > 0:
                S = list(odd_vertices)
                even_vertices = [v for v in vertices if degrees[v] % 2 == 0]
                rng = random.Random(42 + K)
                rng.shuffle(even_vertices)
                while len(S) < 2 * K:
                    # Add pairs of even vertices
                    if even_vertices:
                        u = even_vertices.pop()
                        S.append(u)
                        S.append(u)
                    else:
                        break
            else:
                even_vertices = list(vertices)
                rng = random.Random(42 + K)
                rng.shuffle(even_vertices)
                while len(S) < 2 * K:
                    if even_vertices:
                        u = even_vertices.pop()
                        S.append(u)
                        S.append(u)
                    else:
                        break
                        
            if len(S) < 2 * K:
                continue
                
            # Try pairings
            found = False
            for attempt in range(100):
                if time.time() > deadline - 10:
                    break
                    
                rng = random.Random(attempt * 100 + K)
                endpoints = list(S)
                rng.shuffle(endpoints)
                pairs = [(endpoints[i], endpoints[i+1]) for i in range(0, 2*K, 2)]
                
                # Build augmented adjacency
                aug_adj = {u: list(v) for u, v in adj.items()}
                for u, v in pairs:
                    aug_adj[u].append(v)
                    aug_adj[v].append(u)
                    
                # Find circuit
                start_node = pairs[0][0]
                circuit = find_eulerian_circuit(aug_adj, start_node)
                
                # Extract paths
                # Aux edges are pairs
                # Note: If u==v, it's a loop? No, simple graph.
                # If pairs has (u,v) and (u,v), we have 2 aux edges.
                
                aux_edges = set()
                # We need to distinguish between the two (u,v) edges if they are same.
                # But for extraction, we just check if edge is in pairs.
                # If (u,v) appears twice, we need to cut twice.
                
                # Count aux edges
                aux_counts = defaultdict(int)
                for u, v in pairs:
                    aux_counts[tuple(sorted((u, v)))] += 1
                    
                paths = []
                # Walk circuit
                current_path = [circuit[0]]
                for i in range(1, len(circuit)):
                    u = circuit[i-1]
                    v = circuit[i]
                    e = tuple(sorted((u, v)))
                    
                    if aux_counts.get(e, 0) > 0:
                        # Cut
                        if len(current_path) >= 2:
                            paths.append(current_path)
                        current_path = [v]
                        aux_counts[e] -= 1
                    else:
                        current_path.append(v)
                        
                # Closing edge
                u = circuit[-1]
                v = circuit[0]
                e = tuple(sorted((u, v)))
                
                if aux_counts.get(e, 0) > 0:
                    if len(current_path) >= 2:
                        paths.append(current_path)
                    aux_counts[e] -= 1
                else:
                    current_path.append(v) # This should not happen if counts are correct
                    # Actually, if closing edge is not aux, the path wraps?
                    # No, the circuit is closed.
                    # If the closing edge is not aux, it means the last segment connects to the first?
                    # This implies the first and last segments are one path.
                    # But we cut at aux edges.
                    # If the number of aux edges is K, we should get K paths.
                    
                # Validate
                if len(paths) == K:
                    used_edges = set()
                    valid = True
                    for p in paths:
                        if len(set(p)) != len(p):
                            valid = False
                            break
                        for j in range(len(p) - 1):
                            ej = tuple(sorted((p[j], p[j+1])))
                            if ej in used_edges:
                                valid = False
                                break
                            used_edges.add(ej)
                            if ej not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                            
                    if valid and used_edges == edge_set:
                        best_paths = paths
                        best_count = K
                        found = True
                        break
                        
            if found:
                break
                
        if best_paths is None:
            best_paths = [[u, v] for u, v in edges]
            best_count = len(edges)
            
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        edges = instance(n, p, seed)
        paths = decompose_graph(edges, n)
        results.append(paths)
        
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
