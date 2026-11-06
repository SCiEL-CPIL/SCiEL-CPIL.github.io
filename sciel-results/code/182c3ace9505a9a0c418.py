import itertools
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit

    # Helper to generate Paley graph
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Fallback: Paley 37
    n_seed, edges_seed = get_paley(37)

    # Verify a graph has no K5 and no independent 5-set
    def is_valid(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for combo in itertools.combinations(range(n), 5):
            is_k5 = True
            is_i5 = True
            for i in range(5):
                u = combo[i]
                for j in range(i + 1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                return False
        return True

    # Try to find an n-vertex (5,5)-free graph using local search
    def try_find(n, max_attempts=8, max_steps=3000, seed_base=0):
        # Precompute 5-sets and edge-to-5set mapping
        all_5sets = list(itertools.combinations(range(n), 5))
        edge_to_5sets = {}
        for idx, combo in enumerate(all_5sets):
            for i in range(5):
                for j in range(i + 1, 5):
                    edge = (combo[i], combo[j])
                    if edge not in edge_to_5sets:
                        edge_to_5sets[edge] = []
                    edge_to_5sets[edge].append(idx)
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout - 10:
                break
            
            random.seed(seed_base + attempt)
            
            # Initialize with random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Count initial violations
            current_viol = 0
            for combo in all_5sets:
                is_k5 = True
                is_i5 = True
                for i in range(5):
                    u = combo[i]
                    for j in range(i + 1, 5):
                        v = combo[j]
                        if adj[u] & (1 << v):
                            is_i5 = False
                        else:
                            is_k5 = False
                        if not is_k5 and not is_i5:
                            break
                    if not is_k5 and not is_i5:
                        break
                if is_k5 or is_i5:
                    current_viol += 1
            
            # Local search with simulated annealing
            temp = 1.0
            cooling = 0.999
            for step in range(max_steps):
                if time.time() - start_time > timeout - 10:
                    break
                
                if current_viol == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append((i, j))
                    return n, edges
                
                # Pick random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                # Get 5-sets containing this edge
                affected = edge_to_5sets.get((i, j), [])
                
                # Count violations in affected 5-sets before flip
                viol_before = 0
                for idx in affected:
                    combo = all_5sets[idx]
                    is_k5 = True
                    is_i5 = True
                    for a in range(5):
                        u = combo[a]
                        for b in range(a + 1, 5):
                            v = combo[b]
                            if adj[u] & (1 << v):
                                is_i5 = False
                            else:
                                is_k5 = False
                            if not is_k5 and not is_i5:
                                break
                        if not is_k5 and not is_i5:
                            break
                    if is_k5 or is_i5:
                        viol_before += 1
                
                # Flip edge
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
                # Count violations in affected 5-sets after flip
                viol_after = 0
                for idx in affected:
                    combo = all_5sets[idx]
                    is_k5 = True
                    is_i5 = True
                    for a in range(5):
                        u = combo[a]
                        for b in range(a + 1, 5):
                            v = combo[b]
                            if adj[u] & (1 << v):
                                is_i5 = False
                            else:
                                is_k5 = False
                            if not is_k5 and not is_i5:
                                break
                        if not is_k5 and not is_i5:
                            break
                    if is_k5 or is_i5:
                        viol_after += 1
                
                # Accept if better or with probability based on temperature
                delta_viol = viol_after - viol_before
                if delta_viol <= 0:
                    current_viol += delta_viol
                else:
                    # Accept with probability exp(-delta/temp)
                    prob = np.exp(-delta_viol / temp)
                    if random.random() < prob:
                        current_viol += delta_viol
                    else:
                        # Revert
                        adj[i] ^= (1 << j)
                        adj[j] ^= (1 << i)
                
                # Cool down
                temp *= cooling
            
            if current_viol == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges
        
        return None, None

    # Try to find 42-vertex graph, then 41, 40, etc.
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() - start_time > timeout - 30:
            break
        
        n_found, edges_found = try_find(target_n, max_attempts=8, max_steps=3000, seed_base=target_n * 100)
        
        if n_found == target_n and edges_found is not None:
            if is_valid(target_n, edges_found):
                return target_n, edges_found

    # Fallback: return the Paley 37
    return n_seed, edges_seed

# Test runner
if __name__ == "__main__":
    n, edges = construct()
    print(f"Vertices: {n}, Edges: {len(edges)}")
