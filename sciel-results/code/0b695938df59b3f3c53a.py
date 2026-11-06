import itertools
import time
import random

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit

    def elapsed():
        return time.time() - start_time

    # Helper to generate Paley graph
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Fast (5,5)-free check using bitmask DFS
    def has_k5(n, adj):
        """Check if graph has a clique of size 5 using backtracking."""
        def dfs(clique, candidates):
            if len(clique) == 5:
                return True
            if not candidates:
                return False
            # Try each candidate
            cands = list(candidates)
            for i, v in enumerate(cands):
                new_cands = cands[i+1:]  # only vertices after v to avoid duplicates
                new_cands = [w for w in new_cands if (adj[v] >> w) & 1]
                if dfs(clique + [v], new_cands):
                    return True
            return False
        return dfs([], list(range(n)))

    def has_i5(n, adj):
        """Check if graph has an independent set of size 5."""
        # Independent set in G = clique in complement
        comp_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    comp_adj[i] |= (1 << j)
                    comp_adj[j] |= (1 << i)
        def dfs(clique, candidates):
            if len(clique) == 5:
                return True
            if not candidates:
                return False
            cands = list(candidates)
            for i, v in enumerate(cands):
                new_cands = cands[i+1:]
                new_cands = [w for w in new_cands if (comp_adj[v] >> w) & 1]
                if dfs(clique + [v], new_cands):
                    return True
            return False
        return dfs([], list(range(n)))

    def is_55_free(n, edges):
        if elapsed() > timeout:
            return False
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        if has_k5(n, adj):
            return False
        if has_i5(n, adj):
            return False
        return True

    # Fallback: Paley 37
    n_seed, edges_seed = get_paley(37)

    # 1. Try Paley(41)
    n_paley41, edges_41 = get_paley(41)
    if is_55_free(n_paley41, edges_41):
        return n_paley41, edges_41

    # 2. Local search with bitmasks and incremental violation counting
    def count_violations(n, adj):
        """Count number of K5s and I5s in the graph."""
        count = 0
        # Precompute all 5-sets
        all_5sets = list(itertools.combinations(range(n), 5))
        for combo in all_5sets:
            is_k5 = True
            is_i5 = True
            for i in range(5):
                u = combo[i]
                for j in range(i + 1, 5):
                    v = combo[j]
                    if (adj[u] >> v) & 1:
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                count += 1
        return count, all_5sets

    def local_search(n, time_budget, seed_base=0):
        """Local search to find (5,5)-free graph on n vertices."""
        deadline = start_time + time_budget
        all_5sets = list(itertools.combinations(range(n), 5))
        
        # Precompute for each edge, which 5-sets it affects
        edge_to_5sets = {}
        for idx, combo in enumerate(all_5sets):
            for i in range(5):
                for j in range(i + 1, 5):
                    edge = (combo[i], combo[j])
                    if edge not in edge_to_5sets:
                        edge_to_5sets[edge] = []
                    edge_to_5sets[edge].append(idx)

        def count_viol_for_5sets(adj, indices):
            count = 0
            for idx in indices:
                combo = all_5sets[idx]
                is_k5 = True
                is_i5 = True
                for i in range(5):
                    u = combo[i]
                    for j in range(i + 1, 5):
                        v = combo[j]
                        if (adj[u] >> v) & 1:
                            is_i5 = False
                        else:
                            is_k5 = False
                        if not is_k5 and not is_i5:
                            break
                    if not is_k5 and not is_i5:
                        break
                if is_k5 or is_i5:
                    count += 1
            return count

        for attempt in range(10):
            if elapsed() > deadline:
                break
            
            random.seed(seed_base + attempt)
            
            # Start from a random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_viol = count_violations(n, adj)[0]
            
            # Simulated annealing
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            
            while elapsed() < deadline and T > min_T:
                if current_viol == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append((i, j))
                    return n, edges
                
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                edge_key = (i, j)
                affected = edge_to_5sets.get(edge_key, [])
                
                viol_before = count_viol_for_5sets(adj, affected)
                
                # Flip edge
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
                viol_after = count_viol_for_5sets(adj, affected)
                
                delta = viol_after - viol_before
                
                if delta <= 0 or random.random() < (T / max(1, abs(delta))):
                    current_viol += delta
                else:
                    # Revert
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                
                T *= cooling
            
            if current_viol == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append((i, j))
                return n, edges
        
        return None, None

    # Try to find larger graphs
    remaining_time = timeout - elapsed()
    if remaining_time > 30:
        # Allocate time to different target sizes
        targets = [38, 39, 40, 41]
        time_per_target = remaining_time / len(targets)
        
        for target_n in targets:
            if elapsed() > timeout - 20:
                break
            t_budget = min(time_per_target, timeout - elapsed() - 15)
            if t_budget < 5:
                break
            
            n_found, edges_found = local_search(target_n, t_budget, seed_base=target_n * 100)
            
            if n_found == target_n and edges_found is not None:
                if is_55_free(target_n, edges_found):
                    return target_n, edges_found

    # Fallback: return the Paley 37
    return n_seed, edges_seed

# Test runner
if __name__ == "__main__":
    n, edges = construct()
    print(f"Vertices: {n}, Edges: {len(edges)}")
