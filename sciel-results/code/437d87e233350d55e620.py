import random
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Constructs a graph on n vertices with no regular induced subgraph on k >= 6 vertices.
    Uses vectorized NumPy-based violation checking and targeted local search.
    """
    start_time = time.time()
    time_limit = 140.0  # 10s safety margin for 150s limit

    def count_violations_6(adj_matrix, n):
        """
        Vectorized check: count how many 6-vertex subsets induce a regular subgraph.
        adj_matrix: n x n binary numpy array
        Returns: (count, list_of_violation_indices) - indices into precomputed combos
        """
        if n < 6:
            return 0, []
        
        # Precompute all 6-combinations as index arrays
        combos = np.array(list(combinations(range(n), 6)), dtype=np.int32)
        if len(combos) == 0:
            return 0, []
        
        # For each combo, compute degrees within the subset
        # combos shape: (num_combos, 6)
        # adj_matrix[combos] would be tricky; let's use einsum or manual
        
        # For each vertex in each combo, count neighbors within the combo
        # degree[i][j] = sum over k in combo_i of adj_matrix[combo_i[j], combo_i[k]]
        # Use: for each combo row, for each position j, sum adj_matrix[row[j], row[k]] for k != j
        
        num_combos = len(combos)
        # Build index arrays for vectorized access
        # For position j (0..5), we want adj_matrix[combos[:, j], combos[:, k]] for k != j
        # This is complex; let's do it differently.
        
        # Alternative: for each combo, the induced subgraph is adj_matrix[np.ix_(combo, combo)]
        # We need to check if all row sums (minus self) are equal.
        
        # Vectorized approach:
        # degrees[c, j] = sum_{k != j} adj_matrix[combos[c, j], combos[c, k]]
        
        degrees = np.zeros((num_combos, 6), dtype=np.int32)
        for j in range(6):
            for k in range(6):
                if j == k:
                    continue
                # adj_matrix[combos[:, j], combos[:, k]] gives shape (num_combos,)
                degrees[:, j] += adj_matrix[combos[:, j], combos[:, k]]
        
        # Check if all 6 degrees are equal
        # A subset is regular if min(degree) == max(degree)
        min_deg = degrees.min(axis=1)
        max_deg = degrees.max(axis=1)
        is_regular = (min_deg == max_deg)
        
        count = int(np.sum(is_regular))
        if count == 0:
            return 0, []
        
        # Return indices of violations (for targeted search)
        viol_indices = np.where(is_regular)[0]
        return count, viol_indices

    def find_violations_full(adj_matrix, n):
        """
        Check all subsets of size >= 6. For practical purposes, checking size 6 first
        and then verifying larger subsets only if needed.
        Returns total violation count.
        """
        # Check size 6
        count6, _ = count_violations_6(adj_matrix, n)
        if count6 > 0:
            return count6  # Short-circuit: if any size-6 violation, we have violations
        
        # Check size 7
        if n >= 7:
            combos7 = np.array(list(combinations(range(n), 7)), dtype=np.int32)
            num_combos = len(combos7)
            degrees = np.zeros((num_combos, 7), dtype=np.int32)
            for j in range(7):
                for k in range(7):
                    if j == k:
                        continue
                    degrees[:, j] += adj_matrix[combos7[:, j], combos7[:, k]]
            min_deg = degrees.min(axis=1)
            max_deg = degrees.max(axis=1)
            is_regular = (min_deg == max_deg)
            count7 = int(np.sum(is_regular))
            if count7 > 0:
                return count6 + count7
        
        # Check size 8 and above (less common in practice)
        for k in range(8, n + 1):
            combos_k = np.array(list(combinations(range(n), k)), dtype=np.int32)
            num_combos = len(combos_k)
            degrees = np.zeros((num_combos, k), dtype=np.int32)
            for j in range(k):
                for kk in range(k):
                    if j == kk:
                        continue
                    degrees[:, j] += adj_matrix[combos_k[:, j], combos_k[:, kk]]
            min_deg = degrees.min(axis=1)
            max_deg = degrees.max(axis=1)
            is_regular = (min_deg == max_deg)
            count_k = int(np.sum(is_regular))
            if count_k > 0:
                return count6 + count7 + count_k
        
        return 0

    def try_local_search(n, initial_adj_list, deadline):
        """
        Local search: flip edges to eliminate regular induced subgraphs.
        Uses targeted edge selection based on violation analysis.
        Returns (success, final_adj_list)
        """
        # Convert to numpy adjacency matrix
        adj_matrix = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                adj_matrix[i, j] = (initial_adj_list[i] >> j) & 1
        # Zero out diagonal
        np.fill_diagonal(adj_matrix, 0)
        
        current_adj = adj_matrix.copy()
        best_adj = current_adj.copy()
        
        # Count initial violations
        best_count = find_violations_full(current_adj, n)
        if best_count == 0:
            return True, list(current_adj)
        
        stagnation_counter = 0
        max_stagnation = 50
        
        while time.time() < deadline:
            # Check for violations
            current_count, viol_indices = count_violations_6(current_adj, n)
            
            if current_count == 0:
                # Check larger subsets
                full_count = find_violations_full(current_adj, n)
                if full_count == 0:
                    return True, list(current_adj)
                # If only larger subsets are violated, try random flips
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                current_adj[u, v] ^= 1
                current_adj[v, u] ^= 1
                new_count = find_violations_full(current_adj, n)
                if new_count < best_count:
                    best_adj = current_adj.copy()
                    best_count = new_count
                    stagnation_counter = 0
                else:
                    current_adj[u, v] ^= 1
                    current_adj[v, u] ^= 1
                    stagnation_counter += 1
                continue
            
            # We have size-6 violations. Target them.
            # Pick a random violation
            if len(viol_indices) > 0:
                viol_idx = random.choice(viol_indices)
                # Get the combo
                combo = list(combinations(range(n), 6))
                verts = combo[viol_idx]
                
                # For this regular subgraph, find which edge flip would break regularity
                # Strategy: pick two vertices in the subgraph with the same degree,
                # and flip an edge between them or an edge from one of them to outside.
                
                # Compute degrees within the subgraph
                sub_adj = current_adj[np.ix_(verts, verts)]
                np.fill_diagonal(sub_adj, 0)
                degs = sub_adj.sum(axis=1)
                
                # All degrees are equal (regular). Pick a random pair to flip.
                # Flipping an internal edge changes degrees of 2 vertices by +-1 each.
                # This breaks regularity unless the graph has special structure.
                
                u, v = random.sample(verts, 2)
                
                # Flip edge u-v
                current_adj[u, v] ^= 1
                current_adj[v, u] ^= 1
                
                # Check improvement
                new_count6, _ = count_violations_6(current_adj, n)
                new_full_count = find_violations_full(current_adj, n)
                
                if new_full_count < best_count:
                    best_adj = current_adj.copy()
                    best_count = new_full_count
                    stagnation_counter = 0
                else:
                    # Revert
                    current_adj[u, v] ^= 1
                    current_adj[v, u] ^= 1
                    stagnation_counter += 1
                    # Try a different strategy: flip edge from a vertex in subgraph to outside
                    if n > 6:
                        outside_verts = [x for x in range(n) if x not in verts]
                        if outside_verts:
                            u = random.choice(verts)
                            v = random.choice(outside_verts)
                            current_adj[u, v] ^= 1
                            current_adj[v, u] ^= 1
                            new_full_count = find_violations_full(current_adj, n)
                            if new_full_count < best_count:
                                best_adj = current_adj.copy()
                                best_count = new_full_count
                                stagnation_counter = 0
                            else:
                                current_adj[u, v] ^= 1
                                current_adj[v, u] ^= 1
                                stagnation_counter += 1
            
            # Check stagnation
            if stagnation_counter > max_stagnation:
                # Random restart: perturb the graph
                for _ in range(n // 2):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        current_adj[u, v] ^= 1
                        current_adj[v, u] ^= 1
                stagnation_counter = 0
        
        # Return best found
        return False, list(best_adj)

    def adj_list_to_edges(adj_list, n):
        """Convert adjacency list to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_list[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Main strategy: try to find valid graphs for n = 22, 21, 20
    best_result = {'n': 1, 'edges': []}
    
    for target_n in [22, 21, 20]:
        if time.time() > time_limit - 15:
            break
        
        success = False
        max_restarts = 8 if target_n <= 21 else 5
        
        for r in range(max_restarts):
            if time.time() > time_limit - 10:
                break
            
            # Initialize with random graph (slightly biased toward sparsity to avoid dense regular subgraphs)
            adj_list = [0] * target_n
            p = 0.4  # Slightly less than 0.5 to reduce chance of regular subgraphs
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj_list[i] |= (1 << j)
                        adj_list[j] |= (1 << i)
            
            # Local search
            ok, final_adj_list = try_local_search(target_n, adj_list, time_limit)
            
            if ok:
                # Final verification
                adj_matrix = np.zeros((target_n, target_n), dtype=np.int32)
                for i in range(target_n):
                    for j in range(target_n):
                        adj_matrix[i, j] = (final_adj_list[i] >> j) & 1
                np.fill_diagonal(adj_matrix, 0)
                
                if find_violations_full(adj_matrix, target_n) == 0:
                    edges = adj_list_to_edges(final_adj_list, target_n)
                    best_result = {'n': target_n, 'edges': edges}
                    success = True
                    break
        
        if success and target_n == 22:
            break
    
    # Fallback: greedy construction for smaller n
    if best_result['n'] < 10 and time.time() < time_limit - 5:
        current_n = 1
        current_adj_list = [0]
        for target_n in range(2, 16):
            if time.time() > time_limit - 2:
                break
            found = False
            for _ in range(50):
                new_adj_list = current_adj_list[:] + [0]
                for i in range(current_n):
                    if random.random() < 0.4:
                        new_adj_list[i] |= (1 << current_n)
                        new_adj_list[current_n] |= (1 << i)
                
                # Quick check
                adj_matrix = np.zeros((target_n, target_n), dtype=np.int32)
                for i in range(target_n):
                    for j in range(target_n):
                        adj_matrix[i, j] = (new_adj_list[i] >> j) & 1
                np.fill_diagonal(adj_matrix, 0)
                
                if find_violations_full(adj_matrix, target_n) == 0:
                    current_adj_list = new_adj_list
                    current_n = target_n
                    found = True
                    break
            if not found:
                break
        
        if current_n > best_result['n']:
            edges = adj_list_to_edges(current_adj_list, current_n)
            best_result = {'n': current_n, 'edges': edges}

    return best_result

if __name__ == "__main__":
    construct()
