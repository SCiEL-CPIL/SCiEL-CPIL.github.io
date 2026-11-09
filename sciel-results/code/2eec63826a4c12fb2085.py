import time
import random
from itertools import combinations, product

def construct():
    n = 18
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill

    def compute_L(adj):
        """Compute the largest triangle-free induced subgraph size using efficient backtracking."""
        # Precompute common neighbors
        common = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common[i][j] = common[j][i] = adj[i] & adj[j]
        
        # For checking if a set S is triangle-free:
        # S is triangle-free iff for every edge (u,v) in S, common[u][v] & S == 0
        
        def is_triangle_free(mask):
            """Check if vertex set (bitmask) induces a triangle-free subgraph."""
            temp = mask
            while temp:
                u = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                # Check all edges from u to other vertices in mask
                neighbors_in_mask = adj[u] & mask
                while neighbors_in_mask:
                    v = (neighbors_in_mask & -neighbors_in_mask).bit_length() - 1
                    neighbors_in_mask &= neighbors_in_mask - 1
                    if v > u and (common[u][v] & mask):
                        return False
            return True
        
        # Binary search on k: find max k such that there exists a k-vertex triangle-free set
        # Use backtracking with pruning
        def backtrack_k(k, mask, next_v, count):
            """Try to build a triangle-free set of size k."""
            if count == k:
                return True
            remaining = n - next_v
            if count + remaining < k:
                return False
            
            for v in range(next_v, n):
                if mask & (1 << v):
                    continue
                # Check if adding v keeps the set triangle-free
                # v must not form a triangle with any pair in mask
                valid = True
                temp = mask
                while temp:
                    u = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    if (adj[u] & (1 << v)) and (common[u][v] & mask):
                        valid = False
                        break
                if valid:
                    if backtrack_k(k, mask | (1 << v), v + 1, count + 1):
                        return True
            return False
        
        # Try from k=n down to 1
        for k in range(n, 2, -1):
            if backtrack_k(k, 0, 0, 0):
                return k
        return 2

    def is_k4_free(adj):
        """Check if graph is K4-free using bitmasks."""
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b+1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                        continue
                    common = adj[a] & adj[b] & adj[c]
                    if common:
                        return False
        return True

    def make_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def blow_up(base_adj, part_sizes, n_total):
        """Blow up a base graph: replace each vertex with a clique-free part.
        base_adj: list of bitmasks for the base graph
        part_sizes: list of part sizes, sum must equal n_total
        Returns adjacency list for the blown-up graph.
        """
        base_n = len(base_adj)
        adj = [0] * n_total
        # Assign vertices to parts
        part_of = [0] * n_total
        part_start = [0] * base_n
        for i in range(base_n):
            part_start[i] = sum(part_sizes[:i])
        
        idx = 0
        for part in range(base_n):
            for _ in range(part_sizes[part]):
                part_of[idx] = part
                idx += 1
        
        # Build edges: connect all vertices in part i to all vertices in part j
        # if there's an edge between i and j in the base graph
        for i in range(base_n):
            for j in range(i+1, base_n):
                if base_adj[i] & (1 << j):
                    for u in range(part_start[i], part_start[i] + part_sizes[i]):
                        for v in range(part_start[j], part_start[j] + part_sizes[j]):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj

    def part_size_distributions(total, k):
        """Generate all compositions of total into k positive parts."""
        if k == 1:
            yield [total]
            return
        for first in range(1, total - k + 2):
            for rest in part_size_distributions(total - first, k - 1):
                yield [first] + rest

    # Define small K4-free base graphs
    # C5: cycle on 5 vertices
    c5_adj = [0]*5
    for i in range(5):
        c5_adj[i] |= (1 << ((i+1) % 5))
        c5_adj[i] |= (1 << ((i-1) % 5))
    
    # C7: cycle on 7 vertices
    c7_adj = [0]*7
    for i in range(7):
        c7_adj[i] |= (1 << ((i+1) % 7))
        c7_adj[i] |= (1 << ((i-1) % 7))
    
    # C6: cycle on 6 vertices
    c6_adj = [0]*6
    for i in range(6):
        c6_adj[i] |= (1 << ((i+1) % 6))
        c6_adj[i] |= (1 << ((i-1) % 6))
    
    # C8: cycle on 8 vertices
    c8_adj = [0]*8
    for i in range(8):
        c8_adj[i] |= (1 << ((i+1) % 8))
        c8_adj[i] |= (1 << ((i-1) % 8))
    
    # C9: cycle on 9 vertices
    c9_adj = [0]*9
    for i in range(9):
        c9_adj[i] |= (1 << ((i+1) % 9))
        c9_adj[i] |= (1 << ((i-1) % 9))
    
    # C10: cycle on 10 vertices
    c10_adj = [0]*10
    for i in range(10):
        c10_adj[i] |= (1 << ((i+1) % 10))
        c10_adj[i] |= (1 << ((i-1) % 10))

    base_graphs = {
        'C5': c5_adj,
        'C6': c6_adj,
        'C7': c7_adj,
        'C8': c8_adj,
        'C9': c9_adj,
        'C10': c10_adj,
    }

    best_L = n + 1
    best_adj = None

    # Try blow-ups of each base graph with various part size distributions
    for name, base_adj in base_graphs.items():
        base_n = len(base_adj)
        
        # Try balanced and near-balanced distributions
        base_size = n // base_n
        remainder = n % base_n
        
        # Generate candidate part sizes
        candidates = []
        
        # Balanced: as equal as possible
        balanced = [base_size] * base_n
        for i in range(remainder):
            balanced[i] += 1
        candidates.append(balanced)
        
        # Try some variations
        if base_n >= 3:
            # One part larger, others smaller
            for i in range(base_n):
                trial = [base_size] * base_n
                trial[i] += remainder
                if sum(trial) == n:
                    candidates.append(trial)
            
            # Two parts larger
            if base_n >= 4:
                for i in range(base_n):
                    for j in range(i+1, base_n):
                        trial = [base_size] * base_n
                        trial[i] += 1
                        trial[j] += 1
                        if sum(trial) == n:
                            candidates.append(trial)
        
        # Also try some random distributions
        random.seed(42)
        for _ in range(50):
            parts = [1] * base_n
            for _ in range(n - base_n):
                parts[random.randint(0, base_n-1)] += 1
            candidates.append(parts)
        
        for part_sizes in candidates:
            if sum(part_sizes) != n or any(p < 1 for p in part_sizes):
                continue
            if time.time() > deadline - 5:
                break
            
            adj = blow_up(base_adj, part_sizes, n)
            if not is_k4_free(adj):
                continue
            
            L = compute_L(adj)
            if L < best_L:
                best_L = L
                best_adj = adj[:]
                print(f"{name} blow-up {part_sizes}: L = {L}", flush=True)
        
        if time.time() > deadline - 5:
            break

    # If no blow-up found, use Turan as fallback
    if best_adj is None:
        adj_turan = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if i % 3 != j % 3:
                    adj_turan[i] |= (1 << j)
                    adj_turan[j] |= (1 << i)
        best_adj = adj_turan
        best_L = compute_L(adj_turan)
        print(f"Turan fallback: L = {best_L}", flush=True)

    # Local search refinement: try edge flips to reduce L further
    if best_adj is not None and time.time() < deadline - 10:
        current_adj = best_adj[:]
        current_L = best_L
        
        for round_num in range(30):
            if time.time() > deadline - 5:
                break
            improved = False
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges_to_try)
            
            for i, j in edges_to_try:
                if time.time() > deadline - 5:
                    break
                old_state = bool(current_adj[i] & (1 << j))
                
                if old_state:
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
                else:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                
                if is_k4_free(current_adj):
                    L = compute_L(current_adj)
                    if L < current_L:
                        current_L = L
                        best_L = L
                        best_adj = current_adj[:]
                        improved = True
                        print(f"Local search round {round_num}: improved L to {L}", flush=True)
                    else:
                        if old_state:
                            current_adj[i] |= (1 << j)
                            current_adj[j] |= (1 << i)
                        else:
                            current_adj[i] &= ~(1 << j)
                            current_adj[j] &= ~(1 << i)
                else:
                    if old_state:
                        current_adj[i] |= (1 << j)
                        current_adj[j] |= (1 << i)
                    else:
                        current_adj[i] &= ~(1 << j)
                        current_adj[j] &= ~(1 << i)
            
            if not improved:
                break

    return make_edge_list(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
