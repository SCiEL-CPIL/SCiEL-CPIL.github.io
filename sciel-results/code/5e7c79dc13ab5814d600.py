import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    total_pairs = n * (n - 1) // 2
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # If u is in N(a) and N(b), then u is added to N({v, a, b}).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def spectral_construct(random_seed):
        """Bipartite spectral construction: partition into two sets of 10,
        use eigenvectors of random matrix to determine edges."""
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Partition vertices into two sets of 10
        left = list(range(10))
        right = list(range(10, 20))
        
        adj = [0] * n
        
        # Generate random 10x10 matrix
        M = np.random.randn(10, 10)
        
        # Get eigenvectors (columns of V)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(M)
        except:
            return greedy_construct(random_seed)
        
        # Use the first few eigenvectors to determine edge pattern
        # For each pair (i in left, j in right), compute a score based on
        # dot products with eigenvectors, then threshold
        for i in left:
            for j in right:
                j_local = j - 10  # 0-9
                # Compute score using top eigenvectors
                score = 0
                for k in range(min(3, 10)):
                    score += eigenvectors[i, k] * eigenvectors[j_local, k]
                
                # Threshold: if score > 0, add edge
                # This naturally limits common neighbors across partition
                if score > 0.1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Also add some intra-partition edges based on eigenvectors
        for i in range(10):
            for j in range(i+1, 10):
                score = 0
                for k in range(min(2, 10)):
                    score += eigenvectors[i, k] * eigenvectors[j, k]
                if score > 0.15:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Validate and clean up
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    if not is_valid_addition([0]*n, i, j):
                        pass  # Will be handled by local search
        
        return adj

    def local_search(adj, max_steps=200, use_kick=False):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Phase 1: Greedy Add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
            
        # Phase 2: Edge Swaps (1 in, 2 out)
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if not edges:
                break
            random.shuffle(edges)
            
            found_swap = False
            for u, v in edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                added = []
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j)) and (i != u or j != v) and (i != v or j != u)]
                random.shuffle(missing)
                
                for a, b in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added.append((a, b))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                    found_swap = True
                elif len(added) == 1:
                    current_edges += 0
                    found_swap = True
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        # Phase 3: Kick + Greedy Fill (if enabled)
        if use_kick:
            for _ in range(3):
                if time.time() > deadline:
                    break
                
                # Kick: randomly remove and add some edges
                kick_adj = current_adj[:]
                kick_edges = current_edges
                
                # Remove 2-4 random edges
                kick_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if kick_adj[i] & (1 << j)]
                random.shuffle(kick_edges_list)
                for u, v in kick_edges_list[:random.randint(2, 4)]:
                    kick_adj[u] &= ~(1 << v)
                    kick_adj[v] &= ~(1 << u)
                    kick_edges -= 1
                
                # Greedy fill
                improved = True
                while improved:
                    if time.time() > deadline:
                        break
                    improved = False
                    pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (kick_adj[i] & (1 << j))]
                    random.shuffle(pairs)
                    
                    for u, v in pairs:
                        if time.time() > deadline:
                            break
                        if is_valid_addition(kick_adj, u, v):
                            kick_adj[u] |= (1 << v)
                            kick_adj[v] |= (1 << u)
                            kick_edges += 1
                            improved = True
                
                if kick_edges > best_edges:
                    best_edges = kick_edges
                    best_adj = kick_adj[:]
                    current_adj = kick_adj[:]
                    current_edges = kick_edges
        
        return best_adj, best_edges

    def perturb_adj(adj, num_perturb=3):
        """Create a perturbed version of adj for escaping local optima."""
        perturbed = adj[:]
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if perturbed[i] & (1 << j)]
        random.shuffle(edges)
        
        # Remove some edges
        for u, v in edges[:num_perturb]:
            perturbed[u] &= ~(1 << v)
            perturbed[v] &= ~(1 << u)
        
        return perturbed

    best_adj = [0] * n
    best_edges = 0
    
    # Strategy 1: Spectral construction + local search
    num_spectral_restarts = 200
    for i in range(num_spectral_restarts):
        if time.time() > deadline:
            break
        
        adj = spectral_construct(i * 7 + 13)
        adj, edges = local_search(adj, max_steps=100, use_kick=True)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Strategy 2: Random greedy + local search with kicks
    num_random_restarts = 3000
    for i in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i * 31 + 7)
        adj, edges = local_search(adj, max_steps=150, use_kick=(i % 5 == 0))
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Strategy 3: Perturbation of best solution
    if time.time() < deadline - 10:
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            perturbed = perturb_adj(best_adj, random.randint(2, 5))
            adj, edges = local_search(perturbed, max_steps=200, use_kick=True)
            
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
    
    # Final polish on the best solution
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj, max_steps=300, use_kick=True)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
