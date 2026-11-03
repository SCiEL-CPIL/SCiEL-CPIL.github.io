import numpy as np
import itertools
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    D = 11
    
    # 1. Generate Candidates
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
            
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Neighbors
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    conflict_bool = (dot > 2)
    np.fill_diagonal(conflict_bool, False)
    
    neighbors = [np.where(conflict_bool[i])[0] for i in range(n)]
    
    # 3. Greedy Min-Degree with Incremental Updates
    rng = np.random.default_rng(42)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    num_restarts = 500 # Will be limited by time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize
        remaining = np.ones(n, dtype=np.uint8)
        deg = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)
        
        # We need a way to efficiently pick min deg from remaining.
        # Using a heap is tricky with updates.
        # Simple scan is O(N). N=5280.
        # 5280 * 5280 = 27M ops per restart.
        # 500 restarts = 13.5B ops. Too slow in Python.
        
        # We need a faster method.
        # Bucket sort by degree?
        # Max degree is 56.
        # We can maintain a list of lists: `buckets[d]` contains vertices with degree d.
        
        buckets = [[] for _ in range(n + 1)]
        for i in range(n):
            buckets[deg[i]].append(i)
            
        current_min_deg = 0
        selected = np.zeros(n, dtype=np.uint8)
        count = 0
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Find min degree in remaining
            # Scan buckets from current_min_deg upwards
            while current_min_deg <= n:
                if buckets[current_min_deg]:
                    # Pick a random vertex from this bucket
                    # Filter to only those still remaining
                    valid_verts = [v for v in buckets[current_min_deg] if remaining[v]]
                    if valid_verts:
                        break
                current_min_deg += 1
                
            if current_min_deg > n:
                break
                
            v = valid_verts[rng.integers(len(valid_verts))]
            
            # Select v
            selected[v] = 1
            remaining[v] = 0
            count += 1
            
            # Update neighbors
            for u in neighbors[v]:
                if remaining[u]:
                    remaining[u] = 0
                    # Update degree of u's neighbors
                    for w in neighbors[u]:
                        if remaining[w]:
                            deg[w] -= 1
                            # Move w from bucket[deg[w]+1] to bucket[deg[w]]
                            # This is O(1) if we just append to new bucket and mark old as invalid?
                            # Simpler: just update deg[w] and let the bucket scan handle it?
                            # No, bucket lists become stale.
                            
                    # Actually, the "remove from bucket" is expensive.
                    # Let's just use a simple array scan for min-degree.
                    # It's O(N) per step.
                    # Total O(N^2).
                    # 27M ops.
                    # In Python, 27M simple ops takes ~1-2 seconds.
                    # 500 restarts is too many.
                    # Let's limit restarts to ~50.
                    
            # This approach is getting complicated.
            # Let's stick to the original code's structure but fix the local search or remove it if greedy is good enough.
            # The original code found 244.
            # If we just run greedy 50 times, we might find a better set.
            
        # Let's try a different tactic: 
        # The graph is sparse.
        # We can use a simple "First Fit" or "Random Order" greedy.
        # Random order greedy is O(N+E).
        
        pass

    # Let's implement Random Order Greedy with multiple restarts.
    # It's very fast.
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    for r in range(1000):
        if time.time() > deadline - 5.0:
            break
            
        order = rng.permutation(n)
        selected = np.zeros(n, dtype=np.uint8)
        count = 0
        
        for v in order:
            if selected[v]:
                continue
            # Check if v conflicts with any selected
            # We can maintain a "blocked" set.
            # blocked[u] = 1 if u conflicts with any selected.
            # Initially all 0.
            # When we select v, we block neighbors[v].
            
            # But we need to check if v is blocked.
            # If not blocked, select it and block its neighbors.
            
            # This requires a "blocked" array.
            
            pass

    # Let's do the blocked array approach.
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    for r in range(2000):
        if time.time() > deadline - 5.0:
            break
            
        order = rng.permutation(n)
        blocked = np.zeros(n, dtype=np.uint8)
        selected = np.zeros(n, dtype=np.uint8)
        count = 0
        
        for v in order:
            if blocked[v]:
                continue
            selected[v] = 1
            count += 1
            # Block neighbors
            for u in neighbors[v]:
                blocked[u] = 1
                
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
            
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
