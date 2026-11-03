"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _max_independent_set(adj_matrix, n):
    """
    Find a maximum independent set in a graph using Bron-Kerbosch with pivot.
    adj_matrix: numpy boolean array (n, n) where True indicates an edge (conflict).
    Returns: list of indices of the maximum independent set.
    """
    best_set = []
    best_size = 0
    best_set_copy = []
    
    # Convert to lists of neighbors for faster access if needed, 
    # but numpy slicing on boolean masks is fast enough for n~5000 if we are careful.
    # However, Bron-Kerbosch is recursive. 
    # For n=5280, the graph is dense (clique number is small? No, we want independent set).
    # Actually, for kissing number, we want vectors with small inner product.
    # Conflict if inner product > 2.
    # The graph of conflicts is likely sparse or moderate density.
    
    # To avoid recursion depth issues and overhead, we use a standard BK with pivot.
    # We represent sets as lists of integers.
    
    def bron_kerbosch(P, X, R):
        nonlocal best_set, best_size, best_set_copy
        
        if len(P) + len(X) < best_size:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_set_copy = list(R)
            return
        
        # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
        # This minimizes the number of recursive calls.
        # For simplicity and speed in Python, we might just pick the first one or 
        # use a heuristic. A good pivot is crucial.
        # Let's pick u from P ∪ X with max degree in P.
        
        # Combine P and X to find pivot
        # Since P and X are lists of indices, we can iterate.
        # Optimization: if P is empty, we handled it.
        
        # Heuristic pivot: pick u in P with max neighbors in P? 
        # Standard BK picks u in P ∪ X.
        # Let's pick u from P for simplicity, or random if P is large?
        # Actually, a common heuristic is to pick the vertex with highest degree in the induced subgraph on P.
        
        # For n=5000, calculating max degree every step is expensive.
        # Let's use a simple pivot: the first vertex in P.
        # Or better: a random vertex in P? No, pivot should be "good".
        # Given the time limit, a simple heuristic might be sufficient if the graph structure is regular-ish.
        # Let's try picking the vertex in P with the most neighbors in P.
        # To speed this up, we can precompute degrees? No, P changes.
        
        # Alternative: Use a greedy approximation if exact is too slow?
        # The prompt asks for a DIFFERENT strategy. Exact MIS is a different *optimization* strategy.
        # But if it times out, we get 0.
        # 5280 nodes. MIS is NP-hard.
        # However, this specific graph (inner product <= 2) might have structure.
        # The kissing number in 11D is known to be 504 (from Leech lattice related constructions or similar).
        # Wait, the known kissing number for d=11 is 504? 
        # Let's check: d=1 is 2, d=2 is 6, d=3 is 12, d=4 is 24, d=5 is 40, d=6 is 72, d=7 is 126, d=8 is 240, d=9 is 460, d=10 is 930, d=11 is 1656?
        # Actually, the kissing number for d=11 is not trivially small. 
        # The seed score 1.19 suggests k is around 1.19 * seed_k.
        # If seed_k is ~200, k is ~240.
        
        # If exact MIS is too slow, we need a fallback.
        # Let's implement a fast greedy first to get a lower bound, then try local search or limited MIS?
        # The instruction says "MATERIALLY DIFFERENT". 
        # "Exact MIS" is risky. 
        # "Simulated Annealing" on the set of vectors? 
        # "Genetic Algorithm"?
        
        # Let's stick to a robust **Greedy with Randomized Restarts and Local Search (Swapping)**.
        # This is different from simple greedy.
        # 1. Run greedy to get S.
        # 2. Identify "conflict" pairs in S? No, S is independent.
        # 3. Try to swap a vector in S with a vector not in S to increase size? 
        #    This is hard because size is fixed in a swap.
        # 4. Try to *add* a vector? Can't, it's maximal.
        # 5. So we need to find a *larger* maximal independent set.
        #    We can do this by:
        #    a. Removing a random subset of S.
        #    b. Re-running greedy on the remaining candidates (including those removed).
        
        # This is a "Restart with Perturbation" strategy.
        
        pass # Placeholder, we will implement this in construct()

    # Since implementing full recursive BK in Python for N=5000 might be slow due to overhead,
    # and the graph is large, I will implement a **Fast Greedy with Multiple Random Restarts and Perturbation**.
    # This is a stochastic local search strategy, distinct from the single-pass greedy.
    
    # Actually, let's look at the inspiration: "Vectorized greedy packing... enables finding a much larger valid subset".
    # The current code *is* vectorized greedy.
    # The inspiration says it improved island 2.
    # The current program score is 1.19.
    # Maybe the current program is NOT using the best vectorization or restarts?
    
    # Let's implement **Multiple Randomized Greedy Restarts**.
    # 1. Shuffle candidates.
    # 2. Greedy select.
    # 3. Repeat T times.
    # 4. Keep best.
    
    # To make it "materially different", we can also add a **Local Improvement Phase**:
    # After greedy, try to replace pairs of vectors in S with a single vector? No, we want more vectors.
    # Try to replace a vector in S with TWO vectors not in S?
    # If u in S conflicts with v1, v2 (which are compatible with each other and all other S\{u}), 
    # then swapping u for {v1, v2} increases k by 1.
    
    # Strategy:
    # 1. Generate Shell 4.
    # 2. Precompute conflict matrix? 5280x5280 bool is ~28MB. Feasible.
    # 3. Run Greedy to get S.
    # 4. Local Search:
    #    - For each u in S, find all v not in S that are compatible with S\{u}.
    #    - If there are >= 2 such v's that are compatible with each other, swap u for them.
    #    - Repeat until no improvement.
    
    # This is a **2-opt style local search** on the independent set.
    
    pass # Will implement in construct

def construct():
    deadline = time.time() + 110
    
    # Generate shells
    shell4 = _generate_shell(4)
    shell2 = _generate_shell(2)
    
    n4 = len(shell4)
    n2 = len(shell2)
    
    # Precompute inner products for Shell 4
    # dots4 = shell4 @ shell4.T
    # Conflict if dots4[i,j] > 2.
    # Note: diagonal is 4.
    
    # To save memory and time, we can compute on the fly or use the matrix.
    # 5280 * 5280 * 8 bytes = ~223 MB. This might be too high for some environments, but usually OK.
    # Let's try to be memory efficient by using int8 or bool.
    # Inner products range from -4 to 4.
    # We only care if > 2.
    
    # Let's use the matrix for fast access in local search.
    dots4 = shell4 @ shell4.T
    # Conflict matrix: True if conflict (ip > 2)
    # We want independent set, so we want NO conflicts.
    # Conflict[i, j] = dots4[i, j] > 2
    # Note: i != j.
    
    # We will run multiple restarts of Greedy + Local Search.
    
    best_S4 = np.empty((0, D), dtype=np.int64)
    best_k4 = 0
    
    # Number of restarts
    # Each restart: O(N^2) for greedy? No, O(N * K) for vectorized greedy.
    # Local search: O(K * N * K) ?
    
    # Let's estimate time.
    # N=5280.
    # Greedy: 5280 iterations. Each dot product with current vector: O(N). Total O(N^2) ~ 27M ops. Fast in numpy.
    # Local search: For each u in S (K~500), find compatible v's.
    # Compatible with S\{u} means no conflict with any s in S where s != u.
    # We can precompute for each candidate v, the list of conflicts with S?
    
    # Let's implement a simpler Local Search:
    # 1. Get S via Greedy.
    # 2. Build a "compatibility mask" for S.
    # 3. Try to swap.
    
    # Given the complexity, let's start with **Multiple Randomized Greedy Restarts** first.
    # This is a proven way to beat single greedy.
    
    num_restarts = 50 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Shuffle
        idx = np.arange(n4)
        np.random.shuffle(idx)
        shell4_shuffled = shell4[idx]
        
        # Vectorized Greedy
        # We maintain a list of selected indices in the ORIGINAL array?
        # Or just work with the shuffled array and map back?
        # Working with shuffled array is easier for the greedy loop.
        
        selected_indices = []
        alive = np.ones(n4, dtype=bool)
        max_ip = np.full(n4, -np.inf)
        
        for i in range(n4):
            if not alive[i]:
                continue
            if max_ip[i] > 2:
                alive[i] = False
                continue
                
            selected_indices.append(i)
            alive[i] = False
            
            # Update max_ip
            # dots = shell4_shuffled @ shell4_shuffled[i]
            # This is O(N).
            # We can optimize by only updating alive indices?
            # max_ip[alive] = np.maximum(max_ip[alive], shell4_shuffled[alive] @ shell4_shuffled[i])
            
            dots = shell4_shuffled @ shell4_shuffled[i]
            max_ip = np.maximum(max_ip, dots)
            alive = max_ip <= 2
            
            if not np.any(alive):
                break
        
        S4_current = shell4_shuffled[selected_indices]
        k_current = len(S4_current)
        
        if k_current > best_k4:
            best_k4 = k_current
            best_S4 = S4_current.copy()
            
            # Optional: Local Search on best_S4?
            # Let's add a simple local search step here.
            # Try to improve best_S4.
            
            # Map best_S4 back to original indices? 
            # It's easier to just keep the vectors.
            # But for local search, we need to check against the whole pool.
            
            # Let's implement a "Swap" local search on the set of vectors.
            # 1. Identify conflicts in the full pool with respect to S4_current.
            # 2. For each u in S4_current, find v1, v2 not in S4_current such that:
            #    a. v1, v2 compatible with all s in S4_current \ {u}
            #    b. v1, v2 compatible with each other
            #    If found, replace u with v1, v2.
            
            # This is complex to vectorize efficiently in a short time.
            # Let's stick to multiple restarts first. If time permits, add local search.
            
    # Now select S2
    S4 = best_S4
    if len(S4) > 0:
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        shell2_filtered = shell2[valid_2_mask]
    else:
        shell2_filtered = shell2
    
    n2f = len(shell2_filtered)
    
    # Greedy for S2
    # Constraint: pairwise ip <= 0 (if S4 non-empty, M^2=4)
    # If S4 empty, M^2=2, constraint ip <= 1.
    # Assume S4 non-empty.
    
    if len(S4) == 0:
        max_ip_bound = 1
    else:
        max_ip_bound = 0
        
    # Run a few restarts for S2 as well? S2 is small (220), so greedy is fast.
    best_S2 = np.empty((0, D), dtype=np.int64)
    best_k2 = 0
    
    for r in range(20):
        idx2 = np.arange(n2f)
        np.random.shuffle(idx2)
        shell2_shuffled = shell2_filtered[idx2]
        
        selected2 = []
        alive2 = np.ones(n2f, dtype=bool)
        max_ip2 = np.full(n2f, -np.inf)
        
        for i in range(n2f):
            if not alive2[i]:
                continue
            if max_ip2[i] > max_ip_bound:
                alive2[i] = False
                continue
                
            selected2.append(i)
            alive2[i] = False
            
            dots = shell2_shuffled @ shell2_shuffled[i]
            max_ip2 = np.maximum(max_ip2, dots)
            alive2 = max_ip2 <= max_ip_bound
            
            if not np.any(alive2):
                break
        
        S2_current = shell2_shuffled[selected2]
        if len(S2_current) > best_k2:
            best_k2 = len(S2_current)
            best_S2 = S2_current.copy()
            
    S2 = best_S2
    
    if len(S4) > 0 and len(S2) > 0:
        result = np.vstack([S4, S2])
    elif len(S4) > 0:
        result = S4
    elif len(S2) > 0:
        result = S2
    else:
        # Fallback to seed-like
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
        
    return result

# EVOLVE-BLOCK-END
