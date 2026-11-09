import random
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all possible triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    
    # Precompute bitmasks for each triple
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        
    def calc_alpha(triples):
        """
        Calculate the independence number of the hypergraph defined by triples.
        Uses superset marking on a bitmask of size 2^16.
        """
        size = 1 << n
        has_triple = bytearray(size)
        
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            # Mark all supersets of mask
            rest = ((1 << n) - 1) ^ mask
            sub = rest
            while True:
                has_triple[mask | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & rest
        
        best = 0
        # Iterate through all subsets to find the largest one without a triple
        for S in range(size):
            if has_triple[S] == 0:
                c = S.bit_count()
                if c > best:
                    best = c
        return best

    def find_maximal_linear_triples(order):
        """
        Backtracking search to find a maximal linear triple system.
        'order' is a list of indices into all_triples, defining the preference order.
        """
        used_pairs = [[0] * n for _ in range(n)]
        current_triples = []
        
        def backtrack(idx):
            if idx == len(order):
                return current_triples[:]
            
            t_idx = order[idx]
            i, j, k = all_triples[t_idx]
            
            # Check linearity: no pair (i,j), (i,k), (j,k) used before
            if used_pairs[i][j] == 0 and used_pairs[i][k] == 0 and used_pairs[j][k] == 0:
                # Add triple
                used_pairs[i][j] = 1
                used_pairs[i][k] = 1
                used_pairs[j][k] = 1
                current_triples.append(t_idx)
                
                # Recurse
                res = backtrack(idx + 1)
                
                # Backtrack
                current_triples.pop()
                used_pairs[i][j] = 0
                used_pairs[i][k] = 0
                used_pairs[j][k] = 0
                
                if res is not None:
                    return res
            else:
                # Skip this triple
                return backtrack(idx + 1)
                
        return backtrack(0)

    best_alpha = 16
    best_triples = []
    rng = random.Random(42)
    
    # We will try to find a good maximal linear triple system.
    # Since the space of MLTS is large, we sample different orderings.
    # A greedy approach often works well if the order is randomized.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Create a random permutation of triple indices
        order = list(range(num_triples))
        rng.shuffle(order)
        
        # To make the search faster and potentially find denser systems,
        # we can bias the order. However, random permutation is a good baseline.
        # We can also try sorting by some heuristic, but random is robust.
        
        result_indices = find_maximal_linear_triples(order)
        
        if result_indices is None:
            continue
            
        # Convert indices to actual triples
        current_triples = [all_triples[i] for i in result_indices]
        
        # Calculate independence number
        alpha = calc_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples
            # If we reach a very low alpha (e.g., 6), we can stop early as it's likely optimal or near-optimal
            if best_alpha <= 6:
                break

    if not best_triples:
        # Fallback: return a simple valid system
        best_triples = []
        # Just to ensure we return *something* valid
        # A single triple is valid
        best_triples = [(0, 1, 2)]
        
    return [list(t) for t in best_triples]
