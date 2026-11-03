import itertools
import time
import random

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1) using Lagrangian relaxation on the distance-2 graph.
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Handle non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            # Fallback: return all words (valid but large)
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Precompute word-to-index mapping
    def word_to_idx(w):
        idx = 0
        for i, val in enumerate(reversed(w)):
            idx += val * (3 ** i)
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % 3
            temp //= 3
        return w
    
    total_words = 3 ** n
    
    # Precompute spheres (radius-1 neighborhoods)
    spheres = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        sphere = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbor_idx = word_to_idx(w)
                sphere.append(neighbor_idx)
                w[pos] = orig_val
        spheres[idx] = sphere
    
    # Build distance-2 graph: edges connect pairs at Hamming distance 2
    # For the minimum weight vertex cover on G^2, we need to find a set of vertices
    # such that every edge has at least one endpoint in the set.
    # The complement is an independent set in G^2.
    
    # For efficiency, we'll use a Lagrangian relaxation approach:
    # 1. Start with a feasible solution (tetracode-based)
    # 2. Iteratively adjust weights based on uncovered points
    # 3. Solve the resulting problem via greedy independent set
    
    # Initial solution: tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    # Convert to indices
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    best_code = initial_codewords[:]
    best_size = len(best_code)
    
    # Lagrangian relaxation parameters
    lambda_weights = [1.0] * total_words  # Initial weights for uncovered points
    
    for iteration in range(200):  # Max iterations
        if time.time() > deadline:
            break
        
        # Step 1: Compute coverage counts
        counts = [0] * total_words
        for idx in initial_indices:
            for neighbor in spheres[idx]:
                counts[neighbor] += 1
        
        # Step 2: Update Lagrangian multipliers
        # Points with low coverage get higher weight (penalty for being uncovered)
        for i in range(total_words):
            if counts[i] == 0:
                lambda_weights[i] = min(lambda_weights[i] * 1.5, 100.0)
            elif counts[i] == 1:
                lambda_weights[i] = max(lambda_weights[i] * 0.8, 0.1)
            else:
                lambda_weights[i] = max(lambda_weights[i] * 0.95, 0.1)
        
        # Step 3: Greedy independent set on distance-2 graph with weighted vertices
        # A vertex v is "selectable" if removing it doesn't leave any point uncovered
        # We want to find the maximum independent set in G^2, which corresponds to
        # the minimum vertex cover complement.
        
        # For each codeword, compute its "cost" based on how many points it uniquely covers
        # weighted by lambda_weights
        removal_priority = []
        for idx in initial_indices:
            unique_coverage = 0
            for neighbor in spheres[idx]:
                if counts[neighbor] == 1:
                    unique_coverage += lambda_weights[neighbor]
            removal_priority.append((unique_coverage, idx))
        
        # Sort by priority (lowest unique coverage first - these are good candidates for removal)
        removal_priority.sort()
        
        # Try removing codewords in priority order
        new_indices = initial_indices[:]
        new_counts = counts[:]
        removed_any = False
        
        for _, idx in removal_priority:
            if idx not in new_indices:
                continue
            
            # Check if we can remove this codeword
            can_remove = True
            for neighbor in spheres[idx]:
                if new_counts[neighbor] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                new_indices.remove(idx)
                for neighbor in spheres[idx]:
                    new_counts[neighbor] -= 1
                removed_any = True
        
        if removed_any:
            new_codewords = [idx_to_word(idx) for idx in new_indices]
            if len(new_codewords) < best_size:
                best_code = new_codewords
                best_size = len(best_code)
                initial_indices = new_indices
                # Reset lambda weights partially to allow further optimization
                for i in range(total_words):
                    lambda_weights[i] = max(lambda_weights[i] * 0.7, 0.5)
        
        # If we reach the known best, stop
        if best_size <= 73:
            break
        
        # Add some randomization to escape local minima
        if iteration % 10 == 0:
            random.shuffle(initial_indices)
    
    # Final greedy removal pass on the best solution
    final_counts = [0] * total_words
    for idx in [word_to_idx(cw) for cw in best_code]:
        for neighbor in spheres[idx]:
            final_counts[neighbor] += 1
    
    active = [True] * len(best_code)
    for i in range(len(best_code)):
        idx = word_to_idx(best_code[i])
        essential = False
        for neighbor in spheres[idx]:
            if final_counts[neighbor] == 1:
                essential = True
                break
        if not essential:
            active[i] = False
            for neighbor in spheres[idx]:
                final_counts[neighbor] -= 1
    
    final_code = [best_code[i] for i in range(len(best_code)) if active[i]]
    if len(final_code) < len(best_code):
        best_code = final_code
    
    return best_code
