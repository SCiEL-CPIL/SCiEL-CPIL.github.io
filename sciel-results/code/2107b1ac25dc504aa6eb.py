# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct(q, n, radius):
    """
    Construct a covering code using a continuous relaxation (density/entropy approach)
    followed by discrete repair.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        # (Kept for safety, though task is specific)
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = {}
        for w in all_words:
            nb = set()
            for v in all_words:
                dist = sum(1 for i in range(n) if w[i] != v[i])
                if dist <= radius:
                    nb.add(v)
            neighbors[w] = nb
        
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            max_covered = -1
            for w in all_words:
                if w in code:
                    continue
                covered_count = len(neighbors[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_word = w
            if best_word is None:
                break
            code.append(best_word)
            uncovered -= neighbors[best_word]
        return [list(c) for c in code]

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build adjacency matrix A (N x N) where A[i, j] = 1 if word i is within radius of word j
    # A is symmetric for radius 1? Yes, Hamming distance is symmetric.
    # A[i, j] = 1 if d(all_words[i], all_words[j]) <= 1
    N = total_words
    A = np.zeros((N, N), dtype=np.float64)
    
    # Efficient construction of A
    # For each word i, find its neighbors
    for i, w in enumerate(all_words):
        # Self
        A[i, i] = 1.0
        # Neighbors at distance 1
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                A[i, j] = 1.0
    
    # Objective function to minimize:
    # Minimize sum(p) + lambda * sum( max(0, 1 - A@p)^2 )
    # We want p in [0, 1]
    
    # Initial guess: uniform distribution scaled to expected code size ~73
    # p0 = 73 / 729
    p0 = np.full(N, 73.0 / N)
    
    # Lambda needs to be large enough to enforce coverage constraints strongly
    # If we set lambda very high, it becomes a feasibility problem primarily.
    # Let's try a moderate lambda and see.
    # Actually, we can just minimize sum(p) subject to A@p >= 1 (relaxed).
    # Using penalty method:
    # f(p) = sum(p) + mu * sum( (1 - A@p)_+^2 )
    
    def objective(p):
        # p is a vector of length N
        Ap = A.dot(p)
        # Deficit: how much coverage is missing
        deficit = np.maximum(0.0, 1.0 - Ap)
        penalty = np.sum(deficit ** 2)
        cost = np.sum(p) + 100.0 * penalty # 100.0 is lambda
        return cost
    
    def gradient(p):
        Ap = A.dot(p)
        deficit = np.maximum(0.0, 1.0 - Ap)
        # Gradient of penalty term:
        # d/dp_j sum_i (max(0, 1 - sum_k A_ik p_k))^2
        # = sum_i 2 * max(0, 1 - (Ap)_i) * (-A_ij)
        # = -2 * A.T @ deficit
        grad_penalty = -2.0 * A.T.dot(deficit)
        grad_cost = np.ones(N) + 100.0 * grad_penalty
        return grad_cost

    # Use L-BFGS-B with bounds [0, 1]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # We might need multiple restarts or a good initial point.
    # The tetracode structure gives a good initial p.
    # Let's initialize p based on the tetracode construction.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    p_init = np.zeros(N)
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            idx = word_to_idx[w]
            p_init[idx] = 1.0 # Start with 1 for these, 0 for others?
            # Or just set a high probability?
            # Let's set p_init to 1 for tetracode words and 0 for others, then let optimizer lower it?
            # No, if we start with 1s, sum(p) is 81. We want to lower it.
            # But if we start with 0s, coverage is 0.
            # Let's start with the tetracode set having p=1, others p=0.
            # The optimizer will try to reduce sum(p) while maintaining coverage.
            
    # Actually, a better start is the uniform p=73/729.
    # Let's try L-BFGS-B from p0 = np.full(N, 73.0/N)
    
    # To ensure we don't get stuck in a bad local minimum, we can run a few iterations
    # or just rely on the convexity of the penalty part? 
    # The function is not convex because of the max(0, ...) term? 
    # max(0, x)^2 is convex. Sum of convex functions is convex.
    # Wait, A@p is linear. 1 - A@p is linear. max(0, linear)^2 is convex.
    # So the objective is convex!
    # Therefore, any local minimum is a global minimum.
    # L-BFGS-B should find the global minimum of the relaxed problem.
    
    res = minimize(objective, p0, method='L-BFGS-B', jac=gradient, bounds=bounds, 
                   options={'maxiter': 500, 'ftol': 1e-8})
    
    p_opt = res.x
    
    # Convert to discrete set
    # Threshold at 0.5? Or pick the top K?
    # Since it's a relaxation, p_opt values will be between 0 and 1.
    # We want a set of codewords.
    # Strategy: Pick words with p > 0.5.
    # Then repair.
    
    threshold = 0.5
    candidates = [i for i in range(N) if p_opt[i] > threshold]
    
    # If candidates is empty or too few, lower threshold or pick top K
    # Target size is around 73.
    # Let's sort by p_opt and take top 75 to be safe, then repair.
    # Or just take p > 0.5.
    
    if len(candidates) < 50:
        # If relaxation didn't give enough, take top 80
        top_indices = np.argsort(p_opt)[::-1][:80]
        candidates = list(top_indices)
        
    # Repair phase: Ensure all words are covered.
    # Start with set C = candidates
    C = set(candidates)
    
    # Precompute neighbors for repair (reuse A? A is dense, N=729, 729*729*8 bytes ~ 4MB. Fine.)
    # But for repair, we need to check coverage quickly.
    # covered_mask = A.dot(1_C) >= 1
    
    def is_covered(C_set):
        # C_set is a set of indices
        # Create binary vector
        v = np.zeros(N)
        for idx in C_set:
            v[idx] = 1.0
        cov = A.dot(v)
        return np.all(cov >= 1.0 - 1e-9)
    
    # Check if valid
    if not is_covered(C):
        # Add words to cover missing parts
        # Find uncovered words
        v = np.zeros(N)
        for idx in C:
            v[idx] = 1.0
        cov = A.dot(v)
        uncovered_indices = np.where(cov < 1.0 - 1e-9)[0]
        
        # Greedily add words that cover most uncovered
        while uncovered_indices.size > 0:
            # Find word that covers most uncovered
            # For each word w, count how many uncovered indices it covers
            # A[w, :] gives neighbors of w.
            # We want to maximize sum( A[w, u] for u in uncovered_indices )
            # This is A[:, uncovered_indices].sum(axis=1)
            # But A is N x N. Slicing might be slow if done repeatedly?
            # N=729, so it's fast.
            
            # Optimize: only consider words not in C?
            # Actually, adding a word already in C does nothing.
            # So consider w not in C.
            
            best_w = -1
            best_count = -1
            # Iterate all words
            # We can vectorize this:
            # A_sub = A[:, uncovered_indices]
            # counts = A_sub.sum(axis=1)
            # But we must exclude w in C.
            
            counts = A[:, uncovered_indices].sum(axis=1)
            # Mask out words in C
            for idx in C:
                counts[idx] = -1
            best_w = np.argmax(counts)
            
            if counts[best_w] <= 0:
                # No single word covers any uncovered? Should not happen if radius is correct and space is connected
                break
            
            C.add(best_w)
            # Update uncovered
            v[best_w] = 1.0
            cov = A.dot(v)
            uncovered_indices = np.where(cov < 1.0 - 1e-9)[0]
            
    # Now C is a valid covering code.
    # Try to minimize size by removing redundant words.
    improved = True
    while improved:
        improved = False
        for c in list(C):
            C.remove(c)
            if is_covered(C):
                improved = True
                break
            else:
                C.add(c)
                
    # Convert indices back to words
    result = [list(all_words[i]) for i in C]
    return result

# EVOLVE-BLOCK-END
