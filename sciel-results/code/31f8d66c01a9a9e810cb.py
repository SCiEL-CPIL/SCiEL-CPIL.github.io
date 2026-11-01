"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize
import time

def construct(q, n, radius):
    """
    Construct a covering code using a differentiable neural network approach (soft relaxation)
    followed by discrete local search refinement.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _generic_greedy(q, n, radius)

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor indices for each word
    # neighbors[i] is a list of indices j such that d(all_words[i], all_words[j]) <= radius
    neighbors = []
    for i, w in enumerate(all_words):
        nb = []
        for j, v in enumerate(all_words):
            dist = sum(1 for k in range(n) if w[k] != v[k])
            if dist <= radius:
                nb.append(j)
        neighbors.append(nb)
    
    # Convert to numpy arrays for speed
    # We need to compute the product of (1-p) for neighbors efficiently.
    # This is hard to vectorize directly for the product term in a standard gradient step
    # unless we use log-space or similar, but L-BFGS-B works on the function value and gradients.
    # We will implement the function and its gradient manually or use numerical gradients if analytic is too complex.
    # Given 729 vars, numerical gradients (finite differences) would be 729 function evals per gradient step.
    # L-BFGS-B does many steps. This might be slow.
    # Let's try to derive the gradient.
    
    # Let p_i = sigma(z_i).
    # Loss = sum(p_i) + lambda * sum_j P_j, where P_j = prod_{k in N(j)} (1 - p_k).
    # dL/dz_i = p_i(1-p_i) + lambda * sum_{j: i in N(j)} dP_j/dz_i
    # dP_j/dz_i = P_j * d/dz_i [ sum_{k in N(j)} log(1-p_k) ]
    # d/dz_i log(1-p_k) = 0 if k != i.
    # If i in N(j): d/dz_i log(1-p_i) = 1/(1-p_i) * (-dp_i/dz_i) = 1/(1-p_i) * (-p_i(1-p_i)) = -p_i.
    # So dP_j/dz_i = P_j * (-p_i) if i in N(j), else 0.
    # dL/dz_i = p_i(1-p_i) - lambda * p_i * sum_{j: i in N(j)} P_j
    
    # This is computationally feasible if we precompute the reverse adjacency list.
    # reverse_neighbors[i] = list of j such that i in N(j). Note N(j) includes j itself?
    # Yes, distance <= 1 includes the word itself.
    
    rev_neighbors = [[] for _ in range(total_words)]
    for j in range(total_words):
        for i in neighbors[j]:
            rev_neighbors[i].append(j)

    lambda_penalty = 10.0 # Tune this
    
    def loss_and_grad(z):
        p = 1.0 / (1.0 + np.exp(-z))
        
        # Compute P_j = prod_{k in N(j)} (1 - p_k)
        # We can compute log_P_j = sum_{k in N(j)} log(1 - p_k) to avoid underflow, 
        # but for gradient we need P_j. 
        # Given 729 words and small degree (~19), direct product is fine.
        P = np.ones(total_words)
        for j in range(total_words):
            prod = 1.0
            for k in neighbors[j]:
                prod *= (1.0 - p[k])
            P[j] = prod
            
        # Sum of penalties
        penalty_sum = np.sum(P)
        
        # Sum of p
        p_sum = np.sum(p)
        
        loss = p_sum + lambda_penalty * penalty_sum
        
        # Gradient
        # dL/dz_i = p_i(1-p_i) - lambda * p_i * sum_{j in rev_neighbors[i]} P_j
        grad = np.zeros(total_words)
        for i in range(total_words):
            # Sum P_j for all j that cover i
            sum_P = 0.0
            for j in rev_neighbors[i]:
                sum_P += P[j]
            grad[i] = p[i] * (1.0 - p[i]) - lambda_penalty * p[i] * sum_P
            
        return loss, grad

    best_code = None
    best_size = total_words + 1
    
    # Run optimization with a few random restarts
    for attempt in range(5):
        # Random initialization
        z0 = np.random.normal(0, 1, total_words)
        
        res = minimize(loss_and_grad, z0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 500, 'ftol': 1e-8})
        
        z_opt = res.x
        p_opt = 1.0 / (1.0 + np.exp(-z_opt))
        
        # Threshold to get discrete code
        # We want to include words with high probability.
        # A threshold of 0.5 is standard.
        mask = p_opt > 0.5
        code_set = set(np.where(mask)[0].tolist())
        
        if len(code_set) == 0:
            continue
            
        # Post-processing: Ensure valid and minimize size
        # 1. Check validity
        covered = set()
        for idx in code_set:
            covered.update(neighbors[idx])
            
        if len(covered) != total_words:
            # If not covered, we need to add words.
            # Greedily add words covering most uncovered
            uncovered = set(range(total_words)) - covered
            while uncovered:
                best_idx = -1
                max_cov = -1
                for idx in range(total_words):
                    if idx in code_set:
                        continue
                    cov = len(set(neighbors[idx]) & uncovered)
                    if cov > max_cov:
                        max_cov = cov
                        best_idx = idx
                if best_idx == -1:
                    break
                code_set.add(best_idx)
                covered.update(neighbors[best_idx])
                uncovered -= set(neighbors[best_idx])
                
            if len(covered) != total_words:
                # Should not happen, but fallback
                continue

        # 2. Local search to reduce size
        improved = True
        while improved:
            improved = False
            # Try removing
            for idx in list(code_set):
                new_set = code_set - {idx}
                # Check coverage
                cov = set()
                for c in new_set:
                    cov.update(neighbors[c])
                if len(cov) == total_words:
                    code_set = new_set
                    improved = True
                    break
        
        # Try 1-swap
        improved = True
        while improved:
            improved = False
            code_list = list(code_set)
            for i, c_remove in enumerate(code_list):
                # Find best candidate to add
                # We want to add a word that allows removal of c_remove or another word
                # Simple heuristic: try adding a word that is not in code, check if we can remove c_remove
                # Or just try all swaps? 729 * 729 is ~500k checks. Each check is O(729*19) ~ 14k ops. Total ~7 billion. Too slow.
                
                # Optimization: 
                # Calculate coverage count for each word.
                # If we remove c, we lose words covered ONLY by c.
                # Let unique_c = words covered by c but not by any other in code_set - {c}.
                # If unique_c is empty, we can remove c. (Handled above)
                # If we swap c with d, we lose unique_c and gain unique_d (relative to new set).
                # This is complex. Let's stick to the simple remove-first loop which is effective.
                # The neural net should provide a good starting point.
                pass
        
        size = len(code_set)
        if size < best_size:
            best_size = size
            best_code = code_set.copy()
            
        if best_size <= 73:
            break # Can't do better than known best 73
            
    # Convert to list of lists
    if best_code is None:
        # Fallback to greedy if optimization failed
        return _generic_greedy(q, n, radius)
        
    result = [list(all_words[idx]) for idx in best_code]
    return result

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nb = []
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.append(word_to_idx[v])
        neighbors.append(nb)
        
    uncovered = set(range(len(all_words)))
    code = []
    
    while uncovered:
        best_word = -1
        max_covered = -1
        for i in range(len(all_words)):
            if i in code:
                continue
            covered_count = len(set(neighbors[i]) & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = i
        
        if best_word == -1:
            break
            
        code.append(best_word)
        uncovered -= set(neighbors[best_word])
        
    return [list(all_words[c]) for c in code]

# EVOLVE-BLOCK-END
