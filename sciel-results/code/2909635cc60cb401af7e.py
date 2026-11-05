import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    m = len(all_triples)
    
    # Generate all pairs
    all_pairs = list(itertools.combinations(range(n), 2))
    p = len(all_pairs)
    
    # Map pairs to indices
    pair_index = {pair: i for i, pair in enumerate(all_pairs)}
    
    # Build the constraint matrix A for pair conflicts
    # A is p x m, where A[i, j] = 1 if triple j contains pair i
    # We need A @ x <= 1 for all i (each pair covered at most once)
    A = np.zeros((p, m), dtype=np.int8)
    for j, tri in enumerate(all_triples):
        # For each triple, mark its 3 constituent pairs
        A[pair_index[(tri[0], tri[1])], j] = 1
        A[pair_index[(tri[0], tri[2])], j] = 1
        A[pair_index[(tri[1], tri[2])], j] = 1
        
    # Objective: maximize sum(x_j) -> minimize -sum(x_j)
    c = np.full(m, -1.0)
    
    # Constraints: A @ x <= 1
    # LinearConstraint(A_ub, ub, A_lb, lb)
    # A @ x <= 1  =>  A @ x <= 1 and -inf <= A @ x
    # scipy milp uses: A_lb <= A @ x <= A_ub
    constraints = LinearConstraint(A, lb=-np.inf, ub=1.0)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers
    integrality = np.ones(m)
    
    best_triples = []
    best_alpha = n
    
    try:
        # Solve the MILP
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            x = res.x
            selected_indices = np.where(x > 0.5)[0]
            best_triples = [list(all_triples[i]) for i in selected_indices]
            
            # Calculate independence number for this solution
            alpha = _calc_alpha(best_triples, n)
            best_alpha = alpha
            
            # Local Search Refinement
            # Try to swap triples or add/remove to reduce alpha if possible
            # Since we maximized the number of triples, we are in a "dense" region.
            # However, different maximum packings might have different alpha.
            # We can perform a simple local search on the set of selected triples.
            
            current_triples = [t[:] for t in best_triples]
            current_alpha = alpha
            
            # Time check
            if time.time() < deadline - 20.0:
                current_triples = _local_search(current_triples, current_alpha, n, deadline)
                current_alpha = _calc_alpha(current_triples, n)
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples
                    
        else:
            # Fallback to greedy if MILP fails
            best_triples = _greedy_construct(n, deadline)
            best_alpha = _calc_alpha(best_triples, n)
            
    except Exception:
        # Fallback to greedy if MILP crashes
        best_triples = _greedy_construct(n, deadline)
        best_alpha = _calc_alpha(best_triples, n)

    return best_triples

def _calc_alpha(triples, n):
    """Calculate the independence number of the hypergraph."""
    if not triples:
        return n
    
    # Represent triples as bitmasks for fast intersection checks
    triple_masks = []
    for t in triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
    
    # To find the largest set S containing no triple:
    # We can iterate over subsets? 2^16 = 65536 is small.
    # Let's check all subsets of size k from n down to 1.
    # Actually, checking all subsets of size k is C(n, k).
    # Max C(16, 8) is 12870. Sum is 65536.
    # We can just iterate all masks from 1 to 2^n - 1.
    
    # Optimization: If a set S is independent, any subset of S is independent.
    # If a set S contains a triple, it is not independent.
    # We want the max |S| such that S does not contain any triple.
    
    # Let's use a simple loop over all possible subsets.
    max_size = 0
    # We can prune: if we find a set of size k, we don't need to check smaller ones for the max,
    # but we need to be sure no larger one exists.
    # Iterating by size descending is better for early exit if we find a large one?
    # No, we need to check if a set of size k is independent.
    # If we iterate all 2^n subsets, we can just check each.
    
    # Faster check: 
    # A set S is independent iff for all t in triples, S is not a superset of t.
    # i.e., S & mask_t != mask_t for all t.
    
    # Let's iterate all subsets.
    # n=16, 2^16 = 65536. This is fast.
    for mask in range(1, 1 << n):
        # Calculate popcount
        size = bin(mask).count('1')
        if size <= max_size:
            continue
            
        # Check if independent
        is_independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_independent = False
                break
        
        if is_independent:
            max_size = size
            
    return max_size

def _greedy_construct(n, deadline):
    """Greedy construction of a linear triple system."""
    triples = []
    used_pairs = set()
    
    # Sort triples randomly or by some heuristic?
    # Random restarts might help, but we have limited time.
    # Let's try a few random orderings.
    import random
    
    best_triples = []
    best_alpha = n
    
    while time.time() < deadline - 10.0:
        candidates = list(itertools.combinations(range(n), 3))
        random.shuffle(candidates)
        
        current_triples = []
        current_used_pairs = set()
        
        for tri in candidates:
            pairs = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
            valid = True
            for p in pairs:
                if p in current_used_pairs:
                    valid = False
                    break
            if valid:
                current_triples.append(list(tri))
                for p in pairs:
                    current_used_pairs.add(p)
                    
        alpha = _calc_alpha(current_triples, n)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples
            
        # If we hit a good score, maybe stop?
        # Theoretical min alpha for 16 points with linear triples?
        # If we can pack 20 triples (16*15/6 = 40 pairs, 20*3=60 pairs, so max 13 triples? No.
        # Max number of triples in a 2-(16,3,1) packing is bounded by floor(16/3 * floor(15/2) / 3) = floor(5 * 7 / 3) = 11?
        # Actually, max packing number D(16,3,1) is known to be 11 or 12?
        # Let's just keep improving.
        
    return best_triples

def _local_search(triples, alpha, n, deadline):
    """
    Local search to improve alpha.
    Move: Remove a triple, add a different triple if possible.
    Or: Swap two triples.
    """
    import random
    
    current_triples = [t[:] for t in triples]
    current_alpha = alpha
    used_pairs = set()
    for t in current_triples:
        for i in range(3):
            for j in range(i+1, 3):
                used_pairs.add(tuple(sorted((t[i], t[j]))))
                
    # Precompute all triples and their pair sets
    all_triples = list(itertools.combinations(range(n), 3))
    triple_pairs = []
    for t in all_triples:
        ps = [tuple(sorted((t[0], t[1]))), tuple(sorted((t[0], t[2]))), tuple(sorted((t[1], t[2])))]
        triple_pairs.append(ps)
        
    # Map triple to index
    tri_index = {t: i for i, t in enumerate(all_triples)}
    
    # Indices of currently selected triples
    selected_indices = [tri_index[tuple(t)] for t in current_triples]
    
    no_improve_count = 0
    
    while time.time() < deadline - 5.0 and no_improve_count < 100:
        # Pick a random selected triple to remove
        if not selected_indices:
            break
        idx_remove = random.choice(selected_indices)
        t_remove = all_triples[idx_remove]
        
        # Remove it
        current_triples.remove(list(t_remove))
        selected_indices.remove(idx_remove)
        for p in triple_pairs[idx_remove]:
            used_pairs.remove(p)
            
        # Try to add a different triple
        # Candidates: triples that don't conflict with current used_pairs
        # This might be expensive to check all.
        # Let's just try random candidates.
        
        improved = False
        for _ in range(50):
            idx_add = random.randint(0, len(all_triples) - 1)
            if idx_add == idx_remove:
                continue
            t_add = all_triples[idx_add]
            ps = triple_pairs[idx_add]
            
            valid = True
            for p in ps:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                # Add it
                current_triples.append(list(t_add))
                selected_indices.append(idx_add)
                for p in ps:
                    used_pairs.add(p)
                
                new_alpha = _calc_alpha(current_triples, n)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    no_improve_count = 0
                    improved = True
                    break
                else:
                    # Revert
                    current_triples.remove(list(t_add))
                    selected_indices.remove(idx_add)
                    for p in ps:
                        used_pairs.remove(p)
                        
        if not improved:
            no_improve_count += 1
            # Restore the removed triple? Or keep it removed?
            # If we can't add a better one, maybe adding back is best?
            # Or maybe we should try adding *any* valid triple to maintain size?
            # The objective is alpha, not size. But smaller alpha usually correlates with larger size/density.
            # Let's try to add back the original if no improvement.
            ps_rem = triple_pairs[idx_remove]
            valid = True
            for p in ps_rem:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                current_triples.append(list(t_remove))
                selected_indices.append(idx_remove)
                for p in ps_rem:
                    used_pairs.add(p)
            else:
                # Conflict? Shouldn't happen if we only removed it.
                pass

    return current_triples
# EVOLVE-BLOCK-END
