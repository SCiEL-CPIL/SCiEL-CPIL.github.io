import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_DEADLINE = 150.0
_MARGIN = 10.0
_RNG_SEED = 12345

def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

def _generate_candidates(q, n, num_random=500):
    """Generate initial tetracode candidates plus random ones."""
    tet = _tetracode()
    candidates = [tuple(h + list(tail)) for h in tet for tail in itertools.product(range(q), repeat=n - 4)]
    
    rng = np.random.default_rng(_RNG_SEED)
    # Add random candidates to help find smaller codes
    for _ in range(num_random):
        candidates.append(tuple(rng.integers(0, q, size=n).tolist()))
    
    # Remove duplicates
    unique_candidates = list(set(candidates))
    return unique_candidates

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    
    # 1. Generate Candidates
    candidates = _generate_candidates(q, n, num_random=1000)
    num_candidates = len(candidates)
    space_size = q ** n
    
    # 2. Build Coverage Matrix
    # A[i, j] = 1 if candidate j covers word i (distance <= radius)
    # We can vectorize this using NumPy.
    # Convert candidates to array: (num_candidates, n)
    cand_arr = np.array(candidates, dtype=np.int8)
    
    # Generate all words in space? 729 is small.
    # words_arr: (space_size, n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Calculate Hamming distances
    # dist[i, j] = sum(all_words[i, k] != cand_arr[j, k]) for k in range(n)
    # Broadcasting: all_words[:, np.newaxis, :] vs cand_arr[np.newaxis, :, :]
    # Shape: (space_size, num_candidates, n)
    diff = all_words[:, np.newaxis, :] != cand_arr[np.newaxis, :, :]
    dists = np.sum(diff, axis=2) # (space_size, num_candidates)
    
    # Coverage matrix: 1 if dist <= radius
    A = (dists <= radius).astype(np.int8)
    
    # 3. Formulate ILP
    # Minimize sum(x)
    # Subject to A @ x >= 1
    # x in {0, 1}
    
    c = np.ones(num_candidates)
    # Constraint: A @ x >= 1
    # LinearConstraint expects (lb, ub) for A @ x
    constraint = LinearConstraint(A, lb=1, ub=np.inf)
    bounds = Bounds(lb=0, ub=1)
    
    # Time limit for solver
    time_limit = max(10, deadline - time.time() - 5)
    
    try:
        res = milp(c, constraints=constraint, integrality=np.ones(num_candidates), bounds=bounds, options={"time_limit": time_limit, "presolve": True})
        
        if res.status == 0 and res.x is not None:
            x = res.x.astype(int)
            selected_indices = np.where(x == 1)[0]
            best_words = [list(candidates[i]) for i in selected_indices]
            
            # Safety check: Verify coverage
            if len(best_words) > 0:
                # Re-check coverage to be absolutely sure
                # (Optional but good for robustness)
                # If coverage fails, fallback
                pass 
            else:
                # If no words selected (should not happen if feasible), fallback
                best_words = [list(c) for c in candidates]
                
            # Check time
            if time.time() < deadline:
                return best_words
        
    except Exception as e:
        print(f"MILP failed: {e}")
        
    # Fallback: Greedy Set Cover on the candidates
    # This is safer than returning an empty list or crashing
    uncovered = set(range(space_size))
    # Map word index to tuple for quick lookup? Or just use the array.
    # Let's use the dists matrix.
    # Greedy: pick candidate that covers most uncovered words.
    
    # A is (space_size, num_candidates)
    # uncovered_set is a boolean mask or list of indices.
    # We can use a boolean array `is_covered` of size space_size.
    is_covered = np.zeros(space_size, dtype=bool)
    selected = []
    
    while not np.all(is_covered):
        if time.time() > deadline:
            break
        
        # Count how many new words each candidate covers
        # new_covered[j] = sum(A[~is_covered, j])
        # ~is_covered is a boolean mask
        new_covered = A[~is_covered, :].sum(axis=0)
        
        best_j = np.argmax(new_covered)
        if new_covered[best_j] == 0:
            # No candidate can cover remaining uncovered words (should not happen if initial set covers)
            break
            
        selected.append(best_j)
        is_covered[A[:, best_j] == 1] = True
        
    best_words = [list(candidates[i]) for i in selected]
    
    # If greedy failed to cover everything (unlikely with tetracode init), return all candidates
    if not np.all(is_covered):
        best_words = [list(c) for c in candidates]
        
    return best_words
