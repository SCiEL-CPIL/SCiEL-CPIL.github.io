import time
import random
import numpy as np
from itertools import product

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
    A = np.array(A, dtype=np.int64)
    sums = A[:, None] + A[None, :]
    prods = A[:, None] * A[None, :]
    
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    
    return max(n_sums, n_prods)

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_exponent_set(base_primes, max_exponents, max_val=10000):
    """
    Generate a set of numbers defined by exponents of base_primes.
    Returns a list of (exponent_tuple, value) pairs.
    """
    candidates = []
    # Iterate over all combinations of exponents
    ranges = [range(max_exponents + 1) for _ in base_primes]
    for exps in product(*ranges):
        val = 1
        for e, p in zip(exps, base_primes):
            val *= (p ** e)
        if val <= max_val:
            candidates.append((exps, val))
    
    # Remove duplicates (though unlikely with distinct primes)
    unique_candidates = {}
    for exps, val in candidates:
        if val not in unique_candidates:
            unique_candidates[val] = exps
    
    # Convert to list of (val, exps)
    result = [(val, exps) for val, exps in unique_candidates.items()]
    result.sort(key=lambda x: x[0])
    return result

def _decode_exponents(exps, base_primes):
    """Convert exponent tuple to integer value."""
    val = 1
    for e, p in zip(exps, base_primes):
        val *= (p ** e)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 45  # 45s safety margin
    
    rng = random.Random(123)
    
    # Strategy: Search in the space of exponents for a fixed set of small primes.
    # This ensures A*A is very structured (exponents add), minimizing |A*A|.
    # We need to balance this with |A+A|.
    
    # We will try a few different "lattices" (sets of base primes)
    # Lattice 1: {2, 3} -> Numbers of form 2^a * 3^b
    # Lattice 2: {2, 5}
    # Lattice 3: {2, 3, 5}
    
    best_A = None
    best_score = float('inf')
    
    # Pre-compute potential candidates for each lattice
    lattices = [
        [2, 3],
        [2, 5],
        [2, 3, 5],
        [2, 3, 7]
    ]
    
    lattice_data = []
    for primes in lattices:
        # Max exponents: 2^13 is 8192, so 13 is safe for 2.
        # For 3, 3^8 is 6561, so 8 is safe.
        # We'll just generate up to a reasonable bound and filter by max_val.
        max_exp = 14 if primes[0] == 2 else 10
        # Actually, let's just generate all smooth numbers <= 10000 for these primes
        # and keep their exponent vectors.
        
        # Simple generation
        candidates = {}
        # Recursive generation or iterative
        # Let's just iterate exponents
        ranges = []
        for p in primes:
            e = 0
            val = 1
            while val <= 10000:
                e += 1
                val *= p
            ranges.append(range(e + 1))
            
        for exps in product(*ranges):
            val = 1
            for e, p in zip(exps, primes):
                val *= (p ** e)
            if val <= 10000:
                if val not in candidates:
                    candidates[val] = exps
        
        # Create list of (val, exps)
        lattice_entries = [(val, exps) for val, exps in candidates.items()]
        lattice_entries.sort(key=lambda x: x[0])
        lattice_data.append((primes, lattice_entries))
        
    # Initial best: try to pick 20 numbers from the most "dense" lattice
    # Usually {2,3} or {2,3,5} gives the smallest product set.
    
    # We will perform a local search on the exponent vectors.
    # State: A list of 20 exponent vectors (or just indices into the lattice).
    
    # To speed up, we will map value -> index for fast lookup, but we need distinct values.
    # Let's represent the set A by a list of indices into the lattice_entries list.
    
    for primes, entries in lattice_data:
        if time.time() > deadline:
            break
            
        # Map value to index
        val_to_idx = {val: idx for idx, (val, exps) in enumerate(entries)}
        
        # Initial selection: Pick 20 random distinct values from this lattice
        # To start, pick the smallest 20? Or random?
        # Smallest values often lead to smaller sums, but might have many collisions in products?
        # Actually, for products, exponents add. If we pick 2^a * 3^b, the product set size
        # is related to the convex hull of the exponent vectors.
        
        # Let's try a few random starts on this lattice
        n_entries = len(entries)
        if n_entries < 20:
            continue
            
        for start_iter in range(20):
            if time.time() > deadline:
                break
                
            # Randomly select 20 distinct indices
            indices = rng.sample(range(n_entries), 20)
            
            # Sort indices to have a canonical order (though order doesn't matter for set)
            indices.sort()
            
            # Decode to values
            A_vals = [entries[i][0] for i in indices]
            
            if not _is_valid(A_vals):
                continue
                
            current_score = _score(A_vals)
            
            if current_score < best_score:
                best_score = current_score
                best_A = A_vals[:]
            
            # Local Search on this lattice
            # We will mutate the indices.
            # Neighborhood: 
            # 1. Swap two indices
            # 2. Change one index to a nearby index in the lattice (sorted by value)
            
            T = 50.0
            cooling = 0.99
            min_T = 0.5
            no_improve_count = 0
            max_no_improve = 50
            
            while time.time() < deadline:
                if T < min_T or no_improve_count > max_no_improve:
                    break
                    
                # Generate neighbor
                new_indices = indices[:]
                
                move_type = rng.random()
                if move_type < 0.4:
                    # Swap two
                    i, j = rng.sample(range(20), 2)
                    new_indices[i], new_indices[j] = new_indices[j], new_indices[i]
                else:
                    # Perturb one
                    i = rng.randint(0, 19)
                    old_idx = new_indices[i]
                    # Try to move to a nearby index in the lattice
                    # "Nearby" in value space
                    # Let's pick a random offset in index space
                    offset = rng.randint(-5, 5)
                    new_idx = old_idx + offset
                    if 0 <= new_idx < n_entries:
                        # Check if new_idx is already in new_indices
                        if new_idx not in new_indices:
                            new_indices[i] = new_idx
                        else:
                            continue
                    else:
                        continue
                
                # Check validity (distinctness is handled by index check, 
                # but we should ensure values are distinct? 
                # Indices are distinct, and val_to_idx is unique, so values are distinct.)
                
                # Decode new values
                new_A_vals = [entries[i][0] for i in new_indices]
                
                # Quick check: max value
                if max(new_A_vals) > 10000:
                    continue
                
                new_score = _score(new_A_vals)
                
                if new_score < current_score:
                    indices = new_indices
                    current_score = new_score
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_A = new_A_vals[:]
                else:
                    delta = new_score - current_score
                    if delta == 0:
                        prob = 1.0
                    else:
                        prob = np.exp(-delta / T)
                    
                    if rng.random() < prob:
                        indices = new_indices
                        current_score = new_score
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                
                T *= cooling
    
    if best_A is None:
        # Fallback
        return [2**i for i in range(20)]
        
    return sorted(best_A)

if __name__ == "__main__":
    # Test
    A = construct()
    print(f"Result: {A}")
    print(f"Score: {_score(A)}")
