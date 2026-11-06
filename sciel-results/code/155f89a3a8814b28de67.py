import sys
from functools import lru_cache

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a strictly increasing sequence of integers in [2, 128] 
    maximizing length such that all contiguous block products are distinct.
    Uses Beam Search with prime exponent vectors.
    """
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    factor_vecs = {}
    for n in range(2, 129):
        n_tmp = n
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n_tmp % p == 0:
                vec[i] += 1
                n_tmp //= p
            if n_tmp == 1:
                break
        factor_vecs[n] = tuple(vec)
    
    zero_vec = tuple([0] * num_primes)
    
    # State: (values, prefix_vecs, product_set)
    # values: list of ints
    # prefix_vecs: list of tuples, prefix_vecs[0] = 0, prefix_vecs[i] = sum(factor_vecs[values[0:i]])
    # product_set: set of tuples, all contiguous block products
    
    def add_value(state, x):
        """
        Try to add x to the state.
        Returns new state or None if invalid.
        """
        values, prefix_vecs, product_set = state
        k = len(values)
        
        # New prefix
        last_prefix = prefix_vecs[-1]
        new_prefix = tuple(last_prefix[i] + factor_vecs[x][i] for i in range(num_primes))
        
        # Generate new block products ending at x
        new_products = []
        # prefix_vecs has length k+1 (indices 0..k)
        # New blocks are prefix_vecs_new[-1] - prefix_vecs[i] for i in 0..k
        # prefix_vecs_new[-1] is new_prefix
        for i in range(k + 1):
            old_prefix = prefix_vecs[i]
            block_vec = tuple(new_prefix[j] - old_prefix[j] for j in range(num_primes))
            if block_vec in product_set:
                return None
            new_products.append(block_vec)
        
        # If no collision, update state
        new_product_set = product_set.copy()
        new_product_set.update(new_products)
        
        new_values = values + [x]
        new_prefix_vecs = prefix_vecs + [new_prefix]
        
        return (new_values, new_prefix_vecs, new_product_set)
    
    # Beam Search
    beam_width = 50
    # Initial state: empty sequence
    # values: []
    # prefix_vecs: [zero_vec]
    # product_set: set()
    current_beam = [([], [zero_vec], set())]
    
    for x in range(2, 129):
        next_candidates = []
        
        for state in current_beam:
            # Option 1: Skip x
            next_candidates.append(state)
            
            # Option 2: Add x
            new_state = add_value(state, x)
            if new_state is not None:
                next_candidates.append(new_state)
        
        # Prune: keep top beam_width states
        # Sort by length descending, then by values (lexicographically) to be deterministic
        # Since we want max length, primary key is -len(values)
        # Secondary key: values tuple (smaller values preferred? or just arbitrary?)
        # Let's use values tuple as tie-breaker.
        
        # Optimization: If we have many candidates, sorting all is O(N log N).
        # N = 2 * beam_width = 100.
        # Sorting 100 items is fast.
        
        next_candidates.sort(key=lambda s: (-len(s[0]), s[0]))
        
        # Keep top beam_width
        current_beam = next_candidates[:beam_width]
        
        # Early termination if no states have length > 0? No, we always have at least the skip chain.
        # If all states are empty, we can break? No, we might add later.
    
    # Find best state
    best_state = max(current_beam, key=lambda s: len(s[0]))
    return best_state[0]

# EVOLVE-BLOCK-END
