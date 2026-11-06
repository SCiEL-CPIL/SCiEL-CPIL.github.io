"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

def construct():
    # Initialize mask: True means the number is in the tuple
    mask = np.ones(N + 1, dtype=bool)
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Current size of the set
    current_size = N + 1
    
    # We iterate through primes in increasing order.
    # We only need to consider primes p such that p <= current_size.
    # As we remove elements, current_size decreases.
    # If p > current_size, we don't need to check it.
    # Since primes are sorted, we can break early if p > current_size.
    
    for p in primes:
        if p > current_size:
            break
        
        # Check if all residue classes mod p are covered by the current set
        # We need to count how many active numbers fall into each residue class
        # mask is boolean, so indices where mask is True are active.
        active_indices = np.nonzero(mask)[0]
        
        if len(active_indices) == 0:
            break
            
        # Compute residues of active indices mod p
        residues = active_indices % p
        # Count occurrences of each residue
        # bincount is fast for this
        counts = np.bincount(residues, minlength=p)
        
        # If all counts are > 0, all residue classes are covered.
        # We must remove one residue class to break this.
        # To maximize the remaining set size, we remove the residue class with the fewest elements.
        if np.all(counts > 0):
            min_count = np.min(counts)
            min_class = np.argmin(counts)
            
            # Remove all active numbers that are congruent to min_class mod p
            # We update the mask directly
            # indices_to_remove = active_indices[residues == min_class]
            # mask[indices_to_remove] = False
            
            # More efficient:
            # Create a boolean mask for removal
            # We need to identify which global indices are in min_class mod p AND currently active.
            # active_indices[residues == min_class] gives the global indices.
            removal_mask = np.zeros(N + 1, dtype=bool)
            removal_mask[active_indices[residues == min_class]] = True
            mask[removal_mask] = False
            
            # Update current size
            current_size -= min_count
            
    # Return the list of active indices
    return np.nonzero(mask)[0].tolist()
# EVOLVE-BLOCK-END
