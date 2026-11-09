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
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def construct():
    primes = _primes(N)
    
    # Beam search parameters
    beam_width = 10
    
    # Initialize beam with the full set
    # State is a boolean mask of length N+1
    initial_mask = np.ones(N + 1, dtype=bool)
    beam = [initial_mask]
    
    for p in primes:
        new_beam_candidates = []
        
        for mask in beam:
            current_size = np.sum(mask)
            
            # If p > current_size, this state is "safe" for this prime and all larger primes
            # We can just keep it, but we should ensure we don't duplicate identical masks
            # However, since we process primes in increasing order, if p > size, 
            # no future prime q > p will have q <= size either.
            # So this state is final.
            if p > current_size:
                new_beam_candidates.append(mask)
                continue
            
            # Get active indices
            active_indices = np.where(mask)[0]
            
            # Count residues
            residues = active_indices % p
            counts = np.bincount(residues, minlength=p)
            
            # Check if any residue class is already empty
            if counts.min() == 0:
                # Already admissible for this prime, keep state
                new_beam_candidates.append(mask)
                continue
            
            # Find residue classes with the minimum count
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            
            # For each candidate residue, generate a new state
            for target_residue in candidates:
                # Identify elements to remove
                indices_to_remove = active_indices[residues == target_residue]
                
                # Create new mask
                new_mask = mask.copy()
                new_mask[indices_to_remove] = False
                
                new_beam_candidates.append(new_mask)
        
        # Deduplicate and select top beam_width states by size
        # Since N=5000, we can just use a set of hashes or just sort and unique
        # Using a dictionary to keep the best mask for each unique content?
        # Actually, just sort by size and take top K unique masks.
        
        # To deduplicate efficiently, we can convert masks to bytes or hashes.
        # But beam_width is small (10), and candidates per state is small (num min residues).
        # Total candidates in new_beam_candidates is at most beam_width * (num min residues).
        # Num min residues is usually small (1-5).
        # So list size is small.
        
        # Sort by size descending
        new_beam_candidates.sort(key=lambda m: np.sum(m), reverse=True)
        
        # Select top beam_width unique masks
        selected_masks = []
        seen_hashes = set()
        for m in new_beam_candidates:
            # Hash the mask
            h = hash(m.tobytes())
            if h not in seen_hashes:
                seen_hashes.add(h)
                selected_masks.append(m)
                if len(selected_masks) == beam_width:
                    break
        
        beam = selected_masks
        
        # Early termination if beam is empty (should not happen)
        if not beam:
            break
            
    # Return the best mask from the beam
    if not beam:
        return list(range(N + 1))
        
    best_mask = max(beam, key=lambda m: np.sum(m))
    return list(np.where(best_mask)[0])
# EVOLVE-BLOCK-END
