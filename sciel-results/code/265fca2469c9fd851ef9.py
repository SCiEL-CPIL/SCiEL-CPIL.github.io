import numpy as np

def construct():
    p = 1009
    
    # Create array of x values 1 to p-1
    x = np.arange(1, p)
    
    # Compute quadratic residues
    # (x * x) % p
    residues = (x * x) % p
    
    # Create a boolean mask for which values 1..p-1 are quadratic residues
    # We need to check if a value v is in residues.
    # Since p is small (1009), we can use a set or a boolean array.
    # Using a boolean array for O(1) lookup.
    is_qr = np.zeros(p, dtype=bool)
    is_qr[residues] = True
    
    # We need to find the smallest s in 1..1004 such that
    # s, s+1, s+2, s+3, s+4 are all True in is_qr.
    # And they must be non-zero (indices 1..1008).
    # Since s >= 1 and s <= 1004, the indices s..s+4 are in 1..1008.
    # So we just need is_qr[s] to is_qr[s+4] to be True.
    
    # Vectorized check:
    # For each candidate s, check is_qr[s], is_qr[s+1], ..., is_qr[s+4]
    # We can construct a boolean array `valid_s` where valid_s[s] is True if the run starting at s is valid.
    
    # Let's create an array of candidates s from 1 to p-5 (which is 1004)
    candidates = np.arange(1, p - 4)
    
    # For each candidate, we need to check 5 consecutive values.
    # We can use np.all on slices, but doing it in a loop is fine for 1000 iterations.
    # Or we can use convolution/sliding window logic.
    # Simple loop is structurally different from the previous pure-python dict/set approach.
    
    for s in candidates:
        # Check s to s+4
        # is_qr is 0-indexed, so is_qr[s] corresponds to value s.
        # We need is_qr[s], is_qr[s+1], is_qr[s+2], is_qr[s+3], is_qr[s+4] to be True.
        if np.all(is_qr[s:s+5]):
            # Found the minimal s.
            # Now we need the roots.
            # We need to find x such that x*x % p == s+i for i=0..4.
            # We have the array x and residues.
            # We can find the index in residues for each value.
            # Since residues might have duplicates, we just need one root.
            # We can use np.where(residues == val)[0][0] to get the first x index.
            # Then x[index] is the root.
            
            roots = []
            for i in range(5):
                val = s + i
                # Find x such that x*x % p == val
                # residues == val gives boolean mask
                idxs = np.where(residues == val)[0]
                if len(idxs) > 0:
                    # x is the array of roots corresponding to residues
                    # Wait, x is the array of inputs.
                    # The root is x[idxs[0]].
                    roots.append(int(x[idxs[0]]))
                else:
                    # This should not happen if is_qr was True
                    roots.append(0)
                    
            return {'start': int(s), 'roots': roots}
            
    raise RuntimeError('No run found')
