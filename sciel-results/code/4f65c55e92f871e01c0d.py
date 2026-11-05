import time

_START = time.time()
_DEADLINE = 80.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # best_size[r] = minimum subset size that sums to r mod MOD
    # best_terms[r] = the list of denominators achieving that minimum size
    best_size = [N + 1] * MOD
    best_terms = [None] * MOD
    
    # Iterate over all non-empty subsets
    for mask in range(1, 1 << N):
        # Calculate sum of inverses and size for this mask
        s = 0
        size = 0
        terms = []
        for x in range(1, N + 1):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % MOD
                size += 1
                terms.append(x)
        
        if size < best_size[s]:
            best_size[s] = size
            best_terms[s] = terms
    
    # Construct the result
    # If a residue is not covered (best_terms[r] is None), it's an error in problem assumption.
    # But we handle it by returning an empty list or raising an error? 
    # The contract says nonempty. If None, we can't satisfy. 
    # Assuming coverage is guaranteed.
    result = []
    for r in range(MOD):
        if best_terms[r] is None:
            # Fallback: should not happen. Return empty or raise.
            # To be safe against "nonempty" check, if this happens, we are invalid.
            result.append([])
        else:
            result.append(best_terms[r])
            
    return result
