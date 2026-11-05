"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0: sieve[0] = False
    if limit >= 1: sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Start with all integers in [0, N]
    S = list(range(N + 1))
    
    # Precompute primes up to N
    primes = _primes(N)
    
    # Iteratively remove elements to satisfy admissibility
    # We look for the smallest prime p <= len(S) that is fully covered.
    # If found, remove the least-populated residue class mod p.
    
    while True:
        L = len(S)
        if L == 0:
            return S
            
        # Find the smallest prime p <= L that is "unsafe" (fully covered)
        target_p = None
        min_class_size = float('inf')
        target_residue = None
        
        # We only need to check primes up to L
        # Since primes is sorted, we can iterate until p > L
        for p in primes:
            if p > L:
                break
            
            # Check if S covers all residues mod p
            residues = set()
            # Optimization: if we find all p residues, we can break early?
            # But we also need the min class size.
            # Let's count classes.
            counts = [0] * p
            
            for x in S:
                counts[x % p] += 1
            
            # Check if any count is 0
            if min(counts) > 0:
                # This prime is unsafe
                target_p = p
                min_class_size = min(counts)
                target_residue = counts.index(min_class_size)
                break # We pick the smallest unsafe prime
        
        if target_p is None:
            # No unsafe primes found, S is admissible
            return S
        
        # Remove all x in S such that x % target_p == target_residue
        S = [x for x in S if x % target_p != target_residue]
        
        # Safety break if no progress (should not happen)
        if len(S) == L:
            break

    return S
# EVOLVE-BLOCK-END
