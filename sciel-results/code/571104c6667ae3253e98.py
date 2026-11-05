import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin
    
    # Precompute non-zero quadratic non-residues
    # A number r in 1..p-1 is a non-residue if there is no x in 1..p-1 such that x*x % p == r
    # It's faster to compute residues and mark them, then find what's missing.
    is_residue = [False] * p
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        is_residue[r] = True
        if r not in root_map:
            root_map[r] = x
            
    # The non-residues are those r in 1..p-1 where is_residue[r] is False.
    # We want to find the smallest s in 1..1004 such that s, s+1, s+2, s+3, s+4 are all residues.
    # This is equivalent to finding the first gap of length 5 in the non-residues.
    
    # Iterate s from 1 to p-5
    for s in range(1, p - 4):
        # Check if all s..s+4 are residues
        # Using the is_residue array
        if is_residue[s] and is_residue[s+1] and is_residue[s+2] and is_residue[s+3] and is_residue[s+4]:
            roots = [
                root_map[s],
                root_map[s+1],
                root_map[s+2],
                root_map[s+3],
                root_map[s+4]
            ]
            return {'start': s, 'roots': roots}
            
        # Time check (unlikely to trigger for p=1009, but required by directive)
        if time.time() > deadline:
            # If we hit the deadline, we should return the best found so far?
            # But the loop is very fast. This is just a safety guard.
            # Since we are minimizing s, and we scan from 1, if we don't find it, 
            # there is no solution? But there is always a solution.
            # If we timeout, it means the search is too slow. 
            # For p=1009, this is instant.
            pass

    # Fallback (should not happen)
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]} # Invalid but prevents crash
