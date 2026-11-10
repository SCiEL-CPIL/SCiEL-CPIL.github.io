"""
Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Returns the start index and the corresponding smallest positive square roots.
"""

def construct():
    p = 1009
    m = 5
    
    # Boolean array to check if a number is a quadratic residue
    is_res = [False] * p
    # Array to store the smallest positive root for each residue
    root_map = [0] * p
    
    # Precompute residues and roots
    # Iterating from 1 to p-1 ensures the first root found is the smallest positive one
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_res[r] = True
            if root_map[r] == 0:
                root_map[r] = x

    # Find the smallest s such that s, s+1, ..., s+m-1 are all quadratic residues
    # s ranges from 1 to p-m (inclusive) to ensure s+m-1 <= p-1
    for s in range(1, p - m + 1):
        # Check all m consecutive values
        all_res = True
        for i in range(m):
            if not is_res[s + i]:
                all_res = False
                break
        
        if all_res:
            roots = [root_map[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # This should never be reached for p=1009, m=5
    raise RuntimeError("No consecutive run of quadratic residues found")
