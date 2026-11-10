import time

def construct():
    """
    Finds the smallest start 's' such that s, s+1, s+2, s+3, s+4 are quadratic residues mod 1009.
    Returns {'start': s, 'roots': [r0, r1, r2, r3, r4]}
    """
    p = 1009
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # 1. Precompute quadratic residues and their roots
    # roots_map[residue] = root
    # We want to ensure we don't pick 0 as a residue if possible, but the problem 
    # states "zero residues ... are excluded" from the valid set for the run.
    # So we only care about residues 1..1008.
    
    roots_map = {}
    # x^2 mod p is symmetric: x and p-x give same residue.
    # We iterate all x from 1 to p-1 to be safe and simple, 
    # or just 1 to p//2 is enough for existence, but we need the map.
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We just need one root. Storing the first one found is fine.
            # If we want smallest root, we'd need to check, but task doesn't require it.
            if r not in roots_map:
                roots_map[r] = x
    
    # 2. Search for the smallest s
    # s must be between 1 and 1004 (since s+4 <= 1008 to avoid wraparound to 0 or >1008? 
    # Actually, residues are mod 1009, so values are 0..1008.
    # The constraint "1<=ri<=1008" and "zero residues excluded" implies the residues s+i must be in 1..1008.
    # So s+4 must be <= 1008 => s <= 1004.
    # And s >= 1.
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # Fallback: return a known valid late run if we time out (unlikely)
            # But for this small p, it won't time out.
            pass
            
        valid = True
        roots = []
        for i in range(5):
            residue = s + i
            if residue in roots_map:
                roots.append(roots_map[residue])
            else:
                valid = False
                break
        
        if valid:
            return {'start': s, 'roots': roots}
            
    # If no s found (should not happen), raise error
    raise RuntimeError("No valid run found")
