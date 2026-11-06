import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest s >= 1 such that s, s+1, s+2, s+3, s+4 are all
    non-zero quadratic residues mod 1009, with roots for each.
    
    Uses the Legendre symbol sign sequence e_x:
      e_x = 1 if x is a non-zero QR mod p
      e_x = -1 if x is a non-residue mod p
      e_x = 0 if x == 0
    
    The condition that s..s+4 are all residues is equivalent to:
      sum(e_{s+i} for i in 0..4) == 5
    (which also implies each e_{s+i} == 1, since max value is 1)
    
    We use a sliding window over the sign sequence for O(p) total work.
    """
    p = 1009
    m = 5
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Build sign sequence: e[i] = 1 if i is non-zero QR, -1 if non-residue, 0 if i==0
    e = [0] * (p + 1)
    roots = [0] * (p + 1)
    
    for x in range(1, p):
        sq = (x * x) % p
        if e[sq] == 0 and sq != 0:
            # First time we see this residue
            roots[sq] = x
        e[sq] = 1  # mark as QR
    
    # Now set non-residues to -1
    for i in range(1, p):
        if e[i] == 0:
            e[i] = -1
    e[0] = 0  # zero is excluded
    
    # Sliding window: compute sum of e[s..s+4] for each s in [1, p-5]
    # We want the smallest s where this sum == 5 (all five are QR)
    # 
    # Initial window: s=1, sum = e[1]+e[2]+e[3]+e[4]+e[5]
    window_sum = sum(e[1:1+m])
    
    best_s = None
    for s in range(1, p - m + 1):
        if s > 1:
            # Slide window: remove e[s-1], add e[s+m-1]
            window_sum -= e[s - 1]
            window_sum += e[s + m - 1]
        
        if window_sum == m:
            best_s = s
            break
        
        # Deadline check (very cheap, just in case)
        if time.time() > deadline:
            break
    
    if best_s is None:
        # Fallback: brute force check (shouldn't happen for p=1009)
        for s in range(1, p - m + 1):
            if all(e[s + i] == 1 for i in range(m)):
                best_s = s
                break
    
    if best_s is None:
        best_s = 1  # absolute fallback
    
    # Extract roots
    res_roots = [roots[best_s + i] for i in range(m)]
    
    # Verify
    for i in range(m):
        assert 1 <= res_roots[i] <= p - 1
        assert (res_roots[i] ** 2) % p == best_s + i
    
    return {'start': best_s, 'roots': res_roots}
# EVOLVE-BLOCK-END
