"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4 are
    all quadratic residues modulo 1009, with explicit non-zero roots for each.
    
    Strategy:
    1. Precompute a boolean array is_qr where is_qr[r] is True if r is a non-zero quadratic residue mod 1009.
    2. Precompute a roots dict mapping each non-zero QR to its smallest positive root.
    3. Use a sliding window of length 5 over the is_qr array to find the first s where all 5 consecutive values are True.
    4. Return the first valid s with its roots.
    
    This is O(p) preprocessing and O(p) scanning, which is very fast.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    p = 1009
    
    # Precompute is_qr array and roots dict
    is_qr = [False] * p
    roots = {}
    
    # x ranges from 1 to p-1
    for x in range(1, p):
        if time.time() > deadline:
            break
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            # Store the smallest root
            if r not in roots:
                roots[r] = x
    
    # We need to find s in [1, 1004] such that is_qr[s] ... is_qr[s+4] are all True.
    # Note: s+4 can be up to 1008.
    
    # Sliding window optimization
    # Window is [s, s+1, s+2, s+3, s+4]
    # Initially s=1, window is [1, 5]
    
    # Count QRs in [1, 5]
    count = 0
    for i in range(1, 6):
        if is_qr[i]:
            count += 1
    
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        if count == 5:
            # Found the minimum s
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
        
        # Slide window to next s
        # Remove s, add s+5
        if s < 1004:
            if is_qr[s]:
                count -= 1
            next_val = s + 5
            if next_val <= 1008:
                if is_qr[next_val]:
                    count += 1
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
