"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4 are
    all quadratic residues modulo 1009, with explicit non-zero roots for each.
    
    Strategy:
    1. Precompute a boolean array is_qr and a roots dict.
    2. Use a sliding window to find the first valid s.
    3. Include a deadline guard to ensure termination.
    """
    p = 1009
    deadline = time.time() + 58.0  # Safety margin
    
    # Precompute is_qr array and roots dict
    is_qr = [False] * p
    roots = {}
    for x in range(1, p):
        if time.time() > deadline:
            # Fallback: return a known valid solution if we run out of time
            # For p=1009, s=1 is expected to work. If not, we might fail.
            # But this loop is O(1009), so it won't take 58s.
            pass
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if r not in roots:
                roots[r] = x
    
    # Scan for the minimum start s
    window_count = sum(1 for i in range(1, 6) if is_qr[i])
    
    for s in range(1, 1005):
        if time.time() > deadline:
            # If we hit the deadline, we must return something valid.
            # We can try to construct a valid result from the last known state or raise.
            # However, the loop is O(1009), so this is unlikely.
            raise RuntimeError("Time limit exceeded")
            
        if window_count == 5:
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}
        
        if s + 5 <= 1008:
            if is_qr[s]:
                window_count -= 1
            if is_qr[s + 5]:
                window_count += 1
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
