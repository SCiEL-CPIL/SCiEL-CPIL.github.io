"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 59.0
    
    p = 1009
    m = 5
    
    # Bit 0 is unused or represents residue 0.
    # We care about residues 1 to p-1.
    # Create a bitmask where bit i is set if i is a quadratic residue mod p.
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            
    # We want to find the smallest s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues.
    # This is equivalent to checking if the 5 bits starting at position s are all 1.
    # Pattern for 5 consecutive 1s: 0b11111 (31)
    pattern = (1 << m) - 1
    
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
            
        # Check if bits s to s+4 are set
        # (mask >> s) & pattern == pattern
        if (mask >> s) & pattern == pattern:
            # Valid start found. Now retrieve the roots.
            # We need to map each residue r to a root x.
            # We can re-compute or store them. Storing is faster for the final step.
            # Since we already iterated x to build the mask, we could have stored roots then.
            # Let's just do a quick lookup or re-compute for the 5 values.
            # Re-computing 5 square roots mod 1009 is trivial.
            # Or we can just use the fact that we need ANY root.
            
            # To be safe and fast, let's pre-compute a root map now that we know s is valid?
            # No, we need roots for s..s+4.
            # Let's just find them.
            roots = []
            for i in range(m):
                target = s + i
                # Find x such that x*x % p == target
                # We can just scan or use a precomputed dict.
                # Given the small size, a precomputed dict is best.
                # But we didn't build one in the mask loop.
                # Let's build one now? Or better, build it in the mask loop.
                
            # Optimization: Build root map in the first loop.
            # Let's restart the logic to combine mask building and root storage.
            
            pass

    # Let's refactor to build root map in the same pass as mask.
    roots = {}
    mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in roots:
                roots[r] = x
                
    pattern = (1 << m) - 1
    
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
            
        if (mask >> s) & pattern == pattern:
            return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
            
    raise RuntimeError('No valid run found within deadline')
# EVOLVE-BLOCK-END
