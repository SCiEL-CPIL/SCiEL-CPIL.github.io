import zlib

def priority(vector, n):
        # Structural score: prefer vectors with fewer zeros (higher weight)
        # But add noise to avoid rigid lock-in
        zeros = 0
        for x in vector:
            if x == 0:
                zeros += 1
        
        # Score is primarily based on weight, with a random tie-breaker
        # We want to prefer high weight, but not exclusively
        # Let's use a random hash as the main driver, but bias it
        data = bytes(vector)
        h = zlib.crc32(data)
        
        # Bias: add a term that favors fewer zeros
        # Scale: max zeros is n.
        # Let's say the bias is 0.1 * (n - zeros)
        # This is small compared to the hash (which is ~2^32)
        # This might be too small to have an effect.
        
        # Let's try a stronger bias.
        # If we want to mix S_n and S_{n-1}, the bias needs to be comparable to the hash variance.
        # Hash is uniform [0, 2^32].
        # Let's add (n - zeros) * 10^9.
        
        bias = (n - zeros) * 10**9
        return float(h) + bias
