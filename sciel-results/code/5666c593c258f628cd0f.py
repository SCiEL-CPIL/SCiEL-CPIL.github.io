import hashlib

def priority(vector, n):
    # Check if vector is in {0,1}^n
    is_binary = all(v in (0, 1) for v in vector)
    
    if is_binary:
        # High priority for binary vectors
        # To ensure deterministic but varied order among binaries, 
        # we can use a hash of the binary part to break ties, 
        # but they all get the same high bucket.
        # Actually, if they all have the same priority, the evaluator's 
        # stable sort or internal order might matter. 
        # Better to give them distinct priorities to control the order.
        # But any order of {0,1}^n is a cap set.
        # So we can just return 1.0 for all? 
        # If the evaluator uses a stable sort, the original order (lexicographic?) 
        # might be preserved. 
        # Let's add a small hash component to ensure we don't rely on 
        # the evaluator's internal order, although it shouldn't matter for size.
        b = bytes(vector)
        h = hashlib.sha256(b).digest()
        val = int.from_bytes(h[:8], 'big')
        # Scale to [0.9, 1.0)
        return 0.9 + (val / 2**64) * 0.1
    else:
        # Lower priority for non-binary vectors
        b = bytes(vector)
        h = hashlib.sha256(b).digest()
        val = int.from_bytes(h[:8], 'big')
        # Scale to [0.0, 0.9)
        return (val / 2**64) * 0.9
