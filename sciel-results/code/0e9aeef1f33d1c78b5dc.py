"""Cap set greedy priority with finite-field character-theory-inspired scoring."""
# EVOLVE-BLOCK-START
import time
import hashlib
import numpy as np

# Deadline guard: evaluator kills after 330s; we target finishing well before.
_DEADLINE = 300.0  # seconds of wall-clock budget for the whole evaluation

def _is_cap_set(vecs, n):
    """Check if the set of vectors (list of tuples of 0,1,2) is a cap set.
    A cap set has no three distinct elements a, b, c with a + b + c = 0 mod 3."""
    s = set(vecs)
    if len(s) != len(vecs):
        return False
    for i, a in enumerate(vecs):
        for j in range(i + 1, len(vecs)):
            b = vecs[j]
            # c = -a - b mod 3
            c = tuple((-ai - bi) % 3 for ai, bi in zip(a, b))
            if c in s and c != a and c != b:
                return False
    return True

def _check_cap_add(existing_set, vec, n):
    """Check if adding vec to existing_set (a set of tuples) keeps it a cap set.
    Only need to check triples involving vec."""
    if vec in existing_set:
        return False
    # For each pair (a, b) in existing_set, check if a + b + vec = 0 mod 3
    # i.e., vec = -a - b mod 3
    # Equivalently, for each a in existing_set, check if (-a - vec) mod 3 is in existing_set
    # and (-a - vec) != a and (-a - vec) != vec
    for a in existing_set:
        c = tuple((-ai - vi) % 3 for ai, vi in zip(a, vec))
        if c in existing_set and c != a and c != vec:
            return False
    return True

def priority(vector, n):
    """
    Compute a priority for a vector in {0,1,2}^n.
    Higher priority = added earlier in the greedy cap set construction.
    
    Strategy: Use additive character theory. For a fixed pseudo-random frequency
    vector w in F_3^n, compute the character sum magnitude which measures deviation
    from uniformity. Vectors that are 'more spread out' in Fourier space get higher
    priority, helping avoid clumping.
    
    Additionally, use Hamming weight balance (close to n/2) as a secondary criterion
    to avoid extreme weights that form small cap sets.
    """
    # Deterministic pseudo-random frequency vector based on n
    # Use a hash-based approach to generate w deterministically
    seed_bytes = hashlib.sha256(f"capset_freq_{n}".encode()).digest()
    # Generate n values in {0, 1, 2} from the hash
    w = []
    for i in range(n):
        byte_val = seed_bytes[i % 32]
        w.append(byte_val % 3)
    w = np.array(w, dtype=np.int64)
    
    v = np.array(vector, dtype=np.int64)
    
    # Compute the additive character: sum over coordinates of w_i * v_i mod 3
    # The character value is omega^(w·v) where omega = e^(2*pi*i/3)
    # But we want a real-valued priority. Use the real part: cos(2*pi/3 * (w·v mod 3))
    dot = int(np.dot(w, v)) % 3
    # cos(2*pi*k/3) for k=0,1,2: 1, -0.5, -0.5
    char_real = [1.0, -0.5, -0.5][dot]
    
    # Vectors with character value closer to 0 (i.e., -0.5) are more "spread out"
    # in the sense that they don't align with the frequency w.
    # Actually, for cap sets, we want to avoid arithmetic progressions.
    # A different approach: prioritize vectors that are far from forming APs.
    
    # Secondary: Hamming weight balance
    # Count non-zero entries (or use a weighted sum)
    weight = float(np.sum(v != 0))
    # Prefer weights close to n/2
    weight_score = 1.0 - abs(weight - n / 2.0) / (n / 2.0) if n > 0 else 0.0
    
    # Tertiary: Use a hash-based tiebreaker to avoid lexicographic bias
    vec_hash = int.from_bytes(hashlib.md5(str(vector).encode()).digest()[:8], 'big')
    hash_score = float(vec_hash) / float(2**64 - 1)
    
    # Combine: primary is character-based (spread out), secondary is weight balance
    # Higher char_real means more aligned with w (less spread), so we want lower char_real
    # But actually, we want to maximize diversity. Let's use:
    # priority = weight_score * 2.0 + (1.0 - char_real) * 1.0 + hash_score * 0.1
    
    # Actually, let's think more carefully. The character sum approach:
    # For a set S, the Fourier coefficient at frequency w is |sum_{v in S} omega^(w·v)|
    # To minimize this, we want the sum to be close to 0, which means the characters
    # should be balanced. But for individual vectors, we can't directly control this.
    
    # Alternative: Use the L2 norm of the vector's "Fourier profile"
    # For a single vector, compute how it interacts with multiple frequencies.
    
    # Let's use multiple frequencies for a more robust measure
    scores = []
    for seed_idx in range(5):
        seed_bytes = hashlib.sha256(f"capset_freq_{n}_{seed_idx}".encode()).digest()
        w_i = np.array([seed_bytes[j % 32] % 3 for j in range(n)], dtype=np.int64)
        dot_i = int(np.dot(w_i, v)) % 3
        char_i = [1.0, -0.5, -0.5][dot_i]
        scores.append(char_i)
    
    # Average character value across multiple frequencies
    avg_char = float(np.mean(scores))
    # Lower avg_char means more spread out (more -0.5 values)
    spread_score = 1.0 - avg_char  # ranges from 0.0 to 1.5
    
    # Final priority: combine spread, weight balance, and hash tiebreaker
    priority_val = spread_score * 2.0 + weight_score * 1.5 + hash_score * 0.3
    
    return float(priority_val)
# EVOLVE-BLOCK-END
