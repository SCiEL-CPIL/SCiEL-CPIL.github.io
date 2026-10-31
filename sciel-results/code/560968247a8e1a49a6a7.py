"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for greedy cap set construction.
    Lower priority value = higher preference (added first).
    
    We minimize the number of affine lines this vector would form
    with pairs in the current set, plus a penalty for how many
    future vectors it would block.
    """
    # Convert to numpy array for efficient computation
    v = np.array(vector, dtype=np.int64)
    
    # We need access to the current set. Since the contract says
    # "vectors are added greedily in descending priority", the evaluator
    # must maintain the set. But our function only gets (vector, n).
    # 
    # The key insight: we cannot access the current set directly from
    # this function signature. So we need a different approach.
    #
    # Alternative: use structural properties of the vector itself to
    # predict its "danger" in a cap set context.
    #
    # Vectors with more "balanced" coordinate distributions tend to
    # form fewer lines. Vectors that are "close" to the origin or
    # have many repeated coordinates may be more dangerous.
    #
    # A better approach: count the number of "line partners" a vector
    # has in the full space. For vector v, the number of pairs (a,b)
    # such that v,a,b form a line is related to how many vectors w
    # satisfy w ≡ -v mod 3 being in S... but we don't know S.
    #
    # Let's use a different structural heuristic:
    # - Count the number of distinct values in the vector
    # - Vectors with more 0s tend to be "safer" as they're less likely
    #   to form lines with arbitrary pairs
    # - Use the L1 norm or coordinate distribution as a proxy
    
    # Count coordinate frequencies
    counts = np.bincount(v, minlength=3)
    
    # Entropy of coordinate distribution: more uniform = more dangerous
    # (more symmetric vectors form more lines)
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([1/3, 1/3, 1/3])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # L1 distance from uniform: vectors far from uniform are "safer"
    uniform_dist = np.sum(np.abs(counts - n/3))
    
    # Number of zero coordinates
    num_zeros = counts[0]
    
    # Combine: prefer vectors with low entropy (less symmetric)
    # and high zero count (less likely to form lines)
    # Return negative so that "safer" vectors have higher priority (lower value)
    
    score = entropy * 1.0 - uniform_dist * 0.3 - num_zeros * 0.2
    return score
# EVOLVE-BLOCK-END
