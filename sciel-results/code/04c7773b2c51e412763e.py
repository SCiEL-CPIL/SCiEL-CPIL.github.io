"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import zlib

def priority(vector, n):
    """
    Assign priority to vectors in {0,1,2}^n to maximize cap set size via greedy insertion.
    
    Strategy:
    - Hybrid structured-random approach.
    - The set {0,1}^n is a cap set. By prioritizing vectors without the symbol '2',
      we ensure the greedy algorithm first populates this large, safe subspace.
    - Vectors containing '2' are prioritized lower and ordered pseudo-randomly to
      attempt to extend the cap set beyond the {0,1}^n core.
    """
    # Count occurrences of 2 in the vector
    num_twos = sum(1 for x in vector if x == 2)
    
    if num_twos == 0:
        # High priority for vectors in {0,1}^n
        return 0.0
    else:
        # Lower priority for vectors with 2s, ordered randomly
        vec_str = ''.join(map(str, vector))
        return 1.0 + float(zlib.crc32(vec_str.encode()))
# EVOLVE-BLOCK-END
