"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 115
    
    # Approach: Binary Spherical Code
    # Map bits to {-1, 1}. 
    # Norm of each vector is sqrt(D) = sqrt(11).
    # Distance squared between u, v is sum((u_i - v_i)^2).
    # If u_i, v_i in {-1, 1}, difference is 0 or +/-2.
    # dist^2 = 4 * HammingDistance(u, v).
    # Constraint: dist^2 >= max_norm^2 = D = 11.
    # 4 * d_H >= 11 => d_H >= 2.75 => d_H >= 3.
    # Wait, 4 * 3 = 12 >= 11. So d_H >= 3 is sufficient.
    # Let's re-verify.
    # If d_H = 3, dist^2 = 12. Norm^2 = 11. 12 >= 11. OK.
    # If d_H = 2, dist^2 = 8. 8 < 11. Fail.
    # So we need a binary code of length 11 with minimum Hamming distance 3.
    
    # This is a much looser constraint than d_H >= 6.
    # The maximum size of a binary code with d_H >= 3 is given by Hamming bound?
    # For d=3, it's a Single Error Correcting Code.
    # Max size is roughly 2^11 / (1 + 11) = 2048 / 12 ~ 170.
    # We can likely find a code of size > 100.
    
    # Generate all binary vectors
    n_bits = D
    n_vectors = 2 ** n_bits
    # Use integer representation for speed
    # We need to compute pairwise Hamming distances.
    
    # Greedy construction with Simulated Annealing
    
    # Precompute Hamming distances? 2048x2048 matrix is 4M entries. 
    # int8 is fine.
    # Actually, we can just compute on the fly or precompute a bitmask distance table.
    
    # Let's use a greedy approach first to get a good initial set.
    # Then SA to improve.
    
    # Represent vectors as integers 0..2047
    # popcount(i ^ j) gives Hamming distance.
    
    def hamming_dist(i, j):
        return bin(i ^ j).count('1')
        
    # This is slow in Python loop. Use numpy or precomputed table?
    # 2048 is small enough that we can precompute a distance matrix if needed, 
    # but SA will do many lookups.
    # Let's precompute popcount for 0..2047? No, distance is between any two.
    # We can precompute a 2048x2048 distance matrix.
    # 2048*2048 = 4,194,304 bytes. ~4MB. Acceptable.
    
    dist_matrix = np.zeros((n_vectors, n_vectors), dtype=np.int8)
    for i in range(n_vectors):
        # Use vectorized XOR and popcount?
        # np.bitwise_xor(i, np.arange(n_vectors))
        # Then count bits.
        # This is still slow in pure Python/Numpy loops.
        # Let's just use the greedy method with on-the-fly calculation 
        # but optimize the check.
        
        # Actually, for D=11, we can just use the property that we need d_H >= 3.
        # This is equivalent to finding a large code.
        
        # Let's use a simple greedy:
        # 1. Randomize order of all 2048 vectors.
        # 2. Iterate, add if compatible with all already added.
        
        best_set = []
        rng = np.random.default_rng(42)
        
        # We will run multiple restarts of Greedy + Local Search (Swap)
        # But specifically for Binary Codes.
        
        # Precompute compatible masks?
        # For each vector i, mask of vectors j such that d_H(i,j) >= 3.
        # This is a 2048x2048 boolean matrix.
    
    # Let's build the compatibility matrix efficiently.
    # d_H(i, j) < 3 means d_H is 0, 1, or 2.
    # For each i, the number of vectors at distance 1 is 11.
    # At distance 2 is C(11, 2) = 55.
    # At distance 0 is 1.
    # So each vector is INCOMPATIBLE with 1 + 11 + 55 = 67 vectors.
    # It is COMPATIBLE with 2048 - 67 = 1981 vectors.
    # This is a very sparse incompatibility graph!
    # We can represent incompatibility as a list of neighbors.
    
    incompat = [[] for _ in range(n_vectors)]
    for i in range(n_vectors):
        # Distance 0
        incompat[i].append(i)
        # Distance 1
        for b in range(n_bits):
            j = i ^ (1 << b)
            incompat[i].append(j)
        # Distance 2
        for b1 in range(n_bits):
            for b2 in range(b1 + 1, n_bits):
                j = i ^ (1 << b1) ^ (1 << b2)
                incompat[i].append(j)
        
    # Now, greedy selection:
    # Maintain a set of selected vectors.
    # A vector v is compatible if it is not in the "forbidden" set.
    # Forbidden set = union of incompatibility lists of all selected vectors.
    
    best_size = 0
    best_selection = []
    
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        order = rng.permutation(n_vectors)
        selected = []
        # Use a boolean array for "forbidden"
        forbidden = np.zeros(n_vectors, dtype=bool)
        
        for v in order:
            if not forbidden[v]:
                selected.append(v)
                # Mark all incompatible vectors as forbidden
                for u in incompat[v]:
                    forbidden[u] = True
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_selection = selected.copy()
            
    # Convert best_selection (integers) to integer vectors {-1, 1}^11
    # The task requires integer vectors. {-1, 1} are integers.
    
    result = np.zeros((best_size, D), dtype=np.int64)
    for i, val in enumerate(best_selection):
        for b in range(D):
            if (val >> b) & 1:
                result[i, b] = 1
            else:
                result[i, b] = -1
                
    return result

# EVOLVE-BLOCK-END
