import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # We don't know rank yet, but max is n*m*p = 84.
    # We will build it dynamically or pre-allocate and slice.
    # Let's pre-allocate for 84 and then slice to actual rank.
    
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    current_rank = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal current_rank
        U[i * m + j, current_rank] = 1.0
        V[j * p + k, current_rank] = 1.0
        W[k * n + i, current_rank] = 1.0
        current_rank += 1

    # Helper to add a Strassen block
    # i0, i1: row indices
    # j0, j1: col indices of A / row indices of B
    # k0, k1: col indices of B / col indices of C
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal current_rank
        
        # Local indices 0, 1 map to global i0, i1 etc.
        # A indices: (i, j) -> global (i0+i, j0+j)
        # B indices: (j, k) -> global (j0+j, k0+k)
        # C indices: (i, k) -> global (i0+i, k0+k)
        
        # Strassen Multiplications M_1 to M_7
        # Each M_q is defined by coefficients for A (alpha), B (beta), and C (gamma)
        
        # M_1: (A00 + A11) * (B11 - B00)
        # Alpha: A00:1, A11:1
        # Beta: B11:1, B00:-1
        # Gamma: C00:1, C11:1
        # Note: C00 is local (0,0), C11 is local (1,1)
        
        # M_2: (A10 + A11) * B00
        # Alpha: A10:1, A11:1
        # Beta: B00:1
        # Gamma: C10:1, C11:-1
        
        # M_3: A00 * (B01 - B11)
        # Alpha: A00:1
        # Beta: B01:1, B11:-1
        # Gamma: C01:1, C11:1
        
        # M_4: A11 * (B10 - B00)
        # Alpha: A11:1
        # Beta: B10:1, B00:-1
        # Gamma: C00:1, C10:1
        
        # M_5: (A00 + A01) * B11
        # Alpha: A00:1, A01:1
        # Beta: B11:1
        # Gamma: C00:-1, C01:1
        
        # M_6: (A10 - A00) * (B00 + B01)
        # Alpha: A10:1, A00:-1
        # Beta: B00:1, B01:1
        # Gamma: C11:1
        
        # M_7: (A01 - A11) * (B10 + B11)
        # Alpha: A01:1, A11:-1
        # Beta: B10:1, B11:1
        # Gamma: C00:1
        
        # List of (alpha_dict, beta_dict, gamma_dict)
        # Keys are (local_i, local_j) for A, (local_j, local_k) for B, (local_i, local_k) for C
        strassen_terms = [
            # M_1
            ({(0,0): 1, (1,1): 1}, {(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}),
            # M_2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M_3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M_4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M_5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M_6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M_7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for alpha, beta, gamma in strassen_terms:
            # Add term to U
            for (li, lj), val in alpha.items():
                gi = i0 + li
                gj = j0 + lj
                U[gi * m + gj, current_rank] += val
            
            # Add term to V
            for (lj, lk), val in beta.items():
                gj = j0 + lj
                gk = k0 + lk
                V[gj * p + gk, current_rank] += val
            
            # Add term to W
            for (li, lk), val in gamma.items():
                gi = i0 + li
                gk = k0 + lk
                W[gk * n + gi, current_rank] += val
            
            current_rank += 1

    # Define the 6 disjoint Strassen blocks
    # Dimensions: n=3, m=4, p=7
    # J partitions: {0,1} and {2,3}
    # I pairs: {0,1} (leaves 2)
    # K pairs: {0,1}, {2,3}, {4,5} (leaves 6)
    
    # Blocks using J={0,1}
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 0, 1, 2, 3)
    add_strassen_block(0, 1, 0, 1, 4, 5)
    
    # Blocks using J={2,3}
    add_strassen_block(0, 1, 2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    add_strassen_block(0, 1, 2, 3, 4, 5)
    
    # Now add schoolbook for all remaining triples
    # We need to know which triples are already covered.
    # Covered triples are those where:
    # (i in {0,1} AND j in {0,1} AND k in {0,1,2,3,4,5})
    # OR
    # (i in {0,1} AND j in {2,3} AND k in {0,1,2,3,4,5})
    # Essentially, if i in {0,1} and j in {0,1,2,3} and k in {0,1,2,3,4,5}, it is covered.
    # Wait, the blocks cover specific K pairs.
    # Block 1 covers k in {0,1}. Block 2 covers k in {2,3}. Block 3 covers k in {4,5}.
    # So for J={0,1}, k in {0,1,2,3,4,5} is covered.
    # For J={2,3}, k in {0,1,2,3,4,5} is covered.
    # So if i in {0,1} and j in {0,1,2,3} and k in {0,1,2,3,4,5}, it is covered.
    
    covered = np.zeros((n, m, p), dtype=bool)
    
    # Mark covered regions
    # J=0,1
    covered[0:2, 0:2, 0:6] = True
    # J=2,3
    covered[0:2, 2:4, 0:6] = True
    
    # Iterate over all triples and add schoolbook if not covered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    add_schoolbook(i, j, k)
    
    # Slice to actual rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    # Convert to list of lists of floats (or ints if possible)
    # The problem asks for integer/half-integer entries.
    # Strassen coefficients are integers (-1, 0, 1).
    # Schoolbook coefficients are integers (1).
    # So all entries should be integers.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verification step (optional but good for debugging)
    # def verify(U, V, W, r):
    #     ...
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
