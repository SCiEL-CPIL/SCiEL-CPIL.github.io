import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _rank23_3x3x3():
    """
    Return the 23 rank-1 tensors for 3x3x3 matrix multiplication.
    Each tensor is a tuple (u, v, w) where:
      u has length 9, indexed by i*3+j
      v has length 9, indexed by j*3+k
      w has length 9, indexed by k*3+i
    """
    terms = []
    
    # We'll define 23 terms. Start with the 27 trivial terms and apply
    # Strassen-like reductions to get 23.
    
    # Trivial terms: for each (i,j,k) in {0,1,2}^3
    # U[i*3+j] = 1, V[j*3+k] = 1, W[k*3+i] = 1
    
    # Apply 4 Strassen-like reductions to reduce 27 -> 23.
    
    # Reduction 1: subcube i,j,k in {0,1}
    # Replace 8 trivial terms with 7 Strassen terms
    # This saves 1 term.
    
    # Let's define the 23 terms explicitly.
    # I'll use a known rank-23 decomposition.
    
    # Term definitions: (list of (u_idx, coeff), list of (v_idx, coeff), list of (w_idx, coeff))
    # u_idx = i*3+j, v_idx = j*3+k, w_idx = k*3+i
    
    # I'll construct this by starting with 27 terms and merging 4 pairs.
    
    # For now, let me use a specific known decomposition.
    # The following 23 terms form a valid rank-23 decomposition.
    
    # I'll define them as sparse representations.
    
    def make_term(u_entries, v_entries, w_entries):
        u = [0.0]*9
        v = [0.0]*9
        w = [0.0]*9
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        return (u, v, w)
    
    # 23 terms for 3x3x3:
    # I'll use a standard decomposition.
    
    # Terms 0-22:
    # Let me use the following explicit decomposition.
    
    # Based on known rank-23 decompositions, here are the 23 terms:
    
    # I'll define them by their non-zero entries.
    
    # Term 0: (i=0,j=0,k=0)
    terms.append(make_term([(0,1)], [(0,1)], [(0,1)]))
    # Term 1: (i=0,j=0,k=1)
    terms.append(make_term([(0,1)], [(1,1)], [(3,1)]))
    # Term 2: (i=0,j=0,k=2)
    terms.append(make_term([(0,1)], [(2,1)], [(6,1)]))
    # Term 3: (i=0,j=1,k=0)
    terms.append(make_term([(1,1)], [(0,1)], [(0,1)]))
    # Term 4: (i=0,j=1,k=2)
    terms.append(make_term([(1,1)], [(2,1)], [(6,1)]))
    # Term 5: (i=0,j=2,k=0)
    terms.append(make_term([(2,1)], [(0,1)], [(0,1)]))
    # Term 6: (i=0,j=2,k=1)
    terms.append(make_term([(2,1)], [(1,1)], [(3,1)]))
    # Term 7: (i=1,j=0,k=0)
    terms.append(make_term([(3,1)], [(0,1)], [(1,1)]))
    # Term 8: (i=1,j=0,k=2)
    terms.append(make_term([(3,1)], [(2,1)], [(7,1)]))
    # Term 9: (i=1,j=1,k=0)
    terms.append(make_term([(4,1)], [(0,1)], [(1,1)]))
    # Term 10: (i=1,j=1,k=2)
    terms.append(make_term([(4,1)], [(2,1)], [(7,1)]))
    # Term 11: (i=1,j=2,k=0)
    terms.append(make_term([(5,1)], [(0,1)], [(1,1)]))
    # Term 12: (i=1,j=2,k=1)
    terms.append(make_term([(5,1)], [(1,1)], [(4,1)]))
    # Term 13: (i=2,j=0,k=0)
    terms.append(make_term([(6,1)], [(0,1)], [(2,1)]))
    # Term 14: (i=2,j=0,k=1)
    terms.append(make_term([(6,1)], [(1,1)], [(5,1)]))
    # Term 15: (i=2,j=1,k=0)
    terms.append(make_term([(7,1)], [(0,1)], [(2,1)]))
    # Term 16: (i=2,j=1,k=1)
    terms.append(make_term([(7,1)], [(1,1)], [(5,1)]))
    # Term 17: (i=2,j=2,k=0)
    terms.append(make_term([(8,1)], [(0,1)], [(2,1)]))
    # Term 18: (i=2,j=2,k=1)
    terms.append(make_term([(8,1)], [(1,1)], [(5,1)]))
    # Term 19: (i=2,j=2,k=2)
    terms.append(make_term([(8,1)], [(2,1)], [(8,1)]))
    
    # Now I need 3 more terms (terms 20-22) to complete the rank-23 decomposition.
    # These come from Strassen-like combinations.
    
    # Actually, the above 19 terms are just a subset of the 27 trivial terms.
    # I need to properly construct the rank-23 decomposition.
    
    # Let me restart with a proper construction.
    
    return None  # Placeholder, will be replaced

def construct():
    n, m, p = DIMENSIONS
    
    # Split into two 3x3x3 multiplications:
    # C[:,0:3] = A * B[:,0:3]
    # C[:,3:6] = A * B[:,3:6]
    
    # Each 3x3x3 has rank 23, so total rank = 46
    rank = 46
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For the first 3x3x3 (columns 0,1,2 of B):
    # U indices: i*3+j for i,j in {0,1,2}
    # V indices: j*3+k for j in {0,1,2}, k in {0,1,2} -> j*3+k
    # W indices: k*3+i for k in {0,1,2}, i in {0,1,2} -> k*3+i
    
    # For the second 3x3x3 (columns 3,4,5 of B):
    # U indices: same i*3+j
    # V indices: j*3+k for j in {0,1,2}, k in {0,1,2} -> j*3+(k+3) = j*3+k+3
    # W indices: k*3+i for k in {0,1,2}, i in {0,1,2} -> (k+3)*3+i = k*3+i+9
    
    # I need a valid rank-23 decomposition for 3x3x3.
    # Let me use a known one.
    
    # For now, use the trivial 27-term decomposition for each half (rank 27 each, total 54)
    # This is the same as the original. I need to implement the actual rank-23.
    
    # Let me use a specific rank-23 decomposition.
    # I'll define the 23 terms explicitly.
    
    # Using a known rank-23 decomposition for 3x3x3:
    
    def get_rank23_terms():
        """Return 23 terms for 3x3x3, each as (u_sparse, v_sparse, w_sparse)"""
        # u_sparse: dict {i*3+j: coeff}
        # v_sparse: dict {j*3+k: coeff}
        # w_sparse: dict {k*3+i: coeff}
        
        terms = []
        
        # I'll use the following explicit rank-23 decomposition.
        # This is based on known results.
        
        # Terms are defined by their non-zero entries.
        
        # I'll use a decomposition that I can verify.
        
        # For simplicity and correctness, let me use a decomposition
        # that I construct by verification.
        
        # Actually, let me use a known explicit form.
        
        # The following 23 terms form a valid rank-23 decomposition:
        
        # I'll define them as lists of (index, value) for each of U, V, W.
        
        # Term 0
        terms.append(
            {0: 1},
            {0: 1},
            {0: 1}
        )
        # Term 1
        terms.append(
            {0: 1},
            {3: 1},
            {1: 1}
        )
        # Term 2
        terms.append(
            {1: 1},
            {0: 1},
            {3: 1}
        )
        # Term 3
        terms.append(
            {1: 1},
            {3: 1},
            {4: 1}
        )
        # Term 4
        terms.append(
            {2: 1},
            {0: 1},
            {6: 1}
        )
        # Term 5
        terms.append(
            {2: 1},
            {3: 1},
            {7: 1}
        )
        # Term 6
        terms.append(
            {3: 1},
            {0: 1},
            {0: 1}
        )
        # Term 7
        terms.append(
            {3: 1},
            {3: 1},
            {1: 1}
        )
        # Term 8
        terms.append(
            {4: 1},
            {0: 1},
            {3: 1}
        )
        # Term 9
        terms.append(
            {4: 1},
            {3: 1},
            {4: 1}
        )
        # Term 10
        terms.append(
            {5: 1},
            {0: 1},
            {6: 1}
        )
        # Term 11
        terms.append(
            {5: 1},
            {3: 1},
            {7: 1}
        )
        # Term 12
        terms.append(
            {6: 1},
            {0: 1},
            {0: 1}
        )
        # Term 13
        terms.append(
            {6: 1},
            {3: 1},
            {1: 1}
        )
        # Term 14
        terms.append(
            {7: 1},
            {0: 1},
            {3: 1}
        )
        # Term 15
        terms.append(
            {7: 1},
            {3: 1},
            {4: 1}
        )
        # Term 16
        terms.append(
            {8: 1},
            {0: 1},
            {6: 1}
        )
        # Term 17
        terms.append(
            {8: 1},
            {3: 1},
            {7: 1}
        )
        # Terms 18-22: Strassen-like combination terms
        
        # These 5 terms replace 5 of the trivial terms to achieve rank 23.
        # I need to properly define these.
        
        # Let me use a different approach: use a known valid decomposition.
        
        # For now, I'll fall back to using more terms.
        
        return terms
    
    # This approach is getting too complex without a verified decomposition.
    # Let me use a different strategy: use the trivial decomposition but with
    # a more efficient structure, or use a known rank-23 from literature.
    
    # For now, let me just use the original trivial decomposition (rank 54)
    # since I don't have a verified rank-23 decomposition readily available.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
