import time
import itertools
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for triples
    tri_masks = np.array([0] * n_triples, dtype=np.uint32)
    for i, t in enumerate(all_triples):
        m = 0
        for p in t:
            m |= (1 << p)
        tri_masks[i] = m
        
    if HAS_SCIPY_MILP:
        # Precompute linearity matrix
        pairs = list(itertools.combinations(points, 2))
        n_pairs = len(pairs)
        pair_masks = np.array([0] * n_pairs, dtype=np.uint32)
        for j, (u, v) in enumerate(pairs):
            m = 0
            m |= (1 << u)
            m |= (1 << v)
            pair_masks[j] = m
            
        # A_lin[j, i] = 1 if pair j is subset of triple i
        # (pair_masks[j] & tri_masks[i]) == pair_masks[j]
        A_lin = ((pair_masks[:, None] & tri_masks[None, :]) == pair_masks[:, None]).astype(int)
        
        def solve_for_k(K):
            subsets = list(itertools.combinations(points, K+1))
            n_subsets = len(subsets)
            
            sub_masks = np.array([0] * n_subsets, dtype=np.uint32)
            for j, S in enumerate(subsets):
                m = 0
                for p in S:
                    m |= (1 << p)
                sub_masks[j] = m
                
            # A_ind[j, i] = 1 if triple i is subset of subset j
            # (sub_masks[j] & tri_masks[i]) == tri_masks[i]
            A_ind = ((sub_masks[:, None] & tri_masks[None, :]) == tri_masks[None, :]).astype(int)
            
            A = np.vstack([-A_ind, A_lin])
            b = np.concatenate([-np.ones(n_subsets), np.ones(n_pairs)])
            
            # Maximize number of triples (minimize negative sum)
            c = -np.ones(n_triples)
            
            bounds = Bounds(0, 1)
            cons = LinearConstraint(A, -np.inf, b)
            
            res = milp(c=c, constraints=cons, integrality=np.ones(n_triples), bounds=bounds, options={"time_limit": 15})
            
            if res.status == 0:
                x = res.x
                triples = []
                for i in range(n_triples):
                    if x[i] > 0.5:
                        triples.append(list(all_triples[i]))
                return triples
            return None

        for K in range(6, 11):
            if time.time() > deadline - 10:
                break
            t = solve_for_k(K)
            if t is not None:
                return t

    # Fallback: Return a simple valid system
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

# EVOLVE-BLOCK-END
