import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools
import time

def construct():
    n = 16
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    
    # Generate all pairs and map to index
    all_pairs = []
    pair_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_index[(i, j)] = idx
            all_pairs.append((i, j))
            idx += 1
            
    num_pairs = len(all_pairs)
    
    # Map triples to pair indices
    # triple_pair_indices[t] = [idx1, idx2, idx3]
    triple_pair_indices = []
    for t in all_triples:
        i, j, k = t
        p1 = pair_index[(i, j)]
        p2 = pair_index[(i, k)]
        p3 = pair_index[(j, k)]
        triple_pair_indices.append([p1, p2, p3])
        
    # Function to solve ILP for a given alpha K
    # Constraint: For all S subset of points with |S| = K+1, sum(x_t for t subset of S) >= 1
    # Constraint: For all pairs p, sum(x_t for p in t) <= 1
    def solve_for_alpha(K):
        # Generate all subsets of size K+1
        subsets = list(itertools.combinations(range(n), K+1))
        num_subsets = len(subsets)
        
        # Build coverage constraint matrix A_cov (num_subsets x num_triples)
        # A_cov[s, t] = 1 if triple t is subset of subset s
        # We can build this row by row or column by column.
        # Column by column: for each triple t, find which subsets contain it.
        # A subset S contains t if S is a superset of t.
        # Since |S| = K+1 and |t|=3, S is formed by adding K-2 points to t.
        
        # This might be sparse.
        # Let's create the matrix as a list of lists for now, or use scipy.sparse.
        import scipy.sparse as sp
        
        A_rows = []
        A_cols = []
        A_data = []
        
        for t_idx, t in enumerate(all_triples):
            t_set = set(t)
            # Find all subsets S of size K+1 that contain t
            # The other K-2 points are chosen from the n-3 remaining points
            remaining = [p for p in range(n) if p not in t_set]
            for comb in itertools.combinations(remaining, K-2):
                S = t_set.union(comb)
                # We need the index of S in the 'subsets' list.
                # This is slow if we do it naively.
                # Let's pre-compute a map from frozenset(S) to index.
                pass
                
        # Pre-compute subset index map
        subset_to_idx = {}
        for idx_s, S in enumerate(subsets):
            subset_to_idx[frozenset(S)] = idx_s
            
        # Re-build A
        A_rows = []
        A_cols = []
        A_data = []
        
        for t_idx, t in enumerate(all_triples):
            t_set = set(t)
            remaining = [p for p in range(n) if p not in t_set]
            for comb in itertools.combinations(remaining, K-2):
                S = t_set.union(comb)
                idx_s = subset_to_idx.get(frozenset(S), -1)
                if idx_s != -1:
                    A_rows.append(idx_s)
                    A_cols.append(t_idx)
                    A_data.append(1)
                    
        A_cov = sp.csc_matrix((A_data, (A_rows, A_cols)), shape=(num_subsets, num_triples))
        
        # Build pair constraint matrix A_pair (num_pairs x num_triples)
        # A_pair[p, t] = 1 if pair p is in triple t
        A_pair_rows = []
        A_pair_cols = []
        A_pair_data = []
        for t_idx, p_indices in enumerate(triple_pair_indices):
            for p_idx in p_indices:
                A_pair_rows.append(p_idx)
                A_pair_cols.append(t_idx)
                A_pair_data.append(1)
                
        A_pair = sp.csc_matrix((A_pair_data, (A_pair_rows, A_pair_cols)), shape=(num_pairs, num_triples))
        
        # Combine constraints
        # We want:
        # A_cov @ x >= 1
        # A_pair @ x <= 1
        # 0 <= x <= 1
        
        A = sp.vstack([A_cov, A_pair], format='csc')
        lb = np.concatenate([np.ones(num_subsets), np.zeros(num_pairs)])
        ub = np.concatenate([np.inf * np.ones(num_subsets), np.ones(num_pairs)])
        
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(0, 1)
        
        # Objective: Minimize 0 (feasibility)
        # Or Maximize sum(x) to get a maximal system?
        # Let's just find any feasible solution.
        c = np.zeros(num_triples)
        
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_triples))
        
        if res.status == 0:
            x = res.x
            selected_triples = [all_triples[i] for i in range(num_triples) if x[i] > 0.5]
            return selected_triples
        else:
            return None

    # Try K from small to large
    # Theoretical lower bound for alpha?
    # For n=16, alpha is at least 4?
    # Let's try 4, 5, 6, 7, 8
    
    start_time = time.time()
    for K in range(4, 10):
        if time.time() - start_time > 110:
            break
        triples = solve_for_alpha(K)
        if triples is not None:
            # Verify linearity just in case
            used_pairs = set()
            valid = True
            for t in triples:
                i, j, k = t
                p1 = (i, j) if i < j else (j, i)
                p2 = (i, k) if i < k else (k, i)
                p3 = (j, k) if j < k else (k, j)
                if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                    valid = False
                    break
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
            
            if valid:
                return [list(t) for t in triples]
    
    # Fallback: Random Greedy
    import random
    random.seed(42)
    used_pairs = set()
    cur_triples = []
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    for t in shuffled:
        i, j, k = t
        p1 = (i, j) if i < j else (j, i)
        p2 = (i, k) if i < k else (k, i)
        p3 = (j, k) if j < k else (k, j)
        if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
            cur_triples.append(t)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
    return [list(t) for t in cur_triples]
