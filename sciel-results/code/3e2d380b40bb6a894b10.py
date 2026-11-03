import itertools
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy fallback
        total_words = q ** n
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        neighbors = [[] for _ in range(total_words)]
        for i, w in enumerate(words):
            nbrs = [i]
            for pos in range(n):
                for new_digit in range(q):
                    if new_digit != w[pos]:
                        new_w = list(w)
                        new_w[pos] = new_digit
                        new_w = tuple(new_w)
                        nbrs.append(word_to_idx[new_w])
            neighbors[i] = nbrs
        
        covered = [False] * total_words
        code = []
        while any(not c for c in covered):
            best_count = -1
            best_candidates = []
            for w in range(total_words):
                count = sum(1 for nbr in neighbors[w] if not covered[nbr])
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
            selected = random.choice(best_candidates)
            code.append(selected)
            for nbr in neighbors[selected]:
                covered[nbr] = True
        return [list(words[w]) for w in code]
    
    # Optimized path for q=3, n=6, radius=1
    total_words = 729
    n = 6
    q = 3
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        nbrs = [w]
        for pos in range(n):
            div = 3 ** (n - 1 - pos)
            curr_digit = (w // div) % q
            for new_digit in range(q):
                if new_digit != curr_digit:
                    new_w = w + (new_digit - curr_digit) * div
                    nbrs.append(new_w)
        neighbors[w] = nbrs
    
    # Try ILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Variables: x_i for i in 0..728
        # Objective: min sum(x_i)
        c = np.ones(total_words)
        
        # Constraints: For each word j, sum(x_i for i in neighbors_of_j) >= 1
        # Note: neighbors[w] contains centers that cover w? 
        # No, neighbors[w] in my precomputation above is: 
        # "for w in range(total_words): nbrs = [w]... new_w... neighbors[w] = nbrs"
        # This computes the set of words within distance 1 of w.
        # So if we pick center c, it covers words in neighbors[c].
        # We need: for every word j, there exists a center c such that j is in neighbors[c].
        # This is equivalent to: for every word j, sum(x_c for c such that j in neighbors[c]) >= 1.
        
        # Let's build the constraint matrix A where A[j, c] = 1 if j is in neighbors[c]
        # This is the transpose of the incidence matrix of centers to covered words.
        
        # Instead of building a dense 729x729 matrix, we can use sparse.
        # Actually, 729x729 is small enough for dense, but sparse is better.
        
        row_indices = []
        col_indices = []
        data = []
        
        for c in range(total_words):
            for j in neighbors[c]:
                row_indices.append(j)
                col_indices.append(c)
                data.append(1)
                
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(total_words, total_words))
        
        # Constraints: A @ x >= 1
        # milp requires constraints in form lb <= A_ub @ x <= ub
        # So -A @ x <= -1  =>  lb = -inf, ub = -1? 
        # Or just A @ x >= 1 => -A @ x <= -1.
        # scipy.optimize.milp takes LinearConstraint(A, lb, ub)
        # We want A @ x >= 1.
        # This is equivalent to -A @ x <= -1.
        # So we can pass A_matrix = -A, lb = -inf, ub = -1.
        # Or does it support >=? 
        # Documentation: "Linear constraint: lb <= A @ x <= ub"
        # So we must use -A.
        
        A_con = -A
        lb = -np.inf * np.ones(total_words)
        ub = -np.ones(total_words)
        
        constraints = LinearConstraint(A_con, lb, ub)
        integrality = np.ones(total_words) # all binary
        bounds = Bounds(0, 1)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            code_ints = [i for i in range(total_words) if x[i] > 0.5]
            
            # Convert to list of lists
            final_code = []
            for w in code_ints:
                word = [0] * n
                temp = w
                for i in range(n-1, -1, -1):
                    word[i] = temp % q
                    temp //= q
                final_code.append(word)
            return final_code
        else:
            # ILP failed or timed out, fallback to Local Search
            pass
            
    except Exception:
        # Fallback to Local Search
        pass

    # Fallback: Local Search Heuristic
    # 1. Greedy construction
    # 2. Local optimization: try to remove redundant codewords, try swaps
    
    rng = random.Random(42)
    best_code = None
    best_len = float('inf')
    
    for restart in range(50):
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Greedy Phase
        while not covered.all():
            best_count = -1
            best_candidates = []
            for w in range(total_words):
                # Count new covers
                # Optimization: only consider w that cover at least one new point
                # Simple loop for now
                count = 0
                for nbr in neighbors[w]:
                    if not covered[nbr]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_candidates = [w]
                elif count == best_count and count > 0:
                    best_candidates.append(w)
                    
            if best_count == 0:
                break
            selected = rng.choice(best_candidates)
            code.append(selected)
            for nbr in neighbors[selected]:
                covered[nbr] = True
                
        # Local Search Phase
        improved = True
        while improved:
            improved = False
            
            # 1. Try to remove redundant codewords
            for i in range(len(code)):
                # Check if removing code[i] leaves all its neighbors covered by others
                # We need to know if any neighbor of code[i] is ONLY covered by code[i]
                # This requires counting coverage.
                
                # To be efficient, we can maintain a count array.
                # But for simplicity in this fallback, we do a check.
                # If len(code) is small, this is fast.
                
                # Let's use a coverage count array for the current code
                # This is O(N * CodeSize) per check, which might be slow.
                # Let's optimize: 
                # We can precompute for each word, which codewords cover it?
                # That's expensive to update.
                
                # Alternative: Just try to remove, check validity, if invalid, put back.
                # Check validity: 
                # For all nbr in neighbors[code[i]], is there another c in code\{i} such that nbr in neighbors[c]?
                
                is_redundant = True
                for nbr in neighbors[code[i]]:
                    # Check if any other codeword covers nbr
                    covered_by_other = False
                    for j, c in enumerate(code):
                        if j == i:
                            continue
                        if nbr in neighbors[c]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    code.pop(i)
                    improved = True
                    break # Restart removal loop as indices changed
                    
            # 2. Try to swap one codeword with another to improve
            # This is complex to implement efficiently in a short time.
            # Let's stick to removal and maybe adding if gaps exist (shouldn't happen if we start valid).
            
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
    # Convert best_code to output format
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code
