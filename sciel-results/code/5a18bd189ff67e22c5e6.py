"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    """
    if q == 3 and n == 6 and radius == 1:
        return _solve_cover_milp()
    else:
        return _generic_greedy_cover(q, n, radius)

def _solve_cover_milp():
    """
    Solves K_3(6,1) using Mixed Integer Linear Programming (MILP).
    Formulates as Set Cover: Minimize sum(x_w) s.t. for all u, sum_{v: d(u,v)<=1} x_v >= 1.
    Uses symmetry breaking: Fix x_000000 = 1.
    """
    alphabet = range(3)
    all_words = list(itertools.product(alphabet, repeat=6))
    N = len(all_words) # 729
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build the constraint matrix A (Sparse)
    # A[i, j] = 1 if word j covers word i (i.e., d(word_i, word_j) <= 1)
    rows = []
    cols = []
    data = []
    
    for i, w_i in enumerate(all_words):
        # Word i is covered by itself
        rows.append(i)
        cols.append(i)
        data.append(1)
        
        # Word i is covered by its neighbors
        for pos in range(6):
            for val in alphabet:
                if val == w_i[pos]:
                    continue
                new_w = list(w_i)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                rows.append(i)
                cols.append(j)
                data.append(1)
                
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    # Objective: Minimize number of codewords
    c = np.ones(N)
    
    # Constraints: A x >= 1
    lb_constraint = np.ones(N)
    ub_constraint = np.inf * np.ones(N)
    constraints = LinearConstraint(A, lb=lb_constraint, ub=ub_constraint)
    
    # Integrality: All variables are binary (integer)
    integrality = np.ones(N)
    
    # Bounds: 0 <= x_i <= 1
    # Symmetry Breaking: Fix x_0 (word 000000) to 1
    lb = np.zeros(N)
    ub = np.ones(N)
    lb[0] = 1.0
    ub[0] = 1.0
    
    # Solve
    # options can be tuned, but defaults for CBC are usually fine for this size
    res = milp(c,
               constraints=constraints,
               integrality=integrality,
               bounds=Bounds(lb, ub))
               
    if res.status == 0:
        x = res.x
        codewords = [all_words[i] for i in range(N) if x[i] > 0.5]
        return [list(cw) for cw in codewords]
    else:
        # Fallback to greedy if MILP fails or times out (unlikely for 729 vars)
        return _fallback_greedy_cover()

def _fallback_greedy_cover():
    """
    Fallback greedy set cover with local search.
    """
    alphabet = range(3)
    all_words = list(itertools.product(alphabet, repeat=6))
    n_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    cover_sets = []
    for w in all_words:
        covered = {word_to_idx[w]}
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.add(word_to_idx[tuple(new_w)])
        cover_sets.append(covered)
        
    full_set = set(range(n_words))
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        uncovered = set(full_set)
        codewords = []
        while uncovered:
            best_word = -1
            best_count = -1
            candidates = []
            for i in range(n_words):
                count = len(cover_sets[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = i
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            if not candidates:
                break
            if len(candidates) > 1:
                best_word = random.choice(candidates)
            codewords.append(best_word)
            uncovered -= cover_sets[best_word]
        return codewords

    def local_search(codewords):
        counts = [0] * n_words
        for cw in codewords:
            for idx in cover_sets[cw]:
                counts[idx] += 1
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords)):
                cw = codewords[i]
                can_remove = True
                for idx in cover_sets[cw]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for idx in cover_sets[cw]:
                        counts[idx] -= 1
                    codewords.pop(i)
                    improved = True
                    break
        return codewords

    best_codewords = None
    best_size = float('inf')
    start_time = time.time()
    timeout = 110
    num_restarts = 0
    while time.time() - start_time < timeout:
        seed = num_restarts
        cw = greedy_cover(seed)
        cw = local_search(cw)
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            if best_size <= 73:
                break
        num_restarts += 1
        
    if best_codewords is None:
        best_codewords = greedy_cover(0)
        best_codewords = local_search(best_codewords)
        
    return [list(all_words[i]) for i in best_codewords]

def _generic_greedy_cover(q, n, radius):
    """Generic greedy set cover for arbitrary q, n, radius."""
    alphabet = range(q)
    all_words = list(itertools.product(alphabet, repeat=n))
    n_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    covered_sets = []
    for w in all_words:
        covered = set()
        for candidate in all_words:
            dist = sum(1 for a, b in zip(w, candidate) if a != b)
            if dist <= radius:
                covered.add(word_to_idx[candidate])
        covered_sets.append(covered)
        
    uncovered = set(range(n_words))
    codewords = []
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(n_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
        codewords.append(best_word)
        uncovered -= covered_sets[best_word]
        
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            temp_codewords = codewords[:i] + codewords[i+1:]
            covered = set()
            for cw in temp_codewords:
                covered |= covered_sets[cw]
            if covered == set(range(n_words)):
                codewords = temp_codewords
                improved = True
                break
                
    return [list(all_words[i]) for i in codewords]
# EVOLVE-BLOCK-END
