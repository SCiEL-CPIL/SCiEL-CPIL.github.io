import math
import time
import numpy as np
from scipy.optimize import minimize

def _jc_distance(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    start_time = time.time()
    deadline = 250.0
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n < 3:
        if n == 2:
            return f"({labels[0]},{labels[1]})"
        elif n == 1:
            return f"{labels[0]}"
        else:
            return ""

    # 1. Compute Distance Matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # 2. NJ Tree Construction
    # Returns a Newick string
    nj_tree_str = _nj_newick(dist_matrix, labels)
    
    # 3. Convert to Tree Structure
    tree = _parse_newick(nj_tree_str)
    
    # 4. Root Tree
    # Root at leaf 0
    rooted_tree = _root_tree(tree, labels[0])
    
    # 5. ML Optimization
    # Convert sequences to array
    seq_lens = len(sequences[labels[0]])
    seqs = np.zeros((n, seq_lens), dtype=np.int8)
    for i, l in enumerate(labels):
        for j, c in enumerate(sequences[l]):
            if c == 'A': seqs[i, j] = 0
            elif c == 'C': seqs[i, j] = 1
            elif c == 'G': seqs[i, j] = 2
            else: seqs[i, j] = 3

    # Optimize branch lengths for initial tree
    best_tree = rooted_tree
    best_ll = _optimize_branch_lengths(best_tree, seqs)
    
    # NNI Search
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Generate NNI moves
        nni_moves = _get_nni_moves(best_tree)
        for move in nni_moves:
            new_tree = _apply_nni(best_tree, move)
            new_ll = _optimize_branch_lengths(new_tree, seqs)
            if new_ll > best_ll:
                best_tree = new_tree
                best_ll = new_ll
                improved = True
                break # Restart NNI scan from new best tree
        
    # 6. Return Newick
    return _tree_to_newick(best_tree)
