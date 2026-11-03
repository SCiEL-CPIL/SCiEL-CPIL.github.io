<<<<<<< SEARCH
import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    if num_sums == 0:
        return -1
        
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)
=======
import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Calculate the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    if num_sums == 0:
        return -1
        
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_bitset(U):
    """
    Faster score computation using bitset operations for sum/diff counting.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    max_u = max(U)
    D = 2 * max_u + 1
    
    # Use bitset for sums
    sum_bitset = 0
    for a in U:
        for b in U:
            if a >= b:
                sum_bitset |= (1 << (a + b))
    num_sums = sum_bitset.bit_count()
    
    # Use bitset for positive differences
    diff_bitset = 0
    for a in U:
        for b in U:
            if b > a:
                diff_bitset |= (1 << (b - a))
    num_pos_diffs = diff_bitset.bit_count()
    num_diffs = 2 * num_pos_diffs + 1
    
    if num_sums == 0:
        return -1
        
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
        
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)
>>>>>>> REPLACE

