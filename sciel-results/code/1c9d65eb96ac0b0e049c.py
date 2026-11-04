<<<<<<< SEARCH
import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)
=======
import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use numpy for faster sum/diff computation for larger sets
    if n > 10:
        arr = np.array(U, dtype=np.int64)
        # Sums: all pairwise sums (i <= j to avoid duplicates, but set handles it)
        sums_arr = arr[:, None] + arr[None, :]
        sums_set = set(sums_arr.flatten().tolist())
        # Diffs: all pairwise differences
        diffs_arr = arr[:, None] - arr[None, :]
        diffs_set = set(diffs_arr.flatten().tolist())
    else:
        sums = set()
        diffs = set()
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                sums.add(ui + U[j])
            for j in range(i + 1, n):
                d = U[j] - ui
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        sums_set = sums
        diffs_set = diffs
    
    sum_count = len(sums_set)
    diff_count = len(diffs_set)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)
>>>>>>> REPLACE

