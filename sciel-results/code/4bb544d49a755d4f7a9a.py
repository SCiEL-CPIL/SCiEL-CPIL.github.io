import time
import math
import numpy as np

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Check Paley tournaments for primes p = 3 mod 4, in increasing order
    for p in range(5, 91):
        if time.time() > deadline:
            break
        if p % 4 != 3:
            continue
        if not is_prime(p):
            continue
            
        # Construct Paley tournament on p vertices
        # i -> j iff (j - i) % p is a quadratic residue
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        # Precompute out-neighborhoods as bitmasks (numpy uint64 for p <= 64, use Python int for larger)
        # For p up to 90, we need arbitrary precision, so use Python ints but leverage numpy for vectorization
        N = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    mask |= (1 << j)
            N[i] = mask
        
        # Precompute ~N[l] & mask for all l, where mask covers bits 0..p-1
        full_mask = (1 << p) - 1
        not_N = [full_mask ^ N[l] for l in range(p)]
        
        # Verify condition: Every 4 vertices have a common dominator
        # For each 4-tuple (i,j,k,l) with i<j<k<l, need N[i] & N[j] & N[k] & N[l] != 0
        # Equivalently, for each triple (i,j,k), common_ijk = N[i]&N[j]&N[k] must have 
        # (common_ijk & N[l]) != 0 for all l > k
        # i.e., common_ijk must NOT be a subset of not_N[l] for any l > k
        # i.e., common_ijk & not_N[l] != 0 is NOT required; we need common_ijk & N[l] != 0
        # which is equivalent to: common_ijk is NOT contained in not_N[l]
        # Actually: common_ijk & N[l] != 0 means there's some vertex in common_ijk that also has an edge from l... 
        # Wait, no. N[l] is the set of vertices that l points TO. We need a vertex v such that 
        # v is dominated by i, j, k, AND l. So v must be in N[i] ∩ N[j] ∩ N[k] ∩ N[l].
        # So we need N[i] & N[j] & N[k] & N[l] != 0.
        
        valid = True
        for i in range(p - 3):
            if time.time() > deadline:
                valid = False
                break
            Ni = N[i]
            for j in range(i + 1, p - 2):
                common_ij = Ni & N[j]
                if common_ij == 0:
                    continue
                for k in range(j + 1, p - 1):
                    common_ijk = common_ij & N[k]
                    if common_ijk == 0:
                        continue
                    # Check all l > k: need common_ijk & N[l] != 0
                    # Vectorize: for l in range(k+1, p), check if (common_ijk & N[l]) == 0
                    # Use numpy for the inner loop
                    for l in range(k + 1, p):
                        if (common_ijk & N[l]) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # 0 means i -> j, 1 means j -> i
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback: return p=67 Paley tournament (known valid)
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
