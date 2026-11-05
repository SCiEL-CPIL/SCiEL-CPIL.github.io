import itertools
import random
import time
import math

N = 16
N2 = N * N
ALL_MASK = (1 << N2) - 1

# Precompute point coordinates
points = list(itertools.product(range(N), repeat=2))

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute iso_mask[u][v] as bitset of k such that u, v, k is isosceles
# Note: iso_mask[u][v] includes u and v themselves? 
# If k == u, d(u,v) = d(v,u) is trivially true? 
# The definition of isosceles triangle usually requires 3 distinct points.
# However, if we allow k==u, then d(u,v) == d(u,u)=0? No.
# Let's stick to distinct points for the triangle definition.
# But for the "blocking" logic: 
# If we add p, and p forms an isosceles triangle with u, v in S, it's invalid.
# The triangle is {p, u, v}. p, u, v must be distinct.
# So iso_mask[u][v] should contain k such that k != u, k != v, and {u, v, k} is isosceles.

iso_mask = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    for j in range(N2):
        if i == j:
            continue
        mask = 0
        d_ij = dist2[i][j]
        for k in range(N2):
            if k == i or k == j:
                continue
            d_ik = dist2[i][k]
            d_jk = dist2[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        iso_mask[i][j] = mask

def popcount(x):
    return bin(x).count('1')

def is_no_isosceles(S):
    """
    Check if S (bitmask) contains no isosceles triangle.
    """
    # Extract indices
    # Optimization: If S is small, just iterate
    idxs = []
    temp = S
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        idxs.append(idx)
        temp ^= lsb
        
    n = len(idxs)
    for i in range(n):
        u = idxs[i]
        for j in range(i + 1, n):
            v = idxs[j]
            # Check if any point in S is in iso_mask[u][v]
            # We exclude u and v from iso_mask, so we just check intersection
            if (S & iso_mask[u][v]) != 0:
                return False
    return True

def get_cover(S):
    """
    Compute the union of iso_mask[u][v] for all pairs u, v in S.
    This represents points that CANNOT be added to S without forming an isosceles triangle with some pair in S.
    """
    idxs = []
    temp = S
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        idxs.append(idx)
        temp ^= lsb
        
    n = len(idxs)
    cover = 0
    for i in range(n):
        u = idxs[i]
        for j in range(i + 1, n):
            v = idxs[j]
            cover |= iso_mask[u][v]
    return cover

def is_maximal(S):
    """
    Check if S is maximal.
    S is maximal if no point outside S can be added.
    A point p can be added if p is not in S and p does not form an iso with any pair in S.
    i.e., p is NOT in get_cover(S).
    So S is maximal if (get_cover(S) | S) == ALL_MASK.
    """
    cover = get_cover(S)
    return (cover | S) == ALL_MASK

def evaluate(S):
    """
    Return a score. Lower is better.
    If S is not no-isosceles, return large penalty.
    If S is no-isosceles but not maximal, return size + penalty * (uncovered points).
    If S is no-isosceles and maximal, return size.
    """
    if not is_no_isosceles(S):
        return 10000 # Large penalty
    
    cover = get_cover(S)
    size = popcount(S)
    
    # Points that can be added are those in ALL_MASK but not in S and not in cover
    uncovered = ALL_MASK & ~(S | cover)
    num_uncovered = popcount(uncovered)
    
    if num_uncovered == 0:
        return size
    else:
        # Penalty encourages maximality
        return size + 10 * num_uncovered

def greedy_construct():
    """
    Build a random no-isosceles set.
    """
    S = 0
    remaining = list(range(N2))
    random.shuffle(remaining)
    
    for p in remaining:
        # Check if p can be added
        valid = True
        # Check against all existing points in S
        temp = S
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            # If adding p creates iso with idx and some other point in S?
            # We need to check if S intersects iso_mask[p][idx]
            if (S & iso_mask[p][idx]) != 0:
                valid = False
                break
            temp ^= lsb
        
        if valid:
            S |= (1 << p)
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_score = 10000
    best_set = 0
    
    # Initial solution
    current_S = greedy_construct()
    current_score = evaluate(current_S)
    
    if current_score < best_score:
        best_score = current_score
        best_set = current_S
        
    T = 50.0 # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    # Simulated Annealing
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Generate neighbor
        r = random.random()
        if r < 0.3:
            # Add a point
            not_in_S = ALL_MASK & ~current_S
            if not_in_S == 0:
                continue
            # Pick random bit from not_in_S
            # Efficient random bit selection
            temp = not_in_S
            # Randomly pick a position
            # This is slow if done naively. 
            # Alternative: pick random index, if not in S, add.
            p = random.randint(0, N2 - 1)
            if (current_S >> p) & 1:
                continue
            new_S = current_S | (1 << p)
        elif r < 0.6:
            # Remove a point
            if current_S == 0:
                continue
            p = random.randint(0, N2 - 1)
            if not ((current_S >> p) & 1):
                continue
            new_S = current_S & ~(1 << p)
        else:
            # Swap
            if current_S == 0:
                continue
            p = random.randint(0, N2 - 1)
            if not ((current_S >> p) & 1):
                continue
            q = random.randint(0, N2 - 1)
            if (current_S >> q) & 1:
                continue
            new_S = (current_S & ~(1 << p)) | (1 << q)
            
        new_score = evaluate(new_S)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_S = new_S
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_S
                
        T *= cooling
        
    # Convert best_set to list of points
    result = []
    temp = best_set
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(points[idx])
        temp ^= lsb
        
    return result

if __name__ == "__main__":
    # Test run
    res = construct()
    print(f"Found set of size {len(res)}")
