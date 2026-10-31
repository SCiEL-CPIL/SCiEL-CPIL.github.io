import time
import math
import random

def score_of(U):
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -float('inf')
    max_u = max(U)
    if max_u == 0:
        return -float('inf')
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0 or n_diff <= 0:
        return -float('inf')
    ratio = n_diff / n_sum
    if ratio <= 0:
        return -float('inf')
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -float('inf')
    return 1 + math.log(ratio) / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    best_U = [0, 1, 3]
    best_score = score_of(best_U)
    
    # Initial candidates
    candidates = []
    for base in [2, 3, 4, 5, 6]:
        U = [0]
        val = 1
        while val < 100000 and len(U) < 18:
            U.append(val)
            val *= base
        candidates.append(list(U))
    
    for k in range(2, 50):
        for n_pairs in range(2, min(12, 20000 // k + 1)):
            U = [0, 1]
            for i in range(1, n_pairs):
                U.append(i * k)
                U.append(i * k + 1)
            if max(U) < 100000:
                candidates.append(U)
    
    random.seed(42)
    for _ in range(300):
        n = random.randint(3, 14)
        U = [0]
        cur = random.randint(1, 5)
        for _ in range(n - 1):
            cur += random.randint(1, max(1, cur // 3 + 1))
            if cur < 100000:
                U.append(cur)
        if len(U) >= 3:
            candidates.append(U)
    
    for U in candidates:
        s = score_of(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    def local_search(U, max_val, iters=300):
        U = sorted(list(set(U)))
        cs = score_of(U)
        for _ in range(iters):
            if time.time() > deadline - 5:
                break
            r = random.random()
            if r < 0.25 and len(U) < 16:
                nv = random.randint(1, max_val)
                if nv not in U:
                    nU = sorted(U + [nv])
                    s = score_of(nU)
                    if s > cs:
                        cs, U = s, nU
            elif r < 0.45 and len(U) > 3:
                others = [x for x in U if x != 0]
                if others:
                    rem = random.choice(others)
                    nU = [x for x in U if x != rem]
                    s = score_of(nU)
                    if s > cs:
                        cs, U = s, nU
            elif r < 0.85:
                others = [x for x in U if x != 0]
                if others:
                    old = random.choice(others)
                    deltas = random.sample([-10,-5,-3,-2,-1,1,2,3,5,10,20], min(4,10))
                    for d in deltas:
                        nv = old + d
                        if 0 < nv <= max_val and nv not in U:
                            nU = sorted([x if x != old else nv for x in U])
                            s = score_of(nU)
                            if s > cs:
                                cs, U = s, nU
                                break
            else:
                others = [x for x in U if x != 0]
                if len(others) >= 2:
                    i, j = random.sample(range(len(others)), 2)
                    nU = list(U)
                    ii = nU.index(others[i])
                    ij = nU.index(others[j])
                    nU[ii], nU[ij] = nU[ij], nU[ii]
                    nU = sorted(nU)
                    if len(set(nU)) == len(nU):
                        s = score_of(nU)
                        if s > cs:
                            cs, U = s, nU
        return U, cs
    
    max_val = max(200, max(best_U) * 3)
    
    while time.time() < deadline - 8:
        nU, ns = local_search(best_U, max_val, 200)
        if ns > best_score:
            best_score, best_U = ns, nU
            max_val = max(max_val, max(best_U) * 2 + 50)
        elif random.random() < 0.2:
            U_list = list(best_U)
            others = [x for x in U_list if x != 0]
            if others and len(U_list) > 3:
                U_list.remove(random.choice(others))
                add = random.randint(1, max_val)
                if add not in U_list:
                    U_list = sorted(U_list + [add])
                    s = score_of(U_list)
                    if s > best_score:
                        best_score, best_U = s, U_list
    return best_U
