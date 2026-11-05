"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80
    random.seed(99)
    
    N = 20
    MAX_VAL = 10000
    
    def compute_scores(A):
        sset = set()
        pset = set()
        for i in range(N):
            ai = A[i]
            for j in range(i, N):
                sset.add(ai + A[j])
                pset.add(ai * A[j])
        return len(sset), len(pset)
    
    def objective(A):
        s, p = compute_scores(A)
        return max(s, p)
    
    def local_improve(A, score, time_limit=30):
        """Local search by single-element replacement."""
        t_end = time.time() + time_limit
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for idx in range(N):
                orig = A[idx]
                best_v = None
                best_sc = score
                # Generate candidates
                cands = set()
                for d in range(-30, 31):
                    v = orig + d
                    if 1 <= v <= MAX_VAL:
                        cands.add(v)
                for p in [2, 3, 5, 7, 11]:
                    for e in range(0, 14):
                        v = p ** e
                        if 1 <= v <= MAX_VAL:
                            cands.add(v)
                for other in A:
                    for mult in [2, 3, 4, 5, 6, 8]:
                        v = other * mult
                        if 1 <= v <= MAX_VAL:
                            cands.add(v)
                        v = other // mult if mult > 1 and other % mult == 0 else None
                        if v and 1 <= v <= MAX_VAL:
                            cands.add(v)
                
                for v in cands:
                    if v in A:
                        continue
                    test_A = A[:]
                    test_A[idx] = v
                    sc = objective(test_A)
                    if sc < best_sc:
                        best_sc = sc
                        best_v = v
                if best_v is not None:
                    A[idx] = best_v
                    score = best_sc
                    improved = True
        return A, score
    
    # Generate many structured candidates
    candidates = []
    
    # 2D lattices {p^i * q^j}
    for p in [2, 3, 5]:
        for q in [2, 3, 5, 7]:
            if p == q:
                continue
            for i_max in range(0, 6):
                for j_max in range(0, 12):
                    pts = [(i, j) for i in range(i_max+1) for j in range(j_max+1)]
                    if len(pts) < N:
                        continue
                    vals = [p**i * q**j for i, j in pts]
                    if len(set(vals)) == len(vals) and max(vals) <= MAX_VAL:
                        candidates.append(sorted(vals)[:N])
    
    # {2^j, k*2^j} for various k
    for k in [3, 5, 7, 9, 11, 13, 15]:
        cand = sorted(set([2**j for j in range(10)] + [k*(2**j) for j in range(10)]))
        if len(cand) == N and max(cand) <= MAX_VAL:
            candidates.append(cand)
    
    # {2^i * 3^j} with rectangular grids
    for im in range(1, 5):
        for jm in range(1, 11):
            if (im+1)*(jm+1) < N:
                continue
            vals = [2**i * 3**j for i in range(im+1) for j in range(jm+1)]
            if len(set(vals)) == len(vals) and max(vals) <= MAX_VAL:
                candidates.append(sorted(vals)[:N])
    
    # Consecutive and arithmetic progressions
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(1, 41, 2)))  # odd numbers 1..39
    candidates.append(list(range(2, 41, 2)))  # even numbers 2..40
    
    # Geometric progressions
    for ratio in [1.5, 1.6, 1.7, 1.8, 2.0]:
        vals = [int(round(ratio**i)) for i in range(N)]
        if len(set(vals)) == N and max(vals) <= MAX_VAL and min(vals) >= 1:
            candidates.append(sorted(vals))
    
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != N or max(cand) > MAX_VAL or min(cand) < 1 or len(set(cand)) != N:
            continue
        sc = objective(cand)
        if sc < best_score:
            best_score = sc
            best_A = cand[:]
    
    if best_A is None:
        best_A = list(range(1, N+1))
        best_score = objective(best_A)
    
    # Local improvement on best candidate
    if time.time() < deadline - 15:
        best_A, best_score = local_improve(best_A, best_score, time_limit=min(25, deadline - time.time() - 15))
    
    # Try local improvement on top-5 candidates
    if time.time() < deadline - 10:
        scored = []
        for cand in candidates:
            if len(cand) != N or max(cand) > MAX_VAL or min(cand) < 1 or len(set(cand)) != N:
                continue
            sc = objective(cand)
            scored.append((sc, cand))
        scored.sort()
        for sc, cand in scored[:5]:
            if time.time() > deadline - 10:
                break
            A2, sc2 = local_improve(cand[:], sc, time_limit=min(10, deadline - time.time() - 10))
            if sc2 < best_score:
                best_score = sc2
                best_A = A2
    
    return best_A
# EVOLVE-BLOCK-END
