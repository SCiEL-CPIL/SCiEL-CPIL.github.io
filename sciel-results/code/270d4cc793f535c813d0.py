import time
import itertools
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Precompute: for each candidate chord k (2..38), which target lengths it directly covers
    # and which pairwise lengths it can form with other chords
    target_lengths = set(range(3, N))  # 3..39
    candidates = list(range(2, N - 1))  # 2..38

    # Direct lengths from chord k: k+1 and N-k+1
    direct_lengths = {}
    for k in candidates:
        dl = set()
        l1 = k + 1
        l2 = N - k + 1
        if l1 in target_lengths:
            dl.add(l1)
        if l2 in target_lengths:
            dl.add(l2)
        direct_lengths[k] = dl

    # Pairwise lengths: for pair (u,v), length = abs(u-v)+2
    # Precompute for all pairs
    pair_lengths = {}
    for u, v in itertools.combinations(candidates, 2):
        l = abs(u - v) + 2
        if l in target_lengths:
            pair_lengths[(u, v)] = l

    def greedy_find_S(m, rng=None):
        S = set()
        covered = set()  # lengths covered so far
        
        for _ in range(m):
            best_k = None
            best_gain = -1
            best_new_covered = None
            
            for k in candidates:
                if k in S:
                    continue
                # Compute what new lengths this chord would add
                new_covered = set(covered)
                # Direct lengths from k
                for l in direct_lengths[k]:
                    new_covered.add(l)
                # Pairwise lengths with existing chords in S
                for u in S:
                    pair_key = (min(k, u), max(k, u))
                    if pair_key in pair_lengths:
                        new_covered.add(pair_lengths[pair_key])
                
                gain = len(new_covered) - len(covered)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
                    best_new_covered = new_covered
            
            if best_k is None:
                return None
            
            S.add(best_k)
            covered = best_new_covered
            
            if len(covered) == len(target_lengths):
                break
        
        if len(covered) == len(target_lengths):
            return S
        return None

    def build_result(S):
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
        for k in S:
            edges.append((0, k))
        
        cycles = []
        length_map = {}
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)
                
        for u, v in itertools.combinations(S, 2):
            l = abs(u - v) + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', min(u, v), max(u, v))
                
        length_map[N] = ('rim',)
        
        for L in range(3, N + 1):
            if L not in length_map:
                return None
            
            typ = length_map[L]
            if typ[0] == 'rim':
                cycles.append(list(range(N)))
            elif typ[0] == 'chord_short':
                k = typ[1]
                cycles.append([0] + list(range(1, k + 1)))
            elif typ[0] == 'chord_long':
                k = typ[1]
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif typ[0] == 'chord_pair':
                u, v = typ[1], typ[2]
                cycles.append([0, u] + list(range(u + 1, v + 1)))
                
        return {'edges': edges, 'cycles': cycles}

    # m must be at least 8 (need 37 distinct lengths, max with m chords is 2m+C(m,2)+1)
    # m=7 gives at most 36, so start at m=8
    rng = random.Random(42)
    for m in [8, 9, 10, 11]:
        if time.time() > deadline:
            break
        # Try deterministic first
        S = greedy_find_S(m)
        if S:
            result = build_result(S)
            if result:
                return result
        # Try randomized restarts
        for attempt in range(20):
            if time.time() > deadline:
                break
            S = greedy_find_S(m, rng)
            if S:
                result = build_result(S)
                if result:
                    return result

    # Fallback: use a known valid construction with more chords
    # Use all chords that give useful lengths
    S_fallback = set(range(2, N - 1))  # all chords 2..38
    result = build_result(S_fallback)
    if result:
        return result

    # Last resort: Fan Graph
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}
