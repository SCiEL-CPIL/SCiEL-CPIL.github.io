import time
import itertools
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def get_lengths(S):
        lengths = {N}
        for k in S:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        for u, v in itertools.combinations(S, 2):
            lengths.add(abs(u - v) + 2)
        return lengths

    def is_pancyclic(S):
        lengths = get_lengths(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def greedy_find_S(m, rng=None):
        target_lengths = set(range(3, N))
        candidates = list(range(2, N - 1))
        if rng:
            rng.shuffle(candidates)
        S = set()
        
        for _ in range(m):
            best_k = None
            best_gain = -1
            
            current_lengths = get_lengths(S)
            current_covered = current_lengths.intersection(target_lengths)
            
            for k in candidates:
                if k in S:
                    continue
                new_S = S.union({k})
                new_lengths = get_lengths(new_S)
                new_covered = new_lengths.intersection(target_lengths)
                gain = len(new_covered) - len(current_covered)
                
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
            
            if best_k is None:
                return None
            
            S.add(best_k)
            if len(get_lengths(S).intersection(target_lengths)) == len(target_lengths):
                break
                
        if is_pancyclic(S):
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

    # Try to find minimal m with randomized restarts
    rng = random.Random(42)
    for m in [6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        # Try deterministic first
        S = greedy_find_S(m)
        if S:
            result = build_result(S)
            if result:
                return result
        # Try randomized restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            S = greedy_find_S(m, rng)
            if S:
                result = build_result(S)
                if result:
                    return result

    # Fallback: Fan Graph
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}
