import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
    
    def get_lengths(S):
        lengths = {N}
        S_list = sorted(S)
        for k in S_list:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        for i in range(len(S_list)):
            ui = S_list[i]
            for j in range(i + 1, len(S_list)):
                vj = S_list[j]
                lengths.add(vj - ui + 2)
        return lengths

    def is_pancyclic(S):
        lengths = get_lengths(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def missing_lengths(S):
        lengths = get_lengths(S)
        return [L for L in range(3, N + 1) if L not in lengths]

    def greedy_find_S(m):
        candidates = list(range(2, N - 1))
        cand_set = set(candidates)
        
        seeds = [
            [2],
            [38],
            [2, 38],
            [3],
            [37],
            [2, 3],
            [2, 37],
            [38, 37],
            [38, 3],
            [2, 3, 37, 38],
            [2, 3, 4, 36, 37, 38],
            [2, 4, 6, 34, 36, 38],
            [3, 5, 7, 33, 35, 37],
            [2, 3, 4, 5, 35, 36, 37, 38],
            [2, 3, 5, 7, 33, 35, 37, 38],
            [2, 4, 6, 8, 32, 34, 36, 38],
            [1, 2, 3, 37, 38, 39],
            [2, 3, 4, 36, 37, 38, 39],
        ]
        
        for seed in seeds:
            if not set(seed).issubset(cand_set):
                continue
            S = set(seed)
            if len(S) > m:
                continue
            
            while len(S) < m:
                missing = missing_lengths(S)
                
                if not missing:
                    return S
                
                best_k = None
                best_score = -1
                best_new_missing = None
                
                for k in candidates:
                    if k in S:
                        continue
                    new_S = S | {k}
                    new_missing = missing_lengths(new_S)
                    gain = len(missing) - len(new_missing)
                    
                    # Rarity bonus: middle lengths are harder to cover
                    score = gain * 10.0
                    covered = set(missing) - set(new_missing)
                    for L in covered:
                        if 15 <= L <= 25:
                            score += 0.5
                        elif 10 <= L <= 30:
                            score += 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_new_missing = new_missing
                
                if best_k is None:
                    break
                S.add(best_k)
            
            if is_pancyclic(S):
                return S
        
        # Random restarts with local search
        rng = random.Random(42)
        for restart in range(8000):
            if time.time() > deadline:
                return None
            
            S = set(rng.sample(candidates, m))
            
            # Local search: try swaps to improve
            for step in range(300):
                missing = missing_lengths(S)
                if not missing:
                    return S
                
                in_s = list(S)
                out_s = [x for x in candidates if x not in S]
                if not in_s or not out_s:
                    break
                
                u = rng.choice(in_s)
                v = rng.choice(out_s)
                S.remove(u)
                S.add(v)
            
            # Also try single-element replacement
            for step in range(150):
                missing = missing_lengths(S)
                if not missing:
                    return S
                
                in_s = list(S)
                out_s = [x for x in candidates if x not in S]
                if not in_s or not out_s:
                    break
                
                u = rng.choice(in_s)
                v = rng.choice(out_s)
                S.remove(u)
                S.add(v)
        
        return None

    def find_S(m):
        S = greedy_find_S(m)
        if S:
            return S
        
        # Fallback: pure random search
        candidates = list(range(2, N - 1))
        rng = random.Random(123)
        attempts = 0
        
        while time.time() < deadline and attempts < 300000:
            attempts += 1
            S = set(rng.sample(candidates, m))
            if is_pancyclic(S):
                return S
        
        return None

    def build_result(S):
        edges = []
        # Rim
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chords
        for k in sorted(S):
            edges.append((0, k))
        
        # Generate cycles
        cycles = []
        length_map = {}
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)
                
        for u, v in itertools.combinations(sorted(S), 2):
            l = abs(u - v) + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', min(u, v), max(u, v))
                
        # Rim
        length_map[N] = ('rim',)
        
        # Construct cycles 3..N
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

    # Search for minimal m
    # Try m=8 first, then m=9, m=10 as fallback
    for m in [7, 8, 9, 10, 11]:
        if time.time() > deadline:
            break
        S = find_S(m)
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
