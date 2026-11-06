import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin

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

    def greedy_find_S(m, seed=None):
        candidates = list(range(2, N - 1))
        cand_set = set(candidates)
        
        if seed is None:
            # Try a few deterministic seeds first
            seeds = [
                {2},
                {2, 3},
                {2, 38},
                {2, 3, 37, 38},
                set(random.sample(candidates, min(m, 6)))
            ]
            for s in seeds:
                if not s.issubset(cand_set):
                    continue
                if len(s) > m:
                    continue
                res = greedy_find_S(m, s)
                if res:
                    return res
            return None

        S = set(seed)
        if len(S) > m:
            return None
        
        # If already valid, return it (even if smaller than m)
        if is_pancyclic(S):
            return S

        # Greedy addition
        while len(S) < m:
            best_k = None
            best_gain = -1
            current_lengths = get_lengths(S)
            missing = set(range(3, N + 1)) - current_lengths
            
            for k in candidates:
                if k in S:
                    continue
                new_S = S | {k}
                new_lengths = get_lengths(new_S)
                new_missing = set(range(3, N + 1)) - new_lengths
                gain = len(missing - new_missing)
                
                # Tie-breaker: prefer smaller k? Or random?
                # Prefer k that covers more, then smaller k for determinism/stability
                if gain > best_gain or (gain == best_gain and best_k is not None and k < best_k):
                    best_gain = gain
                    best_k = k
                    
            if best_k is None:
                break
            
            S.add(best_k)
            
            if is_pancyclic(S):
                return S
        
        if is_pancyclic(S):
            return S
        return None

    def local_search(S, m, max_iter=500):
        candidates = list(range(2, N - 1))
        cand_set = set(candidates)
        S = set(S)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return S if is_pancyclic(S) else None
            
            if is_pancyclic(S):
                return S
            
            # Try swapping one element
            in_s = list(S)
            out_s = [x for x in candidates if x not in S]
            
            if not in_s or not out_s:
                break
                
            # Pick a random element to remove
            u = random.choice(in_s)
            # Pick a random element to add
            v = random.choice(out_s)
            
            S.remove(u)
            S.add(v)
            
        if is_pancyclic(S):
            return S
        return None

    def find_S(m):
        # 1. Try greedy with various seeds
        S = greedy_find_S(m)
        if S:
            return S
        
        # 2. Try random restarts with local search
        candidates = list(range(2, N - 1))
        attempts = 0
        max_attempts = 2000
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            # Start with a random set of size m
            S = set(random.sample(candidates, m))
            
            # Run local search
            S = local_search(S, m, max_iter=100)
            if S:
                return S
        
        # 3. Pure random fallback
        while time.time() < deadline:
            S = set(random.sample(candidates, m))
            if is_pancyclic(S):
                return S
        
        return None

    def build_result(S):
        edges = []
        # Rim
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chords
        for k in S:
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
                
        for u, v in itertools.combinations(S, 2):
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
                # 0, 1, ..., k
                cycles.append([0] + list(range(1, k + 1)))
            elif typ[0] == 'chord_long':
                k = typ[1]
                # 0, N-1, ..., k
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif typ[0] == 'chord_pair':
                u, v = typ[1], typ[2]
                # 0, u, u+1, ..., v
                cycles.append([0, u] + list(range(u + 1, v + 1)))
                
        return {'edges': edges, 'cycles': cycles}

    # Search for minimal m
    # m=7 is theoretically impossible for this structure? 
    # Let's try 7, 8, 9, 10
    for m in [7, 8, 9, 10]:
        if time.time() > deadline:
            break
        S = find_S(m)
        if S:
            result = build_result(S)
            if result:
                return result

    # Fallback: Fan Graph
    # Hub 0, Path 1..39
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        # Cycle 0, 1, ..., L-1, 0?
        # Vertices: 0, 1, ..., L-1.
        # Edges: (0,1), (1,2)...(L-2, L-1), (L-1, 0).
        # (L-1, 0) is a spoke. Exists.
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}
