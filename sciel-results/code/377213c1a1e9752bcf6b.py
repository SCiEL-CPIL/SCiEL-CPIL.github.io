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

    def greedy_find_S(m):
        # Heuristic search
        # Start with anchors that cover extremes
        # Length 3 needs |u-v|=1 or k=2 (k+1=3)
        # Length 39 needs k=2 (N-k+1=39) or |u-v|=37
        # So 2 is a very good anchor. 38 is symmetric.
        
        candidates = list(range(2, N - 1))
        
        # Try a few different starting seeds
        seeds = [
            {2, 38},
            {2, 3, 37, 38},
            {2, 3, 4, 36, 37, 38},
            set(random.sample(candidates, min(m, 6)))
        ]
        
        for seed in seeds:
            if not seed.issubset(set(candidates)):
                continue
            S = set(seed)
            if len(S) > m:
                continue
            
            # Greedy addition
            while len(S) < m:
                best_k = None
                best_gain = -1
                current_lengths = get_lengths(S)
                
                for k in candidates:
                    if k in S:
                        continue
                    new_S = S | {k}
                    new_lengths = get_lengths(new_S)
                    gain = len(new_lengths - current_lengths)
                    if gain > best_gain:
                        best_gain = gain
                        best_k = k
                        # If we cover all, we can stop early?
                        # But we must add exactly m chords? No, we want size m.
                        # If we cover all with fewer, we can stop?
                        # The problem asks for minimizing edges.
                        # If we find a valid set of size < m, that's even better.
                        # But we are searching for specific m.
                        # If m=8, and we find valid set of size 6, we should return that.
                        # Let's check validity inside the loop.
                        if is_pancyclic(new_S):
                            return new_S
                
                if best_k is None:
                    break
                S.add(best_k)
            
            if is_pancyclic(S):
                return S
        
        # Local Search / Random Restarts
        # If greedy failed, try random restarts with local improvement
        for _ in range(2000):
            if time.time() > deadline:
                return None
            
            # Start with random set of size m
            S = set(random.sample(candidates, m))
            
            # Local search: swap elements to increase coverage
            # We just want *any* valid set.
            # Let's try to fix the "missing" lengths.
            # Actually, just random swaps and check.
            for _ in range(100):
                if is_pancyclic(S):
                    return S
                
                # Pick random element in S and random element not in S
                if not S:
                    break
                in_s = list(S)
                out_s = [x for x in candidates if x not in S]
                if not out_s:
                    break
                u = random.choice(in_s)
                v = random.choice(out_s)
                S.remove(u)
                S.add(v)
        
        return None

    def find_S(m):
        # First try the specialized greedy/local search
        S = greedy_find_S(m)
        if S:
            return S
        
        # Fallback to pure random if greedy fails (unlikely for m=8)
        candidates = list(range(2, N - 1))
        attempts = 0
        max_attempts = 100000
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
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
                cycles.append([0] + list(range(1, k + 1)))
            elif typ[0] == 'chord_long':
                k = typ[1]
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif typ[0] == 'chord_pair':
                u, v = typ[1], typ[2]
                cycles.append([0, u] + list(range(u + 1, v + 1)))
                
        return {'edges': edges, 'cycles': cycles}

    # Search for minimal m
    # m=7 is impossible.
    # Try m=8 first.
    for m in [8, 9, 10]:
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
