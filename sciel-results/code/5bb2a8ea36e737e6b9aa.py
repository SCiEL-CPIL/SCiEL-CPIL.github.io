import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
    
    # Precompute: for each chord (i,j) with 0 < i < j < N, what cycle lengths does it create?
    # A chord (i,j) creates cycles of length (j-i+1) and (N-j+i+1)
    # Also, pairs of chords can create additional cycles
    
    # For the fan topology (chords from vertex 0 to k), chord (0,k) creates:
    # - cycle of length k+1: 0,1,2,...,k
    # - cycle of length N-k+1: 0,k,k-1,...,1
    # Two chords (0,u) and (0,v) with u<v create cycle of length v-u+2: 0,u,u+1,...,v
    
    def get_lengths_fan(S):
        """Get all cycle lengths achievable with fan chords S (chords from 0 to k for k in S)"""
        lengths = {N}  # rim
        for k in S:
            lengths.add(k + 1)       # short cycle
            lengths.add(N - k + 1)    # long cycle
        for u, v in itertools.combinations(S, 2):
            lengths.add(abs(u - v) + 2)  # chord pair cycle
        return lengths

    def is_pancyclic_fan(S):
        lengths = get_lengths_fan(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def build_result_fan(S):
        """Build result from fan chord set S"""
        edges = []
        # Rim
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chords from 0 to k
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

    # DLX-based exact cover search for minimum chords
    # Universe: cycle lengths 3..39 (37 elements)
    # Each candidate chord k (2 <= k <= 38) covers:
    #   - length k+1
    #   - length N-k+1  
    #   - plus we need to handle chord-pair cycles separately
    
    # Actually, let's use a smarter approach: 
    # 1. First find minimum S using greedy + local search
    # 2. Then try to verify/improve with more systematic search

    def greedy_find_S(m):
        candidates = list(range(2, N - 1))
        
        seeds = [
            [2, 38],
            [2, 3, 37, 38],
            [2, 3, 4, 36, 37, 38],
            [3, 5, 7, 33, 35, 37],
            [2, 4, 6, 34, 36, 38],
            [2, 38],
            [2],
            [38],
        ]
        
        for seed in seeds:
            if not set(seed).issubset(set(candidates)):
                continue
            S = set(seed)
            if len(S) > m:
                continue
            
            while len(S) < m:
                current_lengths = get_lengths_fan(S)
                missing = [L for L in range(3, N + 1) if L not in current_lengths]
                
                if not missing:
                    return S
                
                best_k = None
                best_score = -1
                
                for k in candidates:
                    if k in S:
                        continue
                    new_S = S | {k}
                    new_lengths = get_lengths_fan(new_S)
                    gain = len(new_lengths - current_lengths)
                    
                    if gain > best_score:
                        best_score = gain
                        best_k = k
                
                if best_k is None:
                    break
                S.add(best_k)
            
            if is_pancyclic_fan(S):
                return S
        
        # Random restarts
        rng = random.Random(42)
        for restart in range(3000):
            if time.time() > deadline:
                return None
            
            S = set(rng.sample(candidates, m))
            
            for step in range(300):
                if time.time() > deadline:
                    return None
                if is_pancyclic_fan(S):
                    return S
                
                current_lengths = get_lengths_fan(S)
                missing = [L for L in range(3, N + 1) if L not in current_lengths]
                
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
        
        while time.time() < deadline and attempts < 100000:
            attempts += 1
            S = set(rng.sample(candidates, m))
            if is_pancyclic_fan(S):
                return S
        
        return None

    # Search for minimal m
    for m in [8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        S = find_S(m)
        if S:
            result = build_result_fan(S)
            if result:
                return result

    # Fallback: Fan Graph
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
        
    return {'edges': edges, 'cycles': cycles}
