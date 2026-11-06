import time
import itertools
import random

N = 40

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s limit

    # Universe of cycle lengths to cover: 3..39 (length 40 is the rim)
    # We model this as a set cover problem.
    # Elements: Lengths 3..39
    # Sets: 
    #   1. Single chord (0, k) covers lengths k+1 and N-k+1
    #   2. Pair of chords (0, u), (0, v) covers length |u-v|+2
    #   3. (Optional) Other chord structures could be added, but the fan structure (0,k) 
    #      allows us to easily verify cycles. 
    #      Actually, the cycle formed by chords (0,u) and (0,v) is 0-u-...-v-0.
    #      This uses edges (0,u), (0,v) and rim edges u..v.
    #      This is valid as long as u, v are on the rim.
    #      The length is (v-u) + 2 if v > u.
    #      So the "set" for a pair {u, v} is { |u-v|+2 }.
    #      Note: A single chord (0,k) creates a cycle 0-1-...-k-0 of length k+1 
    #            AND 0-(N-1)-...-k-0 of length N-k+1.
    #            So a single chord is a set of size 2 (usually).
    #            A pair is a set of size 1.
    #            
    #      We want to select a set of chords S (endpoints) such that the union of 
    #      their coverage sets covers all lengths 3..39.
    #      Minimize |S|.
    #      Total edges = N (rim) + |S| (chords).
    #      Score = N + |S| - N = |S|.

    # Candidates for chord endpoints: 2..38
    # Why not 1 or 39? 
    #   k=1: covers 2 (invalid, need >=3) and 40 (rim, already covered). So k=1 is useless for 3..39.
    #   k=39: covers 40 (rim) and 2 (invalid). Useless.
    #   So candidates are 2..38.
    
    candidates = list(range(2, N - 1))  # 2 to 38
    num_candidates = len(candidates)
    
    # Precompute coverage masks for each candidate
    # Bitmask for lengths 3..39. 
    # We can use a Python integer as a bitmask.
    # Bit i corresponds to length i.
    
    def make_mask(lengths):
        m = 0
        for l in lengths:
            if 3 <= l <= 39:
                m |= (1 << l)
        return m

    # single_cover[i] = bitmask of lengths covered by chord i alone
    single_cover = [0] * num_candidates
    for i, k in enumerate(candidates):
        l1 = k + 1
        l2 = N - k + 1
        single_cover[i] = make_mask([l1, l2])
    
    # pair_cover[i][j] = bitmask of length covered by pair (i, j)
    # Note: pair (i, j) covers exactly one length: abs(candidates[i] - candidates[j]) + 2
    pair_cover = [[0] * num_candidates for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            l = abs(candidates[i] - candidates[j]) + 2
            if 3 <= l <= 39:
                pair_cover[i][j] = (1 << l)
                pair_cover[j][i] = (1 << l)
    
    # Target mask: all lengths 3..39
    target_mask = 0
    for l in range(3, 40):
        target_mask |= (1 << l)

    def check_coverage_fast(idx_set):
        """
        Check if the set of chord indices idx_set covers all lengths 3..39.
        idx_set is a set of indices into 'candidates'.
        """
        mask = 0
        # Add single chord coverages
        for i in idx_set:
            mask |= single_cover[i]
        
        # Add pair coverages
        idx_list = sorted(idx_set)
        for a_idx in range(len(idx_list)):
            for b_idx in range(a_idx + 1, len(idx_list)):
                a, b = idx_list[a_idx], idx_list[b_idx]
                mask |= pair_cover[a][b]
        
        return (mask & target_mask) == target_mask

    def build_solution(S):
        """
        Build edges and cycles from chord set S (list of actual vertex indices).
        S is sorted.
        """
        edges = []
        # Rim edges
        for i in range(N):
            edges.append((i, (i + 1) % N))
        # Chord edges from 0
        for k in S:
            edges.append((0, k))
        
        # Map each length 3..39 to a cycle construction
        # Length 40 is the rim.
        length_map = {}
        length_map[40] = ('rim',)
        
        # Single chords
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map and 3 <= l1 <= 39:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map and 3 <= l2 <= 39:
                length_map[l2] = ('chord_long', k)
        
        # Pairs
        for u, v in itertools.combinations(S, 2):
            l = abs(u - v) + 2
            if l not in length_map and 3 <= l <= 39:
                length_map[l] = ('chord_pair', u, v)
        
        # Verify all lengths 3..40 are covered
        for L in range(3, N + 1):
            if L not in length_map:
                return None
        
        # Generate cycles
        cycles = []
        for L in range(3, N + 1):
            entry = length_map[L]
            if entry[0] == 'rim':
                cycles.append(list(range(N)))
            elif entry[0] == 'chord_short':
                k = entry[1]
                # Cycle: 0, 1, ..., k
                cycles.append(list(range(0, k + 1)))
            elif entry[0] == 'chord_long':
                k = entry[1]
                # Cycle: 0, N-1, ..., k
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif entry[0] == 'chord_pair':
                u, v = entry[1], entry[2]
                # Cycle: 0, u, u+1, ..., v
                # This assumes u < v. 
                # The cycle is 0 -> u -> u+1 -> ... -> v -> 0.
                # Edges: (0,u), (u, u+1), ..., (v-1, v), (v, 0).
                # All rim edges exist. Chords (0,u) and (0,v) exist.
                cycles.append([0, u] + list(range(u + 1, v + 1)))
        
        return {'edges': edges, 'cycles': cycles}

    random.seed(42)
    
    best_solution = None
    best_m = 999
    
    # Greedy Set Cover Search
    # We want to find the smallest set S.
    # We will try to find a solution with m chords.
    # Since the problem is small (37 candidates), we can use a heuristic search.
    
    # Strategy:
    # 1. Try m = 7, 8, 9...
    # 2. For each m, use a randomized greedy approach.
    #    - Start with empty set.
    #    - At each step, pick the chord that covers the most *new* lengths.
    #    - Add randomness to escape local minima.
    
    def greedy_search(m, num_restarts=5000):
        for _ in range(num_restarts):
            if time.time() > deadline:
                return None
            
            current_set = set()
            current_mask = 0
            
            # Randomly initialize with a few chords? Or start empty?
            # Starting empty is standard greedy.
            
            while len(current_set) < m:
                # Find best chord to add
                best_chord = None
                best_gain = -1
                best_mask = 0
                
                # Candidate chords not in current_set
                available = [i for i in range(num_candidates) if i not in current_set]
                
                if not available:
                    break
                
                # Evaluate gain for each available chord
                # Gain = (current_mask | single_cover[i] | pair_cover_with_existing) & ~current_mask
                # Actually, we need to compute the new mask if we add chord i.
                # new_mask = current_mask | single_cover[i]
                # for j in current_set:
                #     new_mask |= pair_cover[i][j]
                
                # To speed this up, we can just iterate.
                # 37 candidates * 7 steps * 5000 restarts = ~1.3M operations. Very fast.
                
                for i in available:
                    new_mask = current_mask | single_cover[i]
                    for j in current_set:
                        new_mask |= pair_cover[i][j]
                    
                    # Count bits in (new_mask & ~current_mask)
                    gain = bin(new_mask & ~current_mask).count('1')
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_chord = i
                        best_mask = new_mask
                
                if best_chord is None or best_gain == 0:
                    # No improvement possible with remaining slots?
                    # If we haven't reached m, we must pick something to fill the slot,
                    # but if gain is 0, it's useless.
                    # However, if we are forced to pick m chords, and we pick useless ones,
                    # it doesn't help coverage.
                    # If we haven't covered everything by m chords, this restart fails.
                    # If we have covered everything, we can stop early?
                    # No, the score is m. If we cover with < m, we found a solution with < m chords.
                    # But we are searching for a fixed m.
                    # Actually, if we cover everything with k < m chords, we should return that solution.
                    # Let's check coverage.
                    if (current_mask & target_mask) == target_mask:
                        return current_set.copy()
                    
                    # If gain is 0, we are stuck. Pick a random one to fill the slot?
                    # It won't help coverage, so this path is likely a failure for covering all lengths.
                    # Break and fail this restart.
                    break
                
                current_set.add(best_chord)
                current_mask = best_mask
                
                # Early termination: if we covered everything, we can stop adding chords.
                # But we want exactly m? No, we want *at most* m.
                # The score is the number of chords.
                # So if we cover everything with k chords, score is k.
                # We should return the set of k chords.
                if (current_mask & target_mask) == target_mask:
                    return current_set.copy()
            
            # After m steps (or early break), check if covered
            if (current_mask & target_mask) == target_mask:
                return current_set.copy()
        
        return None

    # Try m from 7 upwards
    for m in range(7, 15):
        if time.time() > deadline:
            break
        
        # Adjust restarts based on time
        # If m is small, greedy is more likely to succeed? Or harder?
        # Usually smaller m is harder to find.
        
        found = False
        # Use more restarts for smaller m
        restarts = 20000 if m < 9 else 5000
        
        result_set = greedy_search(m, restarts)
        
        if result_set:
            S = [candidates[i] for i in result_set]
            S.sort()
            result = build_solution(S)
            if result:
                best_solution = result
                best_m = len(S)
                found = True
                break
        
        # If greedy fails, try some random sampling for this m
        # This helps if greedy gets stuck in a local minimum
        if not found:
            max_attempts = 50000
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                idx_set = set(random.sample(range(num_candidates), m))
                if check_coverage_fast(idx_set):
                    S = [candidates[i] for i in idx_set]
                    S.sort()
                    result = build_solution(S)
                    if result:
                        best_solution = result
                        best_m = m
                        found = True
                        break
    
    if best_solution:
        return best_solution
    
    # Fallback: fan graph (always valid)
    # This has 39 chords. Score 39.
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))
    
    return {'edges': edges, 'cycles': cycles}
